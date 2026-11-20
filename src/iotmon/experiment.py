"""Block: experiment runner.

One run = clean start -> scripted impairment phases -> recovery, with N
emulated devices talking to the broker THROUGH the fault injector. Every phase
change is written to the `impairments` table (ground truth).

The collector must already be running in another terminal:
    python -m iotmon.collector --quiet

Single run:
    python -m iotmon.experiment run --run-id r01 --profile mixed --seed 1 --qos 0
Many runs (4 at a time, each on its own proxy ports):
    python -m iotmon.experiment batch --plan experiments/plan.yaml --parallel 4
"""
import argparse
import asyncio
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from psycopg.types.json import Jsonb

from . import db
from .config import ROOT, settings
from .impair import netem
from .impair.proxy import ImpairmentProxy
from .impair.schedule import generate, load_yaml, to_yaml
from .windows import compute_and_store

RUNS_DIR = ROOT / "experiments" / "runs"


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip() or None
    except Exception:
        return None


async def run_experiment(a) -> None:
    phases = load_yaml(a.schedule) if a.schedule else generate(a.profile, a.seed, a.duration)
    total = sum(p.duration_s for p in phases)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / f"{a.run_id}.yaml").write_text(to_yaml(phases))
    config = {"profile": a.profile, "qos": a.qos, "devices": a.devices, "interval_s": a.interval,
              "backend": a.backend, "duration_s": total, "schedule_file": f"experiments/runs/{a.run_id}.yaml"}

    conn = db.connect()
    db.init_schema(conn)
    conn.execute("DELETE FROM runs WHERE run_id = %s AND NOT EXISTS (SELECT 1 FROM messages WHERE run_id = %s)",
                 (a.run_id, a.run_id))
    conn.execute("INSERT INTO runs (run_id, config, seed, git_commit) VALUES (%s,%s,%s,%s)",
                 (a.run_id, Jsonb(config), a.seed, git_commit()))

    proxy = None
    if a.backend == "proxy":
        proxy = ImpairmentProxy(a.tcp_port, (settings.mqtt_host, settings.mqtt_port),
                                a.udp_port, ("127.0.0.1", settings.udp_probe_port), seed=a.seed)
        await proxy.start()
        broker_port, udp_port = a.tcp_port, a.udp_port
    else:
        netem.setup()
        broker_port, udp_port = netem.IMPAIRED_TCP_PORT, settings.udp_probe_port

    emu = subprocess.Popen([sys.executable, "-m", "iotmon.emulator", "--run-id", a.run_id,
                            "--devices", str(a.devices), "--qos", str(a.qos), "--interval", str(a.interval),
                            "--broker-port", str(broker_port), "--udp-port", str(udp_port),
                            "--keepalive", str(a.keepalive), "--duration", str(total),
                            "--boot-id", str(a.seed * 100 + a.qos)],
                           cwd=ROOT, stdout=subprocess.DEVNULL if a.quiet else None)
    print(f"[experiment] {a.run_id}: {len(phases)} phases, {total:.0f} s, backend={a.backend}")
    try:
        for p in phases:
            if proxy:
                proxy.set_params(**p.params)
            else:
                netem.apply(p.params)
            row = conn.execute("INSERT INTO impairments (run_id, phase, start_at, params) VALUES (%s,%s,%s,%s)"
                               " RETURNING id", (a.run_id, p.name, datetime.now(timezone.utc), Jsonb(p.params))).fetchone()
            if not a.quiet:
                print(f"[experiment] {a.run_id}: {p.name:13s} {p.duration_s:5.1f}s {json.dumps(p.params)}")
            await asyncio.sleep(p.duration_s)
            conn.execute("UPDATE impairments SET end_at = %s WHERE id = %s", (datetime.now(timezone.utc), row[0]))
    finally:
        if proxy:
            proxy.set_params()
        else:
            netem.teardown()
        while emu.poll() is None:
            await asyncio.sleep(0.5)
        if proxy:
            await asyncio.sleep(1)
            await proxy.stop()
            config["proxy_stats"] = proxy.stats
        await asyncio.sleep(3)                       # let the collector flush its last rows
        conn.execute("UPDATE runs SET ended_at = now(), config = %s WHERE run_id = %s", (Jsonb(config), a.run_id))
        conn.close()
    n_obs, n_truth = compute_and_store(a.run_id)
    print(f"[experiment] {a.run_id}: done; {n_obs} observed / {n_truth} truth windows")


def run_batch(a) -> None:
    if a.backend == "netem" and a.parallel > 1:
        sys.exit("netem impairs one shared port: use --parallel 1 with --backend netem")
    plan = yaml.safe_load(Path(a.plan).read_text())
    common = plan.get("common", {})
    runs = plan["runs"]
    pending = list(enumerate(runs))
    active: list = []
    slot_ports = [(18830 + i, 19990 + i) for i in range(a.parallel)]
    free = list(range(a.parallel))
    t0 = time.time()
    while pending or active:
        while pending and free:
            i, r = pending.pop(0)
            slot = free.pop(0)
            spec = {**common, **r}
            cmd = [sys.executable, "-m", "iotmon.experiment", "run", "--quiet",
                   "--run-id", spec["run_id"], "--profile", spec.get("profile", "mixed"),
                   "--seed", str(spec["seed"]), "--qos", str(spec.get("qos", 0)),
                   "--devices", str(spec.get("devices", 10)), "--interval", str(spec.get("interval_s", 0.5)),
                   "--duration", str(spec.get("duration_s", 420)),
                   "--tcp-port", str(slot_ports[slot][0]), "--udp-port", str(slot_ports[slot][1]),
                   "--backend", a.backend]
            print(f"[batch] start {spec['run_id']} (slot {slot})", flush=True)
            active.append((subprocess.Popen(cmd, cwd=ROOT), slot, spec["run_id"]))
        time.sleep(1)
        for item in list(active):
            proc, slot, rid = item
            if proc.poll() is not None:
                active.remove(item)
                free.append(slot)
                print(f"[batch] finished {rid} (exit {proc.returncode}) after {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Run fault-injection experiments")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--run-id", required=True)
    r.add_argument("--profile", default="mixed", choices=["mixed", "burst"])
    r.add_argument("--schedule", help="YAML schedule file (overrides --profile)")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--qos", type=int, default=0, choices=[0, 1])
    r.add_argument("--devices", type=int, default=10)
    r.add_argument("--interval", type=float, default=0.5)
    r.add_argument("--duration", type=float, default=420)
    r.add_argument("--keepalive", type=int, default=5)
    r.add_argument("--backend", default="proxy", choices=["proxy", "netem"])
    r.add_argument("--tcp-port", type=int, default=18830)
    r.add_argument("--udp-port", type=int, default=19990)
    r.add_argument("--quiet", action="store_true")
    b = sub.add_parser("batch")
    b.add_argument("--plan", required=True)
    b.add_argument("--parallel", type=int, default=4)
    b.add_argument("--backend", default="proxy", choices=["proxy", "netem"])
    a = ap.parse_args()
    if a.cmd == "run":
        asyncio.run(run_experiment(a))
    else:
        run_batch(a)


if __name__ == "__main__":
    main()
