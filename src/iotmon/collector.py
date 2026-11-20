"""Block: the collector (the "Python backend").

It does four jobs:
  1. Subscribes to MQTT and stores every telemetry message (with OUR receive time).
  2. Detects sequence gaps live and prints them.
  3. Echoes RTT pings straight back (so devices can time the round trip).
  4. Listens on a UDP port for network-layer probe packets.

Database writes happen on a separate thread in small batches, so a slow
insert never blocks the MQTT network loop.

Run:  python -m iotmon.collector
Stop: Ctrl+C
"""
import argparse
import json
import queue
import socket
import threading
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

from . import db
from .config import TOPIC_ROOT, parse_topic, settings, topic
from .gaps import SequenceTracker
from .protocol import BadMessage, decode_telemetry


def utc(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


class Collector:
    def __init__(self, verbose: bool = True, device_kind: str = "emulated") -> None:
        self.verbose = verbose
        self.device_kind = device_kind
        self.rows: queue.Queue = queue.Queue()
        self.tracker = SequenceTracker()
        self.stats = {"telemetry": 0, "gaps": 0, "missing": 0, "late": 0, "dup": 0, "bad": 0, "udp": 0}
        self._stop = threading.Event()
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"collector-{int(time.time())}")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    # ---------------- MQTT ----------------
    def _on_connect(self, client, userdata, flags, reason_code, properties):
        print(f"[collector] connected to broker ({reason_code})")
        for kind in ("telemetry", "ping", "rtt", "status"):
            client.subscribe(f"{TOPIC_ROOT}/+/+/{kind}", qos=1)

    def _on_message(self, client, userdata, msg):
        now = time.time()                     # receive time: the first thing we record
        parsed = parse_topic(msg.topic)
        if not parsed:
            return
        run_id, device_id, kind = parsed
        if kind == "telemetry":
            self._handle_telemetry(run_id, device_id, msg, now)
        elif kind == "ping":
            # echo immediately; the device times the round trip on its own clock
            client.publish(topic(run_id, device_id, "pong"), msg.payload, qos=0)
        elif kind == "rtt":
            try:
                d = json.loads(msg.payload)
                self.rows.put(("rtt", (run_id, device_id, int(d["probe_seq"]), d.get("rtt_ms"), utc(now))))
            except (ValueError, KeyError, TypeError):
                self.stats["bad"] += 1
        elif kind == "status":
            state = msg.payload.decode(errors="replace").strip()
            if state in ("online", "offline") and not msg.retain:
                self.rows.put(("event", (run_id, device_id, state, utc(now))))

    def _handle_telemetry(self, run_id, device_id, msg, now):
        try:
            m = decode_telemetry(msg.payload)
        except BadMessage as e:
            self.stats["bad"] += 1
            if self.verbose:
                print(f"[collector] bad message on {msg.topic}: {e}")
            return
        self.stats["telemetry"] += 1
        obs = self.tracker.observe((run_id, device_id), m["boot_id"], m["seq"])
        if obs.kind == "gap":
            self.stats["gaps"] += 1
            self.stats["missing"] += obs.missing
            if self.verbose:
                print(f"[collector] GAP {run_id}/{device_id}: {obs.missing} message(s) missing before seq {m['seq']}")
        elif obs.kind == "late":
            self.stats["late"] += 1
        elif obs.kind == "duplicate":
            self.stats["dup"] += 1
            return                              # do not store duplicates twice
        self.rows.put(("msg", (run_id, device_id, m["boot_id"], m["seq"], m["qos"], m.get("interval_ms"),
                               m["uptime_ms"], m.get("rssi"), utc(now), len(msg.payload))))

    # ---------------- UDP probes ----------------
    def _udp_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", settings.udp_probe_port))
        sock.settimeout(0.5)
        while not self._stop.is_set():
            try:
                data, _ = sock.recvfrom(2048)
            except socket.timeout:
                continue
            now = time.time()
            try:
                d = json.loads(data)
                self.rows.put(("udp", (d["run_id"], d["device_id"], int(d["boot_id"]), int(d["probe_seq"]),
                                       int(d["uptime_ms"]), utc(now))))
                self.stats["udp"] += 1
            except (ValueError, KeyError, TypeError):
                self.stats["bad"] += 1
        sock.close()

    # ---------------- database writer ----------------
    def _writer_loop(self):
        conn = db.connect()
        known_runs, known_devices = set(), set()
        while not self._stop.is_set() or not self.rows.empty():
            batch = []
            try:
                batch.append(self.rows.get(timeout=0.5))
                while len(batch) < 2000:
                    batch.append(self.rows.get_nowait())
            except queue.Empty:
                pass
            if not batch:
                continue
            with conn.transaction():
                for kind, r in batch:
                    run_id, device_id = r[0], r[1]
                    if run_id not in known_runs:
                        db.ensure_run(conn, run_id)
                        known_runs.add(run_id)
                    if device_id not in known_devices:
                        db.ensure_device(conn, device_id, self.device_kind)
                        known_devices.add(device_id)
                with conn.cursor() as cur:
                    msgs = [r for k, r in batch if k == "msg"]
                    if msgs:
                        cur.executemany(
                            "INSERT INTO messages (run_id, device_id, boot_id, seq, qos, interval_ms, device_uptime_ms,"
                            " rssi, received_at, payload_bytes) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                            " ON CONFLICT DO NOTHING", msgs)
                    rtts = [r for k, r in batch if k == "rtt"]
                    if rtts:
                        cur.executemany("INSERT INTO rtt_probes (run_id, device_id, probe_seq, rtt_ms, measured_at)"
                                        " VALUES (%s,%s,%s,%s,%s)", rtts)
                    udps = [r for k, r in batch if k == "udp"]
                    if udps:
                        cur.executemany("INSERT INTO udp_probes (run_id, device_id, boot_id, probe_seq, device_uptime_ms,"
                                        " received_at) VALUES (%s,%s,%s,%s,%s,%s)", udps)
                    evs = [r for k, r in batch if k == "event"]
                    if evs:
                        cur.executemany("INSERT INTO connection_events (run_id, device_id, event, at)"
                                        " VALUES (%s,%s,%s,%s)", evs)
        conn.close()

    # ---------------- lifecycle ----------------
    def start(self):
        db.init_schema()
        self._threads = [threading.Thread(target=self._writer_loop, daemon=True),
                         threading.Thread(target=self._udp_loop, daemon=True)]
        for t in self._threads:
            t.start()
        self.client.connect(settings.mqtt_host, settings.mqtt_port, keepalive=30)
        self.client.loop_start()

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()
        self._stop.set()
        for t in self._threads:
            t.join(timeout=10)


def main():
    ap = argparse.ArgumentParser(description="MQTT + UDP collector")
    ap.add_argument("--quiet", action="store_true", help="do not print every gap")
    ap.add_argument("--device-kind", default="emulated", choices=["emulated", "wokwi", "physical"])
    args = ap.parse_args()
    c = Collector(verbose=not args.quiet, device_kind=args.device_kind)
    c.start()
    print(f"[collector] listening: MQTT {settings.mqtt_host}:{settings.mqtt_port}, UDP :{settings.udp_probe_port}")
    try:
        while True:
            time.sleep(10)
            s = c.stats
            print(f"[collector] received={s['telemetry']} gaps={s['gaps']} missing={s['missing']} "
                  f"late={s['late']} dup={s['dup']} udp={s['udp']} bad={s['bad']} queue={c.rows.qsize()}")
    except KeyboardInterrupt:
        print("\n[collector] stopping, flushing to database ...")
    finally:
        c.stop()


if __name__ == "__main__":
    main()
