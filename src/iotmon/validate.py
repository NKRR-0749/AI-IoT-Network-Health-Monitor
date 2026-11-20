"""Block: testbed validation (Phase 2 gate).

For every impairment phase of a run, compare what was INJECTED with what the
system MEASURED:
    injected UDP loss  vs  measured UDP probe loss      (should match)
    injected delay     vs  measured RTT increase         (should show up)
    outages            vs  disconnect events / message loss
If these do not line up, the testbed is broken and no ML result can be trusted.

Run:  python -m iotmon.validate --run-id pilot-q0
      python -m iotmon.validate --summary 'mix-%'      (RQ1 table over many runs)
"""
import argparse

import numpy as np
import pandas as pd

from . import db
from .windows import _query


def phase_table(conn, run_id: str) -> pd.DataFrame:
    ph = _query(conn, "SELECT phase, extract(epoch from start_at) AS t0, extract(epoch from end_at) AS t1, params"
                      " FROM impairments WHERE run_id=%s ORDER BY start_at", [run_id])
    msgs = _query(conn, "SELECT device_id, boot_id, seq, device_uptime_ms AS up, extract(epoch from received_at) AS ts"
                        " FROM messages WHERE run_id=%s", [run_id])
    udp = _query(conn, "SELECT device_id, boot_id, probe_seq AS seq, device_uptime_ms AS up,"
                       " extract(epoch from received_at) AS ts FROM udp_probes WHERE run_id=%s", [run_id])
    rtt = _query(conn, "SELECT rtt_ms, extract(epoch from measured_at) AS ts FROM rtt_probes WHERE run_id=%s", [run_id])
    ev = _query(conn, "SELECT event, extract(epoch from at) AS ts FROM connection_events WHERE run_id=%s", [run_id])
    for df in (ph,):
        df["t0"], df["t1"] = df["t0"].astype(float), df["t1"].astype(float)

    def send_times(df):
        """Expand to every seq (received or not) with an estimated send time (hindsight, as in truth windows)."""
        rows = []
        for (_, _), g in df.groupby(["device_id", "boot_id"]):
            g = g.sort_values("seq")
            off = (g["ts"] - g["up"] / 1000).min()
            s = g["seq"].to_numpy()
            allseq = np.arange(s.min(), s.max() + 1)
            rows.append(pd.DataFrame({"t": np.interp(allseq, s, (off + g["up"] / 1000).to_numpy()),
                                      "rx": np.isin(allseq, s)}))
        return pd.concat(rows) if rows else pd.DataFrame(columns=["t", "rx"])

    m_all, u_all = send_times(msgs), send_times(udp)
    out = []
    for _, p in ph.iterrows():
        m = m_all[(m_all["t"] >= p.t0) & (m_all["t"] < p.t1)]
        u = u_all[(u_all["t"] >= p.t0) & (u_all["t"] < p.t1)]
        r = rtt[(rtt["ts"] >= p.t0) & (rtt["ts"] < p.t1)]["rtt_ms"]
        e = ev[(ev["ts"] >= p.t0) & (ev["ts"] < p.t1)]
        params = p.params or {}
        out.append({
            "phase": p.phase,
            "seconds": round(p.t1 - p.t0, 1),
            "injected_loss_%": params.get("loss_pct", 100 if params.get("outage") else 0),
            "injected_delay_ms": params.get("delay_ms", 0),
            "udp_loss_measured_%": round(100 * (1 - u["rx"].mean()), 2) if len(u) else None,
            "mqtt_msg_loss_%": round(100 * (1 - m["rx"].mean()), 2) if len(m) else None,
            "rtt_median_ms": round(float(r.median()), 1) if r.notna().any() else None,
            "rtt_lost": int(r.isna().sum()),
            "disconnects": int((e["event"] == "offline").sum()),
        })
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", help="one run, phase by phase")
    ap.add_argument("--summary", metavar="PATTERN", help="RQ1 summary over runs matching a SQL LIKE pattern, e.g. 'mix-%%'")
    ap.add_argument("--markdown", help="also write the table to this .md file")
    a = ap.parse_args()
    if a.summary:
        t = summary(a.summary)
        print(t.to_string(index=False))
        if a.markdown:
            open(a.markdown, "w").write(t.to_markdown(index=False) + "\n")
        return
    if not a.run_id:
        ap.error("give --run-id or --summary")
    with db.connect() as conn:
        t = phase_table(conn, a.run_id)
    print(t.to_string(index=False))
    if a.markdown:
        with open(a.markdown, "w") as f:
            f.write(f"# Testbed validation: run `{a.run_id}`\n\n" + t.to_markdown(index=False) + "\n")



def summary(run_like: str = "mix-%") -> pd.DataFrame:
    """RQ1 across many runs: mean measured effect per (phase kind, QoS)."""
    with db.connect() as conn:
        rows = conn.execute("SELECT run_id, (config->>'qos')::int FROM runs WHERE run_id LIKE %s AND ended_at IS NOT NULL",
                            (run_like,)).fetchall()
        parts = []
        for run_id, qos in rows:
            t = phase_table(conn, run_id)
            t["qos"] = qos
            parts.append(t)
    t = pd.concat(parts)
    t = t[t["phase"] != "clean"]
    g = t.groupby(["phase", "qos"]).agg(phases=("seconds", "size"),
                                        udp_loss_pct=("udp_loss_measured_%", "mean"),
                                        mqtt_msg_loss_pct=("mqtt_msg_loss_%", "mean"),
                                        rtt_median_ms=("rtt_median_ms", "mean"),
                                        disconnects_per_phase=("disconnects", "mean"))
    return g.round(2).reset_index()


if __name__ == "__main__":
    main()
