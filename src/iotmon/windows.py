"""Block: windowed metrics.

Raw rows (one per message) are turned into one row per (device, window).
Two kinds of window are computed, and keeping them apart is what prevents
data leakage in the ML part:

OBSERVED windows  - grouped by RECEIVE time. Only uses what the collector had
                    already seen when the window ended. -> model FEATURES and
                    the live dashboard.
TRUTH windows     - grouped by SEND time, with hindsight (a message that
                    arrives 20 s late still counts as received for the window
                    it was sent in). -> model LABELS and evaluation only.

Relative delay: device and collector clocks are not synchronised, so we use
transit = receive_time - device_time and subtract the smallest transit seen
for that boot (RFC 3550 idea). That gives "how much later than the fastest
message", which is defensible without clock sync.

Run:  python -m iotmon.windows --run-id demo        (computes and stores both kinds)
"""
import argparse
import math

import numpy as np
import pandas as pd
from psycopg.types.json import Jsonb

from . import db
from .config import settings
from .gaps import SequenceTracker
from .metrics import Rfc3550Jitter

UDP_PROBE_INTERVAL_S = 0.5   # emulator default; recomputed from data when possible


# ------------------------------------------------------------------ loading
def load_run(conn, run_id: str, since_epoch: float | None = None) -> dict:
    """Read one run's raw data into pandas DataFrames (times as epoch seconds)."""
    cond, params = "run_id = %s", [run_id]
    if since_epoch is not None:
        cond += " AND {col} >= to_timestamp(%s)"
        params.append(since_epoch)

    def q(sql, col):
        return _query(conn, sql.format(cond=cond.format(col=col)), params)

    return {
        "messages": q("SELECT device_id, boot_id, seq, interval_ms, device_uptime_ms AS uptime_ms,"
                      " extract(epoch from received_at) AS ts, id FROM messages WHERE {cond}", "received_at"),
        "rtt": q("SELECT device_id, probe_seq, rtt_ms, extract(epoch from measured_at) AS ts"
                 " FROM rtt_probes WHERE {cond}", "measured_at"),
        "udp": q("SELECT device_id, boot_id, probe_seq, device_uptime_ms AS uptime_ms,"
                 " extract(epoch from received_at) AS ts, id FROM udp_probes WHERE {cond}", "received_at"),
        "events": q("SELECT device_id, event, extract(epoch from at) AS ts FROM connection_events WHERE {cond}", "at"),
    }


def _query(conn, sql, params) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d.name for d in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=cols)
    for c in ("ts", "rtt_ms"):
        if c in df:
            df[c] = df[c].astype(float)
    return df


# ------------------------------------------------------------------ helpers
def _grid(data: dict, window_s: int):
    times = [df["ts"] for df in data.values() if len(df)]
    if not times:
        return None, 0
    allt = pd.concat(times)
    g0 = math.floor(allt.min() / window_s) * window_s
    n = int(math.floor((allt.max() - g0) / window_s)) + 1
    return g0, n


def _replay_gaps(df: pd.DataFrame, seq_col: str) -> pd.DataFrame:
    """Replay messages in receive order through SequenceTracker -> missing/late per row."""
    tr = SequenceTracker()
    missing, late = [], []
    for boot, seq in zip(df["boot_id"].to_numpy(), df[seq_col].to_numpy()):
        o = tr.observe("k", int(boot), int(seq))
        missing.append(o.missing)
        late.append(1 if o.kind == "late" else 0)
    df = df.copy()
    df["missing_revealed"] = missing
    df["late"] = late
    return df


# ------------------------------------------------------------------ observed
def observed_windows(data: dict, window_s: int = None, ontime_ms: float = None) -> pd.DataFrame:
    window_s = window_s or settings.window_s
    ontime_ms = ontime_ms or settings.ontime_ms
    g0, n = _grid(data, window_s)
    if g0 is None:
        return pd.DataFrame()
    msgs, rtt, udp, events = data["messages"], data["rtt"], data["udp"], data["events"]
    devices = sorted(set(msgs["device_id"]) | set(udp["device_id"]))
    out = []
    for dev in devices:
        idx = pd.RangeIndex(n, name="w")
        frame = pd.DataFrame(index=idx)
        m = msgs[msgs["device_id"] == dev].sort_values(["ts", "id"])
        interval_ms = float(m["interval_ms"].median()) if len(m) and m["interval_ms"].notna().any() else 1000.0
        expected_n = window_s * 1000.0 / interval_ms
        if len(m):
            m = _replay_gaps(m, "seq")
            transit = m["ts"] - m["uptime_ms"] / 1000.0
            m["rel_delay_ms"] = (transit - transit.groupby(m["boot_id"]).cummin()) * 1000.0
            jit = Rfc3550Jitter()
            m["jitter_ms"] = [jit.update(t * 1000.0) for t in transit]
            m["iat_ms"] = m["ts"].diff() * 1000.0
            m["late_flag"] = (m["rel_delay_ms"] > ontime_ms).astype(int)
            m["w"] = ((m["ts"] - g0) // window_s).astype(int)
            g = m.groupby("w")
            frame["n_received"] = g.size()
            frame["missing_revealed"] = g["missing_revealed"].sum()
            frame["late_arrivals"] = g["late"].sum()
            frame["delay_mean_ms"] = g["rel_delay_ms"].mean()
            frame["delay_p95_ms"] = g["rel_delay_ms"].quantile(0.95)
            frame["delay_max_ms"] = g["rel_delay_ms"].max()
            frame["frac_over_ontime"] = g["late_flag"].mean()
            frame["iat_mean_ms"] = g["iat_ms"].mean()
            frame["iat_std_ms"] = g["iat_ms"].std()
            frame["iat_max_ms"] = g["iat_ms"].max()
            frame["jitter_ms"] = g["jitter_ms"].last()
            frame["last_rx_ts"] = g["ts"].max()
        for c in ("n_received", "missing_revealed", "late_arrivals"):
            frame[c] = frame.get(c, pd.Series(dtype=float)).reindex(idx).fillna(0)
        frame["jitter_ms"] = frame.get("jitter_ms", pd.Series(dtype=float)).reindex(idx).ffill()
        last_rx = frame.get("last_rx_ts", pd.Series(dtype=float)).reindex(idx).ffill()
        w_end = g0 + (idx.to_numpy() + 1) * window_s
        frame["silence_s"] = np.where(last_rx.notna(), w_end - last_rx, np.nan)
        frame = frame.drop(columns=["last_rx_ts"], errors="ignore")
        frame["expected_n"] = expected_n
        frame["recv_ratio"] = frame["n_received"] / expected_n
        seen = frame["n_received"] + frame["missing_revealed"]
        frame["loss_revealed"] = np.where(seen > 0, frame["missing_revealed"] / seen.replace(0, np.nan), np.nan)

        # RTT probes (application-level round trip)
        r = rtt[rtt["device_id"] == dev]
        if len(r):
            r = r.assign(w=((r["ts"] - g0) // window_s).astype(int))
            g = r.groupby("w")
            frame["rtt_mean_ms"] = g["rtt_ms"].mean()
            frame["rtt_p95_ms"] = g["rtt_ms"].quantile(0.95)
            frame["rtt_lost"] = g["rtt_ms"].apply(lambda s: s.isna().sum())
        frame["rtt_lost"] = frame.get("rtt_lost", pd.Series(dtype=float)).reindex(idx).fillna(0)

        # UDP probes (network-layer loss; no retransmission)
        u = udp[udp["device_id"] == dev].sort_values(["ts", "id"])
        probe_iv = UDP_PROBE_INTERVAL_S
        if len(u) > 5:
            d = u.sort_values("probe_seq")
            step = (d["uptime_ms"].diff() / d["probe_seq"].diff()).median()
            if step and not math.isnan(step) and step > 0:
                probe_iv = step / 1000.0
        if len(u):
            u = _replay_gaps(u, "probe_seq").assign(w=lambda x: ((x["ts"] - g0) // window_s).astype(int))
            g = u.groupby("w")
            frame["udp_received"] = g.size()
            frame["udp_missing_revealed"] = g["missing_revealed"].sum()
        for c in ("udp_received", "udp_missing_revealed"):
            frame[c] = frame.get(c, pd.Series(dtype=float)).reindex(idx).fillna(0)
        frame["udp_recv_ratio"] = frame["udp_received"] / (window_s / probe_iv)

        # connection events
        e = events[events["device_id"] == dev].sort_values("ts")
        frame["offline_events"] = 0.0
        frame["online_events"] = 0.0
        if len(e):
            e = e.assign(w=((e["ts"] - g0) // window_s).astype(int))
            cnt = e.groupby(["w", "event"]).size().unstack(fill_value=0)
            for ev in ("offline", "online"):
                if ev in cnt:
                    frame[f"{ev}_events"] = cnt[ev].reindex(idx).fillna(0)
        last_off = e[e["event"] == "offline"]["ts"].to_numpy() if len(e) else np.array([])
        since = []
        for we in w_end:
            prev = last_off[last_off <= we]
            since.append(min(600.0, we - prev[-1]) if len(prev) else 600.0)
        frame["secs_since_offline"] = since
        state = np.ones(n)
        if len(e):
            ev_w = e["w"].to_numpy()
            ev_on = (e["event"] == "online").to_numpy()
            cur = 1.0
            j = 0
            for i in range(n):
                while j < len(ev_w) and ev_w[j] <= i:
                    cur = 1.0 if ev_on[j] else 0.0
                    j += 1
                state[i] = cur
        frame["connected"] = state

        frame["device_id"] = dev
        frame["window_start"] = g0 + idx.to_numpy() * window_s
        out.append(frame.reset_index(drop=True))
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ truth
def truth_windows(data: dict, window_s: int = None, ontime_ms: float = None) -> pd.DataFrame:
    window_s = window_s or settings.window_s
    ontime_ms = ontime_ms or settings.ontime_ms
    msgs = data["messages"]
    g0, n = _grid(data, window_s)
    if g0 is None or not len(msgs):
        return pd.DataFrame()
    rows = []
    for (dev, boot), m in msgs.groupby(["device_id", "boot_id"]):
        m = m.drop_duplicates("seq").sort_values("seq")
        transit = m["ts"] - m["uptime_ms"] / 1000.0
        offset = transit.min()                       # hindsight: the fastest message of the whole boot
        send_ts = (offset + m["uptime_ms"] / 1000.0).to_numpy()
        rel = ((transit - offset) * 1000.0).to_numpy()
        seqs = m["seq"].to_numpy()
        all_seq = np.arange(seqs.min(), seqs.max() + 1)
        all_send = np.interp(all_seq, seqs, send_ts)   # missing seqs: interpolated send time
        received = np.isin(all_seq, seqs)
        delay = np.full(len(all_seq), np.nan)
        delay[received] = rel
        rows.append(pd.DataFrame({"device_id": dev, "send_ts": all_send, "received": received,
                                  "ontime": received & (delay <= ontime_ms), "delay_ms": delay}))
    t = pd.concat(rows, ignore_index=True)
    t["w"] = ((t["send_ts"] - g0) // window_s).astype(int)
    g = t.groupby(["device_id", "w"])
    out = pd.DataFrame({
        "expected": g.size(),
        "received": g["received"].sum(),
        "ontime": g["ontime"].sum(),
        "delay_p95_ms": g["delay_ms"].quantile(0.95),
    }).reset_index()
    out["pdr"] = out["received"] / out["expected"]
    out["otdr"] = out["ontime"] / out["expected"]
    out["window_start"] = g0 + out["w"] * window_s
    return out.drop(columns=["w"])


# ------------------------------------------------------------------ storing
def _clean(d: dict) -> dict:
    return {k: (None if isinstance(v, float) and math.isnan(v) else
                (float(v) if isinstance(v, (np.floating, np.integer)) else v)) for k, v in d.items()}


def store_windows(conn, run_id: str, kind: str, df: pd.DataFrame, window_s: int) -> int:
    if df.empty:
        return 0
    metric_cols = [c for c in df.columns if c not in ("device_id", "window_start")]
    rows = [(run_id, r["device_id"], kind, float(r["window_start"]), window_s,
             Jsonb(_clean({c: r[c] for c in metric_cols}))) for _, r in df.iterrows()]
    with conn.cursor() as cur:
        cur.execute("DELETE FROM windows WHERE run_id = %s AND kind = %s AND window_s = %s", (run_id, kind, window_s))
        cur.executemany("INSERT INTO windows (run_id, device_id, kind, window_start, window_s, metrics)"
                        " VALUES (%s,%s,%s,to_timestamp(%s),%s,%s)", rows)
    return len(rows)


def compute_and_store(run_id: str, window_s: int = None) -> tuple[int, int]:
    window_s = window_s or settings.window_s
    with db.connect() as conn:
        data = load_run(conn, run_id)
        obs = observed_windows(data, window_s)
        tru = truth_windows(data, window_s)
        return store_windows(conn, run_id, "observed", obs, window_s), store_windows(conn, run_id, "truth", tru, window_s)


def main():
    ap = argparse.ArgumentParser(description="Compute observed + truth windows for a run")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--window-s", type=int, default=settings.window_s)
    a = ap.parse_args()
    n_obs, n_truth = compute_and_store(a.run_id, a.window_s)
    print(f"stored {n_obs} observed and {n_truth} truth windows for run {a.run_id}")


if __name__ == "__main__":
    main()
