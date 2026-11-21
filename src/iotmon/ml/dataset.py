"""Block: ML dataset = features (past) + labels (future), with no leakage.

At every window end t, for every device:

  FEATURES  come only from OBSERVED windows ending at or before t
            (what the collector had actually seen by time t).
  LABEL     comes from TRUTH windows whose messages were SENT in (t, t + horizon]:
              y_degraded = 1 if the on-time delivery ratio over the horizon < 0.9
              y_loss     = 1 if the delivery ratio over the horizon < 0.95
            ("on time" = relative delay <= 1 s, see windows.py)

Rows whose horizon runs past the end of the data are dropped.
The impairment schedule (ground truth) is NEVER a feature.
"""
import numpy as np
import pandas as pd

from .. import db
from ..config import settings
from ..windows import load_run, observed_windows, truth_windows

LOOKBACK = 6                  # windows of history used for rolling stats (6 x 5 s = 30 s)
OTDR_THRESHOLD = 0.9
PDR_THRESHOLD = 0.95

BASE = ["recv_ratio", "loss_revealed", "frac_over_ontime", "delay_mean_ms", "delay_p95_ms", "delay_max_ms",
        "iat_max_ms", "iat_std_ms", "jitter_ms", "silence_s", "rtt_mean_ms", "rtt_p95_ms", "rtt_lost",
        "offline_events", "online_events", "connected", "late_arrivals"]
PROBE = ["udp_recv_ratio"]


def _slope(a: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    if len(a) < 2:
        return 0.0
    x = np.arange(len(a))
    return float(np.polyfit(x, a, 1)[0])


# extra columns used only by the loss-PATTERN classifier (ml/patterns.py): the shape of the gaps
PATTERN_BASE = ["gap_events", "gap_len_max", "gap_len_mean"]
PATTERN_PROBE = ["udp_loss_revealed", "udp_gap_events", "udp_gap_len_max", "udp_gap_len_mean"]


def make_features(obs: pd.DataFrame, qos: int, extra: list[str] | None = None) -> pd.DataFrame:
    """Turn observed windows (one device or many) into one feature row per window end.
    `extra` adds more observed-window columns (now / lag / rolling stats) on top of the standard set."""
    cols = BASE + PROBE + list(extra or [])
    rows = []
    for dev, d in obs.sort_values("window_start").groupby("device_id"):
        d = d.reset_index(drop=True)
        f = {"device_id": pd.Series(dev, index=d.index), "window_start": d["window_start"]}
        for c in cols:
            s = d[c].astype(float) if c in d else pd.Series(np.nan, index=d.index)
            f[f"{c}_now"] = s
            f[f"{c}_lag1"] = s.shift(1)
            roll = s.rolling(LOOKBACK, min_periods=1)
            f[f"{c}_mean6"] = roll.mean()
            f[f"{c}_max6"] = roll.max()
            f[f"{c}_min6"] = roll.min()
        for c in ("recv_ratio", "delay_p95_ms", "rtt_p95_ms", "udp_recv_ratio", "jitter_ms"):
            s = d[c].astype(float) if c in d else pd.Series(np.nan, index=d.index)
            f[f"{c}_slope6"] = s.rolling(LOOKBACK, min_periods=2).apply(_slope, raw=True)
        f["secs_since_offline"] = d["secs_since_offline"]
        f["qos"] = pd.Series(qos, index=d.index)
        rows.append(pd.DataFrame(f))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def feature_columns(df: pd.DataFrame, with_probe: bool = True) -> list[str]:
    cols = [c for c in df.columns if c not in ("device_id", "window_start", "run_id", "seed", "profile")
            and not c.startswith("y_") and not c.startswith("target_")]
    if not with_probe:
        cols = [c for c in cols if not c.startswith("udp_")]
    return cols


def make_labels(truth: pd.DataFrame, window_s: int, horizon_s: int) -> pd.DataFrame:
    """For each window end t: delivery over windows starting in [t, t + horizon)."""
    h = horizon_s // window_s
    out = []
    for dev, d in truth.sort_values("window_start").groupby("device_id"):
        d = d.set_index("window_start")
        grid = np.arange(d.index.min(), d.index.max() + window_s, window_s)
        d = d.reindex(grid)
        exp = d["expected"].fillna(0).to_numpy()
        rec = d["received"].fillna(0).to_numpy()
        ont = d["ontime"].fillna(0).to_numpy()
        for i, w in enumerate(grid):
            # feature row "window_start = w" describes data up to w + window_s; the horizon starts there
            j0, j1 = i + 1, i + 1 + h
            if j1 > len(grid):
                break
            e = exp[j0:j1].sum()
            if e == 0:
                continue
            otdr, pdr = ont[j0:j1].sum() / e, rec[j0:j1].sum() / e
            nxt = ont[j0] / exp[j0] if exp[j0] else np.nan      # the single next window (for onset metrics)
            out.append({"device_id": dev, "window_start": w, "target_otdr": otdr, "target_pdr": pdr,
                        "target_next_otdr": nxt,
                        "y_degraded": int(otdr < OTDR_THRESHOLD), "y_loss": int(pdr < PDR_THRESHOLD)})
    return pd.DataFrame(out)


def build_run(conn, run_id: str, window_s: int = None, horizon_s: int = None) -> pd.DataFrame:
    window_s = window_s or settings.window_s
    horizon_s = horizon_s or settings.horizon_s
    cfg = conn.execute("SELECT config, seed FROM runs WHERE run_id = %s", (run_id,)).fetchone()
    config, seed = (cfg[0] or {}), cfg[1]
    data = load_run(conn, run_id)
    feats = make_features(observed_windows(data, window_s), int(config.get("qos", 0)))
    labels = make_labels(truth_windows(data, window_s), window_s, horizon_s)
    df = feats.merge(labels, on=["device_id", "window_start"], how="inner")
    return df.assign(run_id=run_id, seed=seed, profile=config.get("profile", "unknown"))


def build(run_ids: list[str]) -> pd.DataFrame:
    with db.connect() as conn:
        return pd.concat([build_run(conn, r) for r in run_ids], ignore_index=True)
