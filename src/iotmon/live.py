"""Block: live predictor + alerts.

Every window (5 s) it:
  1. reads the last few minutes of raw data for every active run,
  2. computes OBSERVED windows and features (exactly as in training),
  3. scores the newest complete window of each device with the trained model,
  4. feeds the rule and model alert engines and writes predictions + alerts.

Run live:            python -m iotmon.live
Replay a finished run (fills predictions/alerts so the dashboard can show them):
                     python -m iotmon.live --replay mix-s01-q0
"""
import argparse
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from . import db
from .alerts import AlertEngine
from .config import ROOT, settings
from .ml.dataset import make_features
from .windows import load_run, observed_windows

MODEL_FILE = ROOT / "models" / "degradation_model.joblib"


def utc(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def load_model():
    if not MODEL_FILE.exists():
        print("[live] no trained model found: rule alerts only (run python -m iotmon.ml.train)")
        return None
    return joblib.load(MODEL_FILE)


def write_events(conn, run_id, events):
    for ev in events:
        if ev.action == "raise":
            conn.execute("INSERT INTO alerts (run_id, device_id, raised_at, severity, reason, source)"
                         " VALUES (%s,%s,%s,%s,%s,%s)", (run_id, ev.device_id, utc(ev.at), ev.severity, ev.reason, ev.source))
        else:
            conn.execute("UPDATE alerts SET cleared_at = %s WHERE id = (SELECT id FROM alerts WHERE run_id=%s AND"
                         " device_id=%s AND source=%s AND cleared_at IS NULL ORDER BY raised_at DESC LIMIT 1)",
                         (utc(ev.at), run_id, ev.device_id, ev.source))


def score_rows(bundle, feats_df):
    if bundle is None or feats_df.empty:
        return None
    X = feats_df.reindex(columns=bundle["features"])
    return bundle["model"].predict_proba(X)[:, 1]


def replay(run_id: str) -> None:
    bundle = load_model()
    W = settings.window_s
    with db.connect() as conn:
        qos = (conn.execute("SELECT config->>'qos' FROM runs WHERE run_id=%s", (run_id,)).fetchone() or [0])[0]
        obs = observed_windows(load_run(conn, run_id), W)
        feats = make_features(obs, int(qos or 0)).sort_values("window_start").reset_index(drop=True)
        scores = score_rows(bundle, feats)
        conn.execute("DELETE FROM predictions WHERE run_id=%s", (run_id,))
        conn.execute("DELETE FROM alerts WHERE run_id=%s", (run_id,))
        engine = AlertEngine(model_threshold=bundle["threshold"] if bundle else 0.5)
        preds, events = [], []
        for i, r in feats.iterrows():
            at = float(r["window_start"]) + W
            ev = engine.update_rule(r["device_id"], at, r["recv_ratio_now"], r["delay_p95_ms_now"])
            events += [ev] if ev else []
            if scores is not None:
                s = float(scores[i])
                preds.append((run_id, r["device_id"], utc(at), bundle["horizon_s"], bundle.get("model_name", "model"), bundle["version"],
                              s, int(s >= bundle["threshold"])))
                ev = engine.update_model(r["device_id"], at, s)
                events += [ev] if ev else []
        with conn.cursor() as cur:
            cur.executemany("INSERT INTO predictions (run_id, device_id, predicted_at, horizon_s, model_name,"
                            " model_version, score, label) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)", preds)
        write_events(conn, run_id, events)
    print(f"[live] replayed {run_id}: {len(preds)} predictions, {len(events)} alert events")


def live_loop() -> None:
    bundle = load_model()
    W = settings.window_s
    engines: dict = {}
    last_scored: dict = {}      # (run, device) -> window_start of the last window already scored
    floors: dict = {}           # run -> {(device, boot): smallest transit seen} (delay baseline, as in training)
    print(f"[live] scoring every {W} s ...")
    while True:
        now = time.time()
        with db.connect() as conn:
            active = conn.execute(
                "SELECT r.run_id, r.config->>'qos', extract(epoch from r.ended_at) FROM runs r"
                " WHERE EXISTS (SELECT 1 FROM messages m WHERE m.run_id = r.run_id"
                " AND m.received_at > now() - interval '2 minutes')").fetchall()
            for run_id, qos, ended in active:
                # Build windows up to NOW, so a silence (outage) shows up as empty windows.
                # A run that has finished is only built up to its end, so its end is not an outage.
                until = min(now, float(ended)) if ended is not None else now
                obs = observed_windows(load_run(conn, run_id, since_epoch=now - 600), W,
                                       until=until, transit_floor=floors.setdefault(run_id, {}))
                if obs.empty:
                    continue
                feats = make_features(obs, int(qos or 0))
                feats = feats[feats["window_start"] + W <= until - 1]          # complete windows only
                feats = feats.sort_values("window_start").reset_index(drop=True)
                # Score EVERY complete window not scored yet (not just the newest), so no window is
                # skipped when a loop runs late. A device seen for the first time starts at its newest window.
                todo = []
                for dev, d in feats.groupby("device_id"):
                    last = last_scored.get((run_id, dev))
                    todo.append(d.tail(1) if last is None else d[d["window_start"] > last])
                todo = pd.concat(todo).sort_values("window_start").reset_index(drop=True) if todo else feats.iloc[0:0]
                if todo.empty:
                    continue
                scores = score_rows(bundle, todo)
                eng = engines.setdefault(run_id, AlertEngine(model_threshold=bundle["threshold"] if bundle else 0.5))
                events = []
                for i, r in todo.iterrows():
                    last_scored[(run_id, r["device_id"])] = r["window_start"]
                    at = float(r["window_start"]) + W
                    ev = eng.update_rule(r["device_id"], at, r["recv_ratio_now"], r["delay_p95_ms_now"])
                    events += [ev] if ev else []
                    if scores is not None:
                        s = float(scores[i])
                        conn.execute("INSERT INTO predictions (run_id, device_id, predicted_at, horizon_s, model_name,"
                                     " model_version, score, label) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                                     (run_id, r["device_id"], utc(at), bundle["horizon_s"],
                                      bundle.get("model_name", "model"), bundle["version"], s, int(s >= bundle["threshold"])))
                        ev = eng.update_model(r["device_id"], at, s)
                        events += [ev] if ev else []
                write_events(conn, run_id, events)
                for ev in events:
                    print(f"[live] {run_id}/{ev.device_id}: {ev.source} alert {ev.action.upper()} ({ev.reason})")
        time.sleep(max(0.5, W - (time.time() - now)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", help="score a finished run instead of running live")
    a = ap.parse_args()
    if a.replay:
        replay(a.replay)
    else:
        try:
            live_loop()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
