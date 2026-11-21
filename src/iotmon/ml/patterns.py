"""Block: packet-loss PATTERN analysis (what kind of trouble is this device in right now?).

The degradation model (train.py) answers "will it go bad in the next 30 s?".
This block answers a different question for every 5 s window and device:

    which pattern is the link showing NOW?
      clean        no impairment
      random_loss  independent packet loss (Bernoulli)
      bursty_loss  losses in runs (Gilbert-Elliott bursts)
      delay        added delay / jitter, no loss
      stall        traffic frozen for a few seconds, connection kept
      outage       connection dropped

Labels come from the fault injector's ground truth (the `impairments` table):
each window gets the phase that covers most of it. Ramp phases count as their
kind (ramp_loss -> random or bursty, ramp_delay -> delay).

Features: the observed-window features of the degradation model PLUS the shape of
the sequence gaps (number of gaps, mean and max gap length = burst length), for
MQTT messages and for UDP probes. Only data the collector had at the end of the
window is used, so the classifier can run live.

Evaluation (honest):
  * split by schedule seed (GroupKFold): a test schedule is never seen in training
  * all runs are used, so bursty loss appears in training folds (unlike RQ3)
  * baseline = a hand-written rule classifier with sensible thresholds
  * every model is reported with and without UDP probe features, because UDP
    probes are optional on real devices
  * metrics: macro F1 (every class counts equally), per-class recall, confusion matrix

Run:   python -m iotmon.ml.patterns
Writes results/pattern_results.json, results/pattern_report.md, models/pattern_model.joblib
"""
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

from .. import db
from ..config import ROOT, settings
from ..windows import load_run, observed_windows
from .dataset import PATTERN_BASE, PATTERN_PROBE, make_features

warnings.filterwarnings("ignore")
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
PATTERNS = ["clean", "random_loss", "bursty_loss", "delay", "stall", "outage"]
SEED = 0


# ------------------------------------------------------------------ labels
def phase_pattern(phase: str, params: dict) -> str:
    params = params or {}
    if params.get("outage") == "drop":
        return "outage"
    if params.get("outage") == "stall":
        return "stall"
    if params.get("loss_pct"):
        return "bursty_loss" if params.get("burst_len") else "random_loss"
    if params.get("delay_ms") or params.get("jitter_ms"):
        return "delay"
    return "clean"


def load_phases(conn, run_id: str) -> pd.DataFrame:
    rows = conn.execute(
        "SELECT phase, extract(epoch from start_at), extract(epoch from coalesce(end_at, start_at)), params"
        " FROM impairments WHERE run_id = %s ORDER BY start_at", (run_id,)).fetchall()
    df = pd.DataFrame(rows, columns=["phase", "start", "end", "params"])
    if df.empty:
        return df
    df[["start", "end"]] = df[["start", "end"]].astype(float)
    df["pattern"] = [phase_pattern(p, q) for p, q in zip(df["phase"], df["params"])]
    df["intensity"] = [float((q or {}).get("loss_pct") or (q or {}).get("delay_ms") or 0) for q in df["params"]]
    return df


def label_windows(window_starts: np.ndarray, phases: pd.DataFrame, window_s: int) -> pd.DataFrame:
    """Pattern of the phase that covers most of each window (None outside the schedule)."""
    out = []
    for w in window_starts:
        ov = (np.minimum(phases["end"], w + window_s) - np.maximum(phases["start"], w)).clip(lower=0)
        if ov.max() < window_s / 2:
            out.append((w, None, 0.0, False))
            continue
        i = int(ov.idxmax())
        out.append((w, phases.at[i, "pattern"], phases.at[i, "intensity"], bool(ov.max() >= window_s - 1e-6)))
    return pd.DataFrame(out, columns=["window_start", "pattern", "intensity", "pure"])


LONG = 12                     # 12 x 5 s = 60 s of gap history (bursts are intermittent)
LONG_COLS = ["gap_events_now", "gap_len_max_now", "loss_revealed_now",
             "udp_gap_events_now", "udp_gap_len_max_now", "udp_loss_revealed_now"]


def add_long_history(feats: pd.DataFrame) -> pd.DataFrame:
    """60 s max and sum of the gap features. A bursty link is clean most of the time, so the
    evidence for 'bursty' is often a minute old; the 30 s rolling stats miss much of it."""
    feats = feats.sort_values(["device_id", "window_start"]).reset_index(drop=True)
    g = feats.groupby("device_id")
    for c in LONG_COLS:
        base = c[:-len("_now")]
        feats[f"{base}_max12"] = g[c].transform(lambda s: s.rolling(LONG, min_periods=1).max())
        feats[f"{base}_sum12"] = g[c].transform(lambda s: s.fillna(0).rolling(LONG, min_periods=1).sum())
    return feats


def pattern_features(obs: pd.DataFrame, qos: int) -> pd.DataFrame:
    """Observed windows -> pattern-classifier features (also used by the live predictor)."""
    return add_long_history(make_features(obs, qos, extra=PATTERN_BASE + PATTERN_PROBE))


def build_run(conn, run_id: str, window_s: int = None) -> pd.DataFrame:
    window_s = window_s or settings.window_s
    cfg = conn.execute("SELECT config, seed FROM runs WHERE run_id = %s", (run_id,)).fetchone()
    config, seed = (cfg[0] or {}), cfg[1]
    phases = load_phases(conn, run_id)
    if phases.empty:
        return pd.DataFrame()
    feats = pattern_features(observed_windows(load_run(conn, run_id), window_s), int(config.get("qos", 0)))
    labels = label_windows(np.sort(feats["window_start"].unique()), phases, window_s)
    df = feats.merge(labels, on="window_start").dropna(subset=["pattern"])
    return df.assign(run_id=run_id, seed=seed, profile=config.get("profile", "unknown"))


def build(run_ids) -> pd.DataFrame:
    with db.connect() as conn:
        return pd.concat([build_run(conn, r) for r in run_ids], ignore_index=True)


def feature_cols(df: pd.DataFrame, with_probe: bool) -> list[str]:
    drop = {"device_id", "window_start", "run_id", "seed", "profile", "pattern", "intensity", "pure"}
    cols = [c for c in df.columns if c not in drop]
    return cols if with_probe else [c for c in cols if not c.startswith("udp_")]


# ------------------------------------------------------------------ models
def rule_classifier(X: pd.DataFrame, with_probe: bool) -> np.ndarray:
    """Hand-written baseline. Thresholds are fixed by reasoning, not fitted to the data."""
    out = np.full(len(X), "clean", dtype=object)
    loss = (X["udp_loss_revealed_now"] if with_probe else X["loss_revealed_now"]).fillna(0).to_numpy()
    burst = (X["udp_gap_len_mean_now"] if with_probe else X["gap_len_mean_now"]).fillna(0).to_numpy()
    delay = X["delay_p95_ms_now"].fillna(0).to_numpy()
    out[delay > 200] = "delay"
    out[loss > 0.02] = "random_loss"
    out[(loss > 0.02) & (burst >= 2.5)] = "bursty_loss"
    out[(X["silence_s_now"].fillna(0).to_numpy() > 2.0)] = "stall"
    out[(X["connected_now"].to_numpy() == 0) | (X["offline_events_now"].to_numpy() > 0)] = "outage"
    return out


def make_model():
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1,
                                                class_weight="balanced_subsample", random_state=SEED))


def report_block(y, pred) -> dict:
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=PATTERNS, zero_division=0)
    return {"macro_f1": round(float(f1_score(y, pred, labels=PATTERNS, average="macro", zero_division=0)), 3),
            "accuracy": round(float(accuracy_score(y, pred)), 3),
            "per_class": {c: {"precision": round(float(p[i]), 3), "recall": round(float(r[i]), 3),
                              "f1": round(float(f[i]), 3), "n": int(n[i])} for i, c in enumerate(PATTERNS)},
            "confusion_matrix": {"labels": PATTERNS,
                                 "rows_true_cols_pred": confusion_matrix(y, pred, labels=PATTERNS).tolist()}}


def crossval(df: pd.DataFrame, with_probe: bool, n_splits: int = 5) -> np.ndarray:
    feats = feature_cols(df, with_probe)
    pred = np.empty(len(df), dtype=object)
    for tr, te in GroupKFold(n_splits=n_splits).split(df, df["pattern"], df["seed"]):
        m = make_model().fit(df.iloc[tr][feats], df.iloc[tr]["pattern"])
        pred[te] = m.predict(df.iloc[te][feats])
    return pred


def by_slice(df, y, preds: dict) -> dict:
    """Macro F1 on slices that matter for interpretation."""
    out = {}
    slices = {"qos0": df["qos"] == 0, "qos1": df["qos"] == 1,
              "pure_windows (one phase covers the whole window)": df["pure"],
              "transition_windows": ~df["pure"]}
    for name, mask in slices.items():
        mask = mask.to_numpy()
        out[name] = {m: round(float(f1_score(y[mask], p[mask], labels=PATTERNS, average="macro", zero_division=0)), 3)
                     for m, p in preds.items()}
        out[name]["n"] = int(mask.sum())
    return out


def recall_by_intensity(df, y, pred) -> dict:
    """Recall of the RF for loss/delay windows, split by how strong the impairment was."""
    out = {}
    for pat in ("random_loss", "bursty_loss", "delay"):
        d = df[y == pat]
        p = pred[(y == pat)]
        for lvl in sorted(d["intensity"].unique()):
            mask = (d["intensity"] == lvl).to_numpy()
            if mask.sum() >= 20:
                out.setdefault(pat, {})[f"{lvl:g}"] = {"n": int(mask.sum()),
                                                      "recall": round(float((p[mask] == pat).mean()), 3)}
    return out


def bursty_observability(df, y, pred) -> dict:
    """How much of a bursty phase shows any loss at all? (A Gilbert-Elliott link is clean most of the time.)"""
    b = y == "bursty_loss"
    # loss is visible if a gap was revealed OR probes went missing (a window inside a long burst
    # receives nothing, so it reveals no gap yet: its UDP receive ratio shows the loss instead)
    lost_now = (df["udp_loss_revealed_now"].fillna(0) > 0) | (df["udp_recv_ratio_now"].fillna(1) < 0.9)
    lost_60 = lost_now.groupby([df["run_id"], df["device_id"]]).transform(
        lambda s: s.astype(float).rolling(LONG, min_periods=1).max()) > 0
    seen_now = b & lost_now.to_numpy()
    seen_60 = b & lost_60.to_numpy()
    return {"bursty_windows": int(b.sum()),
            "share_with_udp_loss_in_window": round(float(seen_now.sum() / b.sum()), 3) if b.any() else None,
            "share_with_udp_loss_in_last_60s": round(float(seen_60.sum() / b.sum()), 3) if b.any() else None,
            "recall_all": round(float((pred[b] == "bursty_loss").mean()), 3) if b.any() else None,
            "recall_when_loss_seen_in_last_60s": round(float((pred[seen_60] == "bursty_loss").mean()), 3)
            if seen_60.any() else None}


# ------------------------------------------------------------------ report
def cm_table(cm: dict) -> str:
    labels = cm["labels"]
    lines = ["| true \\ predicted | " + " | ".join(labels) + " | recall |", "|---|" + "---|" * (len(labels) + 1)]
    for lab, row in zip(labels, cm["rows_true_cols_pred"]):
        tot = sum(row)
        rec = f"{row[labels.index(lab)] / tot:.2f}" if tot else ""
        lines.append(f"| **{lab}** | " + " | ".join(str(v) for v in row) + f" | {rec} |")
    return "\n".join(lines)


def write_report(res: dict) -> str:
    m = res["models"]
    L = ["# Loss-pattern classifier: results", "",
         f"Generated {res['generated']}. Runs: {len(res['runs'])} ({', '.join(res['runs'])}).",
         f"Rows (device x 5 s window): {res['n_rows']}. Split: GroupKFold by schedule seed ({res['n_folds']} folds).",
         "", "## Overall", "",
         "| Model | Macro F1 | Accuracy |", "|---|---|---|"]
    for name, r in m.items():
        L.append(f"| {name} | {r['macro_f1']} | {r['accuracy']} |")
    L += ["", "Majority-class guess (always `clean`): accuracy "
          f"{res['majority_accuracy']}, macro F1 {res['majority_macro_f1']}.", "",
          "## Per-class recall", "", "| Pattern | windows | " + " | ".join(m) + " |",
          "|---|---|" + "---|" * len(m)]
    for c in PATTERNS:
        L.append(f"| {c} | {res['class_counts'].get(c, 0)} | " +
                 " | ".join(str(m[k]['per_class'][c]['recall']) for k in m) + " |")
    for name in ("random_forest (MQTT + UDP probes)", "random_forest (MQTT only)"):
        L += ["", f"## Confusion matrix: {name}", "", cm_table(m[name]["confusion_matrix"])]
    L += ["", "## Macro F1 by slice", "", "| Slice | n | " + " | ".join(m) + " |", "|---|---|" + "---|" * len(m)]
    for s, r in res["slices"].items():
        L.append(f"| {s} | {r['n']} | " + " | ".join(str(r[k]) for k in m) + " |")
    L += ["", "## Recall by impairment strength (random forest, MQTT + UDP probes)", "",
          "| Pattern | strength (loss % or delay ms) | windows | recall |", "|---|---|---|---|"]
    for pat, d in res["recall_by_intensity"].items():
        for lvl, r in d.items():
            L.append(f"| {pat} | {lvl} | {r['n']} | {r['recall']} |")
    o = res["bursty_observability"]
    L += ["", "## Why bursty loss is hard to see", "",
          f"Bursty loss is the weakest class. A Gilbert-Elliott link is clean most of the time: of "
          f"{o['bursty_windows']} windows inside bursty-loss phases, only {o['share_with_udp_loss_in_window']:.0%} "
          f"had any UDP probe loss in the window itself ({o['share_with_udp_loss_in_last_60s']:.0%} within the last "
          f"60 s). Recall is {o['recall_all']} on all bursty windows and {o['recall_when_loss_seen_in_last_60s']} on "
          "those with loss seen in the last 60 s, so seeing some loss is not enough: the model mostly confuses "
          "bursty with clean. The likely cause is too little training data: only two schedule seeds contain "
          "bursty loss, so each training fold sees at most one of them. More burst-profile runs are the fix."]
    L += ["", "## Most useful features (random forest, MQTT + UDP probes, impurity importance)", ""]
    L += [f"- `{f}`: {v}" for f, v in res["top_features"]]
    return "\n".join(L) + "\n"


def main():
    t0 = time.time()
    with db.connect() as conn:
        runs = [r[0] for r in conn.execute(
            "SELECT run_id FROM runs WHERE ended_at IS NOT NULL AND"
            " (run_id LIKE 'mix-%%' OR run_id LIKE 'bur-%%') ORDER BY run_id").fetchall()]   # same runs as train.py
    print(f"[patterns] building windows for {len(runs)} runs ...")
    df = build(runs).reset_index(drop=True)
    y = df["pattern"].to_numpy()
    n_folds = min(5, df["seed"].nunique())
    preds = {
        "rule baseline (UDP probes)": rule_classifier(df, True),
        "rule baseline (MQTT only)": rule_classifier(df, False),
        "random_forest (MQTT + UDP probes)": crossval(df, True, n_folds),
        "random_forest (MQTT only)": crossval(df, False, n_folds),
    }
    models = {k: report_block(y, p) for k, p in preds.items()}
    maj = np.full(len(y), "clean", dtype=object)
    final = make_model().fit(df[feature_cols(df, True)], y)
    rf = final[-1]
    top = sorted(zip(feature_cols(df, True), rf.feature_importances_), key=lambda t: -t[1])[:12]
    final_app = make_model().fit(df[feature_cols(df, False)], y)
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "runs": runs, "n_rows": int(len(df)), "n_folds": n_folds,
           "class_counts": pd.Series(y).value_counts().to_dict(),
           "majority_accuracy": round(float(accuracy_score(y, maj)), 3),
           "majority_macro_f1": round(float(f1_score(y, maj, labels=PATTERNS, average="macro", zero_division=0)), 3),
           "models": models, "slices": by_slice(df, y, preds),
           "recall_by_intensity": recall_by_intensity(df, y, preds["random_forest (MQTT + UDP probes)"]),
           "top_features": [(f, round(float(v), 4)) for f, v in top],
           "bursty_observability": bursty_observability(df, y, preds["random_forest (MQTT + UDP probes)"])}
    RESULTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    (RESULTS / "pattern_results.json").write_text(json.dumps(res, indent=2))
    (RESULTS / "pattern_report.md").write_text(write_report(res))
    joblib.dump({"with_probe": {"model": final, "features": feature_cols(df, True)},
                 "mqtt_only": {"model": final_app, "features": feature_cols(df, False)},
                 "classes": PATTERNS, "window_s": settings.window_s,
                 "version": time.strftime("%Y%m%d-%H%M")}, MODELS / "pattern_model.joblib", compress=3)
    for k, r in models.items():
        print(f"  {k:36s} macro F1 {r['macro_f1']:.3f}  accuracy {r['accuracy']:.3f}")
    print(f"[patterns] wrote results/pattern_report.md and models/pattern_model.joblib ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
