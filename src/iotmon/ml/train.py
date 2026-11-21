"""Block: train and evaluate degradation predictors (RQ2, RQ3).

Task  : at time t, predict y_degraded = "on-time delivery over the next 30 s < 0.9".
Split : by SCHEDULE SEED. The QoS 0 and QoS 1 runs of one seed share the same
        impairment schedule, so they always land on the same side of a split.
        4-fold GroupKFold over the 'mixed' runs -> every mixed row gets an
        out-of-fold prediction from a model that never saw its schedule.
RQ3   : train on ALL mixed runs, test on the held-out 'burst' runs
        (bursty loss never appears in training).

Models compared (each must beat the ones above it to be worth using):
  1. persistence   - "the next 30 s will look like the last 5 s"
  2. threshold     - fixed rule on the last window (recv ratio < 0.9 or delay p95 > 1 s)
  3. logistic regression
  4. random forest
  5. LightGBM (gradient boosting)

Metrics: PR-AUC (main; robust when positives are rare), ROC-AUC, precision /
recall / F1 at a threshold chosen on TRAINING data only, early-warning
lead time, false alarms per device-hour.

Run:  python -m iotmon.ml.train
Writes: results/ml_results.json, results/ml_report.md, models/degradation_model.joblib
"""
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import db
from ..config import ROOT, settings
from .dataset import build, feature_columns

warnings.filterwarnings("ignore")
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
TARGET = "y_degraded"
SEED = 0


# ------------------------------------------------------------------ models
def persistence_score(X: pd.DataFrame) -> np.ndarray:
    """1 - (share of expected messages that arrived on time in the last window)."""
    ontime_share = X["recv_ratio_now"].fillna(0).clip(0, 1) * (1 - X["frac_over_ontime_now"].fillna(0))
    return 1 - ontime_share.to_numpy()


def threshold_rule(X: pd.DataFrame) -> np.ndarray:
    return ((X["recv_ratio_now"].fillna(0) < 0.9) | (X["delay_p95_ms_now"].fillna(0) > 1000)).astype(float).to_numpy()


def make_model(name: str):
    if name == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight="balanced"))
    if name == "random_forest":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1,
                                                    class_weight="balanced_subsample", random_state=SEED))
    if name == "lightgbm":
        return LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)
    raise ValueError(name)


ML_MODELS = ["logreg", "random_forest", "lightgbm"]


def best_f1_threshold(y, s) -> float:
    """Pick the score threshold with the best F1 (only ever called on TRAINING predictions)."""
    cands = np.unique(np.quantile(s, np.linspace(0.5, 0.995, 80)))
    f1s = [f1_score(y, s >= c, zero_division=0) for c in cands]
    return float(cands[int(np.argmax(f1s))])


def oof_scores(name, X, y, groups, n_splits=3) -> np.ndarray:
    """Out-of-fold scores inside a training set (used to choose the threshold honestly)."""
    out = np.zeros(len(y))
    if len(np.unique(groups)) < 2:              # too few groups to split: fall back to in-sample scores
        return make_model(name).fit(X, y).predict_proba(X)[:, 1]
    for tr, te in GroupKFold(n_splits=min(n_splits, len(np.unique(groups)))).split(X, y, groups):
        m = make_model(name).fit(X.iloc[tr], y[tr])
        out[te] = m.predict_proba(X.iloc[te])[:, 1]
    return out


# ------------------------------------------------------------------ evaluation
def event_metrics(df: pd.DataFrame, pred: np.ndarray, window_s: int, horizon_s: int,
                  onset_log: list | None = None) -> dict:
    """Early warning: for each degradation ONSET, was there an alarm in the horizon before it?
    onset_log (optional) collects one record per onset: (run_id, device_id, onset time, warned, lead s)."""
    d = df[["run_id", "device_id", "window_start", "target_otdr", "target_next_otdr"]].copy()
    d["pred"] = pred
    caught, leads, onsets = 0, [], 0
    false_alarms, hours = 0, 0.0
    for (rid, dev), g in d.sort_values("window_start").groupby(["run_id", "device_id"]):
        ws = g["window_start"].to_numpy()
        # state of the single window right after each row (the first window of its horizon)
        now_bad = g["target_next_otdr"].fillna(1).to_numpy() < 0.9
        horizon_ok = g["target_otdr"].to_numpy() >= 0.9       # label 0: nothing bad in the whole horizon
        p = g["pred"].to_numpy()
        hours += len(g) * window_s / 3600
        false_alarms += int(((p == 1) & horizon_ok).sum())
        for i in range(3, len(g)):
            if now_bad[i] and not now_bad[i - 3:i].any():          # onset after >= 15 s of good
                onsets += 1
                t_on = ws[i]
                early = np.where((ws < t_on) & (ws >= t_on - horizon_s) & (p == 1))[0]
                if len(early):
                    caught += 1
                    leads.append(float(t_on - ws[early[0]]))
                if onset_log is not None:
                    onset_log.append((rid, dev, float(t_on), bool(len(early)),
                                      float(t_on - ws[early[0]]) if len(early) else None))
    return {"onsets": onsets, "onsets_warned_early": caught,
            "early_warning_rate": round(caught / onsets, 3) if onsets else None,
            "median_lead_time_s": float(np.median(leads)) if leads else None,
            "false_alarm_windows_per_device_hour": round(false_alarms / hours, 1) if hours else None}


def score_block(df, y, score, binary, window_s, horizon_s) -> dict:
    r = {"n": int(len(y)), "positives": int(y.sum()), "prevalence": round(float(y.mean()), 3),
         "pr_auc": round(float(average_precision_score(y, score)), 3) if y.any() else None,
         "roc_auc": round(float(roc_auc_score(y, score)), 3) if 0 < y.sum() < len(y) else None,
         "precision": round(float(precision_score(y, binary, zero_division=0)), 3),
         "recall": round(float(recall_score(y, binary, zero_division=0)), 3),
         "f1": round(float(f1_score(y, binary, zero_division=0)), 3),
         "confusion_matrix [[tn, fp], [fn, tp]]": confusion_matrix(y, binary, labels=[0, 1]).tolist()}
    r.update(event_metrics(df, binary, window_s, horizon_s))
    return r


def evaluate(train_df, test_df, feats, window_s, horizon_s) -> dict:
    """Fit every model on train_df, score test_df. Thresholds come from training data only."""
    Xtr, ytr = train_df[feats], train_df[TARGET].to_numpy()
    Xte, yte = test_df[feats], test_df[TARGET].to_numpy()
    groups = train_df["seed"].to_numpy()
    out, scores = {}, {}
    s_tr = persistence_score(Xtr)
    thr = best_f1_threshold(ytr, s_tr)
    s = persistence_score(Xte)
    out["persistence"], scores["persistence"] = score_block(test_df, yte, s, (s >= thr).astype(int), window_s, horizon_s), s
    out["persistence"]["threshold"] = round(thr, 4)
    s = threshold_rule(Xte)
    out["threshold_rule"], scores["threshold_rule"] = score_block(test_df, yte, s, s.astype(int), window_s, horizon_s), s
    out["threshold_rule"]["threshold"] = 0.5
    for name in ML_MODELS:
        thr = best_f1_threshold(ytr, oof_scores(name, Xtr, ytr, groups))
        model = make_model(name).fit(Xtr, ytr)
        s = model.predict_proba(Xte)[:, 1]
        out[name] = score_block(test_df, yte, s, (s >= thr).astype(int), window_s, horizon_s)
        out[name]["threshold"] = round(thr, 4)
        scores[name] = s
    return out, scores


def crossval(df, feats, window_s, horizon_s, n_splits=4):
    """GroupKFold over seeds; pool the out-of-fold predictions and score them once."""
    oof = {m: np.zeros(len(df)) for m in ["persistence", "threshold_rule"] + ML_MODELS}
    oof_bin = {m: np.zeros(len(df), dtype=int) for m in oof}
    for tr, te in GroupKFold(n_splits=n_splits).split(df, df[TARGET], df["seed"]):
        res, scores = evaluate(df.iloc[tr], df.iloc[te], feats, window_s, horizon_s)
        for m in oof:
            oof[m][te] = scores[m]
            oof_bin[m][te] = (scores[m] >= res[m]["threshold"]).astype(int)
    y = df[TARGET].to_numpy()
    return {m: score_block(df, y, oof[m], oof_bin[m], window_s, horizon_s) for m in oof}, oof, oof_bin


# ------------------------------------------------------------------ onset kinds (gradual vs sudden)
def onset_kind(phases: pd.DataFrame, t_bad: float, window_s: int) -> str:
    """What the fault injector was doing in the first bad window [t_bad, t_bad + window_s): 'gradual'
    for ramp phases, otherwise the sudden phase kind. If only clean overlaps it, the last fault that
    ended in the 10 s before (TCP keeps suffering for a moment after a fault is removed)."""
    cur = phases[(phases["start"] < t_bad + window_s) & (phases["end"] > t_bad) & (phases["phase"] != "clean")]
    if len(cur):
        name = cur.iloc[0]["phase"]
    else:
        prev = phases[(phases["phase"] != "clean") & (phases["end"] <= t_bad) & (phases["end"] > t_bad - 10)]
        if not len(prev):
            return "other"
        name = prev.iloc[-1]["phase"]
    if name.startswith("ramp_"):
        return "gradual (ramp)"
    return {"random_loss": "sudden loss", "burst_loss": "sudden loss", "delay": "sudden delay",
            "outage_stall": "sudden stall", "outage_drop": "sudden outage"}.get(name, "other")


def onsets_by_kind(df, oof_bin: dict, window_s: int, horizon_s: int) -> dict:
    with db.connect() as conn:
        rows = conn.execute("SELECT run_id, phase, extract(epoch from start_at), extract(epoch from end_at)"
                            " FROM impairments WHERE end_at IS NOT NULL ORDER BY start_at").fetchall()
    ph = pd.DataFrame(rows, columns=["run_id", "phase", "start", "end"]).astype({"start": float, "end": float})
    by_run = {r: g.reset_index(drop=True) for r, g in ph.groupby("run_id")}
    out = {}
    for m, b in oof_bin.items():
        log: list = []
        event_metrics(df, b, window_s, horizon_s, onset_log=log)
        rec = pd.DataFrame(log, columns=["run_id", "device_id", "t_on", "warned", "lead"])
        # the onset row's horizon starts one window later: that window is the first bad one
        rec["kind"] = [onset_kind(by_run.get(r, ph.iloc[0:0]), t + window_s, window_s) for r, t in zip(rec["run_id"], rec["t_on"])]
        for k, g in rec.groupby("kind"):
            out.setdefault(k, {})[m] = {"onsets": int(len(g)), "early_warning_rate": round(float(g["warned"].mean()), 3),
                                        "median_lead_time_s": float(g["lead"].median()) if g["warned"].any() else None}
    return out


def by_qos(df, oof, bins_from, window_s, horizon_s) -> dict:
    out = {}
    for q in sorted(df["qos"].unique()):
        mask = (df["qos"] == q).to_numpy()
        y = df[TARGET].to_numpy()[mask]
        out[f"qos{int(q)}"] = {m: {"prevalence": round(float(y.mean()), 3),
                                   "pr_auc": round(float(average_precision_score(y, oof[m][mask])), 3) if y.any() else None}
                               for m in bins_from}
    return out


# ------------------------------------------------------------------ main
def table(results: dict, title: str) -> str:
    cols = ["pr_auc", "roc_auc", "precision", "recall", "f1", "early_warning_rate",
            "median_lead_time_s", "false_alarm_windows_per_device_hour"]
    lines = [f"### {title}", "", "| Model | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for m, r in results.items():
        lines.append(f"| {m} | " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    first = next(iter(results.values()))
    lines.append("")
    lines.append(f"Rows: {first['n']}, positives: {first['positives']} (prevalence {first['prevalence']}), "
                 f"degradation onsets: {first['onsets']}.")
    return "\n".join(lines)


def main():
    t0 = time.time()
    window_s, horizon_s = settings.window_s, settings.horizon_s
    with db.connect() as conn:
        runs = [r[0] for r in conn.execute("SELECT run_id FROM runs WHERE ended_at IS NOT NULL AND"
                                           " (run_id LIKE 'mix-%%' OR run_id LIKE 'bur-%%') ORDER BY run_id")]
    print(f"[train] building dataset from {len(runs)} runs ...")
    df = build(runs)
    mixed = df[df["profile"] == "mixed"].reset_index(drop=True)
    burst = df[df["profile"] == "burst"].reset_index(drop=True)
    feats = feature_columns(df, with_probe=True)
    feats_app = feature_columns(df, with_probe=False)
    print(f"[train] mixed rows={len(mixed)} burst rows={len(burst)} features={len(feats)}")

    cv, oof, oof_bin = crossval(mixed, feats, window_s, horizon_s)
    print("[train] cross-validation done")
    cv_app, _, _ = crossval(mixed, feats_app, window_s, horizon_s)
    by_kind = onsets_by_kind(mixed, oof_bin, window_s, horizon_s)
    heldout, _ = evaluate(mixed, burst, feats, window_s, horizon_s) if len(burst) else ({}, {})
    print("[train] held-out burst test done")

    # final model: the ML model with the best cross-validated PR-AUC, refit on all mixed runs;
    # its threshold comes from the out-of-fold scores (never from in-sample predictions)
    best = max(ML_MODELS, key=lambda m: cv[m]["pr_auc"] or 0)
    y = mixed[TARGET].to_numpy()
    thr = best_f1_threshold(y, oof[best])
    final = make_model(best).fit(mixed[feats], y)
    MODELS.mkdir(exist_ok=True)
    joblib.dump({"model": final, "model_name": best, "features": feats, "threshold": thr, "window_s": window_s,
                 "horizon_s": horizon_s, "trained_on": sorted(mixed["run_id"].unique().tolist()),
                 "version": time.strftime("%Y%m%d-%H%M%S")}, MODELS / "degradation_model.joblib", compress=3)

    # which features matter (permutation importance on the held-out burst runs)
    imp_src = burst if len(burst) else mixed
    pi = permutation_importance(final, imp_src[feats], imp_src[TARGET], scoring="average_precision",
                                n_repeats=5, random_state=SEED, n_jobs=-1)
    importance = sorted(zip(feats, pi.importances_mean), key=lambda t: -t[1])[:15]

    results = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "label": "EMULATED (userspace impairment proxy); not real-hardware results",
        "task": f"{TARGET}: on-time delivery ratio over next {horizon_s}s < 0.9 (on time = relative delay <= {settings.ontime_ms:.0f} ms)",
        "runs_mixed": sorted(mixed["run_id"].unique().tolist()),
        "runs_burst_heldout": sorted(burst["run_id"].unique().tolist()),
        "crossval_mixed_all_features": cv,
        "crossval_mixed_app_features_only": cv_app,
        "heldout_burst_trained_on_mixed": heldout,
        "by_qos_crossval_pr_auc": by_qos(mixed, oof, list(oof), window_s, horizon_s),
        "early_warning_by_onset_kind_crossval": by_kind,
        "permutation_importance_top15": [[f, round(float(v), 4)] for f, v in importance],
        "final_model": best,
        "final_model_threshold": thr,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "ml_results.json").write_text(json.dumps(results, indent=2))

    md = [f"# ML results ({results['generated']})", "",
          f"**Label: {results['label']}.**", "", f"Task: {results['task']}.", "",
          table(cv, "RQ2: 4-fold cross-validation over mixed runs (grouped by schedule seed), all features"), "",
          table(cv_app, "Ablation: application-level features only (no UDP probe features)"), ""]
    if heldout:
        md += [table(heldout, "RQ3: trained on mixed, tested on unseen bursty-loss runs"), ""]
    md += ["### PR-AUC by QoS (cross-validation)", "", "| Model | " + " | ".join(results["by_qos_crossval_pr_auc"]) + " |",
           "|---|" + "---|" * len(results["by_qos_crossval_pr_auc"])]
    for m in oof:
        md.append(f"| {m} | " + " | ".join(f"{results['by_qos_crossval_pr_auc'][q][m]['pr_auc']} "
                                           f"(prev. {results['by_qos_crossval_pr_auc'][q][m]['prevalence']})"
                                           for q in results["by_qos_crossval_pr_auc"]) + " |")
    md += ["", "### Early warning by onset kind (cross-validation): can it see a degradation coming?", "",
           "Each degradation onset is tagged with what the fault injector was doing when it started. "
           "Gradual (ramp) degradations build up over about a minute; sudden ones switch on at once.", "",
           "| Onset kind | onsets | " + " | ".join(oof) + " |", "|---|---|" + "---|" * len(oof)]
    for k in sorted(by_kind, key=lambda k: (k != "gradual (ramp)", k)):
        n = next(iter(by_kind[k].values()))["onsets"]
        md.append(f"| {k} | {n} | " + " | ".join(
            f"{by_kind[k][m]['early_warning_rate']}" + (f" ({by_kind[k][m]['median_lead_time_s']:.0f} s)"
                                                         if by_kind[k][m]['median_lead_time_s'] else "")
            for m in oof) + " |")
    md += ["", "Cells: share of onsets with an alarm in the 30 s before (median lead time).", ""]
    md += ["", f"Deployed model: **{best}** (best cross-validated PR-AUC), alert threshold {thr:.3f}.", "",
           f"### Top features of the deployed model (permutation importance, PR-AUC drop on held-out burst runs)", "",
           "| Feature | Importance |", "|---|---|"] + [f"| `{f}` | {v} |" for f, v in results["permutation_importance_top15"]]
    (RESULTS / "ml_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"[train] done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
