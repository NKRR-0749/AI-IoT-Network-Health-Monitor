# ML results (2026-11-20 06:11)

**Label: EMULATED (userspace impairment proxy); not real-hardware results.**

Task: y_degraded: on-time delivery ratio over next 30s < 0.9 (on time = relative delay <= 1000 ms).

### RQ2: 4-fold cross-validation over mixed runs (grouped by schedule seed), all features

| Model | pr_auc | roc_auc | precision | recall | f1 | early_warning_rate | median_lead_time_s | false_alarm_windows_per_device_hour |
|---|---|---|---|---|---|---|---|---|
| persistence | 0.245 | 0.603 | 0.362 | 0.228 | 0.279 | 0.2 | 25.0 | 31.0 |
| threshold_rule | 0.161 | 0.598 | 0.309 | 0.268 | 0.287 | 0.356 | 25.0 | 46.2 |
| logreg | 0.257 | 0.67 | 0.294 | 0.342 | 0.316 | 0.463 | 20.0 | 63.4 |
| random_forest | 0.363 | 0.675 | 0.33 | 0.294 | 0.311 | 0.326 | 30.0 | 46.2 |
| lightgbm | 0.278 | 0.655 | 0.256 | 0.325 | 0.286 | 0.43 | 30.0 | 73.1 |

Rows: 13965, positives: 1498 (prevalence 0.107), degradation onsets: 365.

### Ablation: application-level features only (no UDP probe features)

| Model | pr_auc | roc_auc | precision | recall | f1 | early_warning_rate | median_lead_time_s | false_alarm_windows_per_device_hour |
|---|---|---|---|---|---|---|---|---|
| persistence | 0.245 | 0.603 | 0.362 | 0.228 | 0.279 | 0.2 | 25.0 | 31.0 |
| threshold_rule | 0.161 | 0.598 | 0.309 | 0.268 | 0.287 | 0.356 | 25.0 | 46.2 |
| logreg | 0.257 | 0.669 | 0.265 | 0.394 | 0.317 | 0.559 | 20.0 | 84.5 |
| random_forest | 0.369 | 0.676 | 0.34 | 0.289 | 0.312 | 0.312 | 27.5 | 43.4 |
| lightgbm | 0.271 | 0.658 | 0.234 | 0.312 | 0.268 | 0.397 | 30.0 | 78.9 |

Rows: 13965, positives: 1498 (prevalence 0.107), degradation onsets: 365.

### RQ3: trained on mixed, tested on unseen bursty-loss runs

| Model | pr_auc | roc_auc | precision | recall | f1 | early_warning_rate | median_lead_time_s | false_alarm_windows_per_device_hour |
|---|---|---|---|---|---|---|---|---|
| persistence | 0.482 | 0.575 | 0.585 | 0.33 | 0.422 | 0.338 | 30.0 | 70.4 |
| threshold_rule | 0.47 | 0.579 | 0.577 | 0.331 | 0.421 | 0.348 | 30.0 | 72.9 |
| logreg | 0.539 | 0.63 | 0.577 | 0.213 | 0.311 | 0.308 | 30.0 | 46.9 |
| random_forest | 0.543 | 0.633 | 0.685 | 0.11 | 0.189 | 0.134 | 30.0 | 15.1 |
| lightgbm | 0.528 | 0.638 | 0.634 | 0.128 | 0.213 | 0.184 | 30.0 | 22.2 |

Rows: 3240, positives: 1351 (prevalence 0.417), degradation onsets: 201.

### PR-AUC by QoS (cross-validation)

| Model | qos0 | qos1 |
|---|---|---|
| persistence | 0.259 (prev. 0.103) | 0.233 (prev. 0.111) |
| threshold_rule | 0.163 (prev. 0.103) | 0.16 (prev. 0.111) |
| logreg | 0.248 (prev. 0.103) | 0.268 (prev. 0.111) |
| random_forest | 0.39 (prev. 0.103) | 0.34 (prev. 0.111) |
| lightgbm | 0.303 (prev. 0.103) | 0.257 (prev. 0.111) |

Deployed model: **random_forest** (best cross-validated PR-AUC), alert threshold 0.670.

### Top features of the deployed model (permutation importance, PR-AUC drop on held-out burst runs)

| Feature | Importance |
|---|---|
| `silence_s_now` | 0.011 |
| `loss_revealed_mean6` | 0.0083 |
| `connected_mean6` | 0.0073 |
| `loss_revealed_max6` | 0.0071 |
| `frac_over_ontime_max6` | 0.0059 |
| `jitter_ms_max6` | 0.0054 |
| `online_events_mean6` | 0.0053 |
| `recv_ratio_slope6` | 0.0044 |
| `jitter_ms_slope6` | 0.0041 |
| `secs_since_offline` | 0.003 |
| `connected_min6` | 0.0029 |
| `udp_recv_ratio_mean6` | 0.0026 |
| `jitter_ms_mean6` | 0.0022 |
| `connected_now` | 0.002 |
| `rtt_p95_ms_slope6` | 0.0019 |
