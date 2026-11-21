# ML results (2026-11-21 13:14)

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
| persistence | 0.451 | 0.6 | 0.585 | 0.328 | 0.42 | 0.312 | 30.0 | 58.9 |
| threshold_rule | 0.427 | 0.6 | 0.579 | 0.331 | 0.421 | 0.324 | 30.0 | 61.0 |
| logreg | 0.511 | 0.633 | 0.588 | 0.203 | 0.302 | 0.315 | 30.0 | 35.9 |
| random_forest | 0.524 | 0.667 | 0.734 | 0.124 | 0.213 | 0.162 | 30.0 | 11.4 |
| lightgbm | 0.508 | 0.663 | 0.604 | 0.143 | 0.231 | 0.208 | 30.0 | 23.7 |

Rows: 6680, positives: 2348 (prevalence 0.351), degradation onsets: 346.

### PR-AUC by QoS (cross-validation)

| Model | qos0 | qos1 |
|---|---|---|
| persistence | 0.259 (prev. 0.103) | 0.233 (prev. 0.111) |
| threshold_rule | 0.163 (prev. 0.103) | 0.16 (prev. 0.111) |
| logreg | 0.248 (prev. 0.103) | 0.268 (prev. 0.111) |
| random_forest | 0.39 (prev. 0.103) | 0.34 (prev. 0.111) |
| lightgbm | 0.303 (prev. 0.103) | 0.257 (prev. 0.111) |

### Early warning by onset kind (cross-validation): can it see a degradation coming?

Each degradation onset is tagged with what the fault injector was doing when it started. Gradual (ramp) degradations build up over about a minute; sudden ones switch on at once.

| Onset kind | onsets | persistence | threshold_rule | logreg | random_forest | lightgbm |
|---|---|---|---|---|---|---|
| gradual (ramp) | 88 | 0.034 (30 s) | 0.284 (10 s) | 0.136 (10 s) | 0.0 | 0.159 (28 s) |
| sudden loss | 57 | 0.07 (20 s) | 0.281 (18 s) | 0.281 (20 s) | 0.158 (20 s) | 0.281 (15 s) |
| sudden outage | 100 | 0.41 (30 s) | 0.43 (30 s) | 0.54 (30 s) | 0.41 (30 s) | 0.45 (30 s) |
| sudden stall | 120 | 0.208 (25 s) | 0.383 (30 s) | 0.725 (15 s) | 0.575 (15 s) | 0.683 (25 s) |

Cells: share of onsets with an alarm in the 30 s before (median lead time).


Deployed model: **random_forest** (best cross-validated PR-AUC), alert threshold 0.670.

### Top features of the deployed model (permutation importance, PR-AUC drop on held-out burst runs)

| Feature | Importance |
|---|---|
| `silence_s_now` | 0.019 |
| `connected_mean6` | 0.0089 |
| `connected_now` | 0.0061 |
| `online_events_mean6` | 0.0057 |
| `connected_min6` | 0.0056 |
| `offline_events_max6` | 0.0052 |
| `recv_ratio_slope6` | 0.0051 |
| `secs_since_offline` | 0.0047 |
| `udp_recv_ratio_now` | 0.0035 |
| `loss_revealed_mean6` | 0.0033 |
| `recv_ratio_now` | 0.0032 |
| `udp_recv_ratio_slope6` | 0.003 |
| `iat_std_ms_max6` | 0.0027 |
| `delay_p95_ms_max6` | 0.0026 |
| `offline_events_mean6` | 0.0022 |
