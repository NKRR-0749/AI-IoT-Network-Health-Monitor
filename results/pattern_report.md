# Loss-pattern classifier: results

Generated 2026-11-21 13:04. Runs: 24 (bur-s101-q0, bur-s101-q1, bur-s102-q0, bur-s102-q1, bur-s103-q0, bur-s103-q1, bur-s104-q0, bur-s104-q1, mix-s01-q0, mix-s01-q1, mix-s02-q0, mix-s02-q1, mix-s03-q0, mix-s03-q1, mix-s04-q0, mix-s04-q1, mix-s05-q0, mix-s05-q1, mix-s06-q0, mix-s06-q1, mix-s07-q0, mix-s07-q1, mix-s08-q0, mix-s08-q1).
Rows (device x 5 s window): 21870. Split: GroupKFold by schedule seed (5 folds).

## Overall

| Model | Macro F1 | Accuracy |
|---|---|---|
| rule baseline (UDP probes) | 0.546 | 0.776 |
| rule baseline (MQTT only) | 0.374 | 0.664 |
| random_forest (MQTT + UDP probes) | 0.832 | 0.897 |
| random_forest (MQTT only) | 0.786 | 0.874 |

Majority-class guess (always `clean`): accuracy 0.555, macro F1 0.119.

## Per-class recall

| Pattern | windows | rule baseline (UDP probes) | rule baseline (MQTT only) | random_forest (MQTT + UDP probes) | random_forest (MQTT only) |
|---|---|---|---|---|---|
| clean | 12140 | 0.923 | 0.93 | 0.941 | 0.934 |
| random_loss | 3750 | 0.675 | 0.0 | 0.906 | 0.877 |
| bursty_loss | 1800 | 0.057 | 0.012 | 0.529 | 0.409 |
| delay | 3420 | 0.753 | 0.775 | 0.932 | 0.91 |
| stall | 160 | 0.219 | 0.219 | 0.594 | 0.531 |
| outage | 600 | 0.867 | 0.867 | 0.948 | 0.927 |

## Confusion matrix: random_forest (MQTT + UDP probes)

| true \ predicted | clean | random_loss | bursty_loss | delay | stall | outage | recall |
|---|---|---|---|---|---|---|---|
| **clean** | 11421 | 250 | 374 | 78 | 17 | 0 | 0.94 |
| **random_loss** | 296 | 3396 | 49 | 9 | 0 | 0 | 0.91 |
| **bursty_loss** | 767 | 65 | 952 | 2 | 10 | 4 | 0.53 |
| **delay** | 46 | 107 | 78 | 3189 | 0 | 0 | 0.93 |
| **stall** | 0 | 1 | 62 | 0 | 95 | 2 | 0.59 |
| **outage** | 31 | 0 | 0 | 0 | 0 | 569 | 0.95 |

## Confusion matrix: random_forest (MQTT only)

| true \ predicted | clean | random_loss | bursty_loss | delay | stall | outage | recall |
|---|---|---|---|---|---|---|---|
| **clean** | 11333 | 270 | 440 | 80 | 14 | 3 | 0.93 |
| **random_loss** | 401 | 3290 | 54 | 0 | 5 | 0 | 0.88 |
| **bursty_loss** | 917 | 108 | 737 | 0 | 28 | 10 | 0.41 |
| **delay** | 55 | 234 | 20 | 3111 | 0 | 0 | 0.91 |
| **stall** | 3 | 0 | 72 | 0 | 85 | 0 | 0.53 |
| **outage** | 44 | 0 | 0 | 0 | 0 | 556 | 0.93 |

## Macro F1 by slice

| Slice | n | rule baseline (UDP probes) | rule baseline (MQTT only) | random_forest (MQTT + UDP probes) | random_forest (MQTT only) |
|---|---|---|---|---|---|
| qos0 | 10940 | 0.553 | 0.381 | 0.828 | 0.788 |
| qos1 | 10930 | 0.539 | 0.366 | 0.836 | 0.785 |
| pure_windows (one phase covers the whole window) | 16680 | 0.578 | 0.407 | 0.738 | 0.707 |
| transition_windows | 5190 | 0.459 | 0.286 | 0.793 | 0.752 |

## Recall by impairment strength (random forest, MQTT + UDP probes)

| Pattern | strength (loss % or delay ms) | windows | recall |
|---|---|---|---|
| random_loss | 1 | 240 | 0.217 |
| random_loss | 5 | 360 | 0.742 |
| random_loss | 10 | 1020 | 0.947 |
| random_loss | 15 | 350 | 1.0 |
| random_loss | 20 | 370 | 0.997 |
| random_loss | 25 | 410 | 1.0 |
| random_loss | 30 | 1000 | 0.982 |
| bursty_loss | 1 | 140 | 0.043 |
| bursty_loss | 5 | 580 | 0.445 |
| bursty_loss | 10 | 140 | 0.529 |
| bursty_loss | 15 | 120 | 0.7 |
| bursty_loss | 20 | 260 | 0.627 |
| bursty_loss | 25 | 120 | 0.733 |
| bursty_loss | 30 | 440 | 0.634 |
| delay | 100 | 460 | 0.954 |
| delay | 133 | 390 | 0.895 |
| delay | 267 | 390 | 1.0 |
| delay | 300 | 440 | 1.0 |
| delay | 400 | 330 | 1.0 |
| delay | 533 | 370 | 1.0 |
| delay | 600 | 160 | 0.906 |
| delay | 667 | 350 | 1.0 |
| delay | 800 | 350 | 0.966 |
| delay | 1200 | 180 | 0.211 |

## Why bursty loss is hard to see

Bursty loss is the weakest class. A Gilbert-Elliott link is clean most of the time: of 1800 windows inside bursty-loss phases, only 21% had any UDP probe loss in the window itself (63% within the last 60 s). Recall is 0.529 on all bursty windows and 0.546 on those with loss seen in the last 60 s, so seeing some loss is not enough: the model mostly confuses bursty with clean. The likely cause is too little training data: only two schedule seeds contain bursty loss, so each training fold sees at most one of them. More burst-profile runs are the fix.

## Most useful features (random forest, MQTT + UDP probes, impurity importance)

- `udp_recv_ratio_now`: 0.0477
- `connected_mean6`: 0.033
- `udp_recv_ratio_slope6`: 0.0323
- `secs_since_offline`: 0.0317
- `connected_now`: 0.0293
- `rtt_mean_ms_now`: 0.0291
- `connected_min6`: 0.025
- `udp_recv_ratio_min6`: 0.0246
- `rtt_p95_ms_now`: 0.0235
- `delay_max_ms_now`: 0.0234
- `delay_mean_ms_now`: 0.0224
- `delay_p95_ms_now`: 0.0224
