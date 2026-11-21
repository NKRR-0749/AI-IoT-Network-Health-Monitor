# Block-by-block guide

This guide walks through the project one block at a time, in the order it was
built. For each block: **what it does, why it exists, which files, how to run
it, how to check it works, and the mistakes to watch for.** Read it top to
bottom once, then use it as a reference.

Labels used everywhere:

| Label | Meaning |
|---|---|
| **Implemented** | Code exists and passes its tests |
| **Simulated** | Runs in Wokwi; network behaviour is not realistic |
| **Emulated** | Real software stack with injected, known impairments |
| **Measured** | Real hardware on a real network |
| **Planned** | Not built yet |

---

## The big picture

```
 Wokwi ESP32 (simulated) ──────────────────────────────┐
 Python emulator (N devices) ─► Fault injector ───────►│ Mosquitto ─► Collector ─► PostgreSQL ─► windows ─► ML ─► live predictor ─► dashboard
 Physical ESP32 (planned) ─────────────────────────────┘                  ▲                       │
                                    │ ground truth (what was injected) ───┘                       └─► alerts
```

The key idea: **MQTT runs over TCP, and TCP hides most IP packet loss** by
resending lost packets. So the project measures three different things and
never mixes them up:

| Layer | Measured by | Called |
|---|---|---|
| Application | gaps in message sequence numbers | message loss / delivery ratio |
| Application | ping → echo → pong round trip | application RTT |
| Network | UDP probe packets (no retransmission) | network-layer packet loss |

---

## Block 0: Settings and the database schema

**Files:** `src/iotmon/config.py`, `db/schema.sql`, `src/iotmon/db.py`
**Status:** Implemented

**What it does.** `config.py` holds every setting (database address, MQTT
port, window length...) in one place. Each setting can be changed with an
environment variable or a `.env` file. `schema.sql` creates the tables, and
`db.py` runs it.

**Why.** If ports and passwords are scattered across files, changing one means
hunting through all of them. And the database design decides what you can
analyse later.

**The tables and why each exists:**

| Table | Holds | Why |
|---|---|---|
| `runs` | one row per experiment (config, seed, git commit) | every result can be traced back to exactly how it was produced |
| `devices` | one row per device, with kind (wokwi/emulated/physical) | labels data as simulated/emulated/measured |
| `messages` | every telemetry message received + **our** receive time | raw data; never edited |
| `rtt_probes` | ping round-trip times (NULL = lost) | application RTT |
| `udp_probes` | UDP probes that arrived | network-layer loss = gaps in `probe_seq` |
| `connection_events` | online/offline | downtime, reconnects |
| `impairments` | what the fault injector did, and when | **ground truth**; never a model feature |
| `windows` | per-device metrics every 5 s, `observed` and `truth` | features and labels |
| `predictions`, `alerts` | model outputs and alert lifecycle | evaluation and dashboard |

`messages` has `UNIQUE (run_id, device_id, boot_id, seq)`, so a duplicate
delivery cannot be stored twice.

**Run:** `python -m iotmon.db init` (safe to run again). `python -m iotmon.db reset` deletes everything.

**Common mistakes**
- *"connection refused" on port 5432*: PostgreSQL isn't running, or the password in `.env` is wrong.
- Times stored without a timezone. Every column here is `timestamptz` (UTC), on purpose.

---

## Block 1: The message format

**File:** `src/iotmon/protocol.py`  **Status:** Implemented

Every device (ESP32 or emulated) sends this JSON on `iot/<run>/<device>/telemetry`:

```json
{"device_id": "dev-01", "seq": 1042, "boot_id": 7, "uptime_ms": 1042331, "qos": 0, "interval_ms": 500, "rssi": -61}
```

- `seq` goes up by 1 for every message. A missing number means a missing message.
- `boot_id` goes up on every reboot. Without it, a reboot (`seq` back to 0) would look like thousands of lost messages.
- `uptime_ms` is the device's own clock. It is used to work out *when* a message was sent.

`decode_telemetry()` checks the fields and raises `BadMessage` if anything is
wrong, so one broken message can't crash the collector.

---

## Block 2: Sequence-gap detection

**File:** `src/iotmon/gaps.py`  **Tests:** `tests/test_gaps.py`  **Status:** Implemented

**What it does.** `SequenceTracker.observe()` classifies each arriving message:

| Sees | Result |
|---|---|
| 41 then 42 | `in_order` |
| 41 then 44 | `gap`, missing = 2 |
| 42 arrives after 44 | `late` (it wasn't lost, just delayed) |
| 42 twice | `duplicate` |
| seq 0 with a higher boot_id | `reboot`, no loss counted |

**Why it's separate.** It is pure Python with no MQTT or database, so it can be
tested in milliseconds with exact expected answers.

**Check:** `pytest tests/test_gaps.py -v` (6 tests).

**Common mistake.** Counting a gap as final loss straight away. A gap only
means "missing so far". QoS 1 messages often turn up later.

---

## Block 3: The collector (the Python backend)

**File:** `src/iotmon/collector.py`  **Status:** Implemented, tested end to end

**What it does.**
1. Subscribes to `iot/+/+/telemetry`, `ping`, `rtt`, `status`.
2. Records **its own receive time** first, then decodes the message.
3. Runs gap detection and prints gaps as they happen.
4. Echoes every `ping` straight back on `pong` (devices time the round trip).
5. Listens on UDP port 9999 for probe packets.
6. Writes to PostgreSQL on a **separate thread**, in batches.

**Why a separate writer thread?** The MQTT callback must return quickly. If it
waited for a slow database insert, messages would pile up inside the client,
and that would add delay you'd wrongly blame on the network.

**Run:** `python -m iotmon.collector` (add `--quiet` to hide per-gap lines).
Every 10 s it prints: `received=… gaps=… missing=… late=… dup=… udp=…`.

**Check (no device needed):**
```
mosquitto_pub -t iot/test/dev-x/telemetry -m '{"device_id":"dev-x","seq":0,"boot_id":1,"uptime_ms":0}'
mosquitto_pub -t iot/test/dev-x/telemetry -m '{"device_id":"dev-x","seq":3,"boot_id":1,"uptime_ms":3000}'
```
The collector should print `GAP test/dev-x: 2 message(s) missing before seq 3`.

**Common mistakes**
- Running two collectors at once: both would answer pings and insert rows.
- Retained `status` messages: on subscribe, the broker replays the last stored "online", which isn't a new event. The collector ignores retained messages for that reason.

---

## Block 4: ESP32 firmware (Wokwi)

**Files:** `firmware/esp32_telemetry/` (`.ino`, `diagram.json`, `libraries.txt`, `wokwi.toml`)
**Status:** Written; **not compiled in the build environment** (Arduino's download servers were blocked there). Compile it in Wokwi. **Simulated** when run in Wokwi.

**What it does.** The same protocol as the emulator: numbered telemetry every
1 s, an RTT ping every 2 s, an "online" status with a Last Will of "offline",
`boot_id` kept in flash (`Preferences`), and optional UDP probes.

**Run in the Wokwi web simulator**
1. New ESP32 project at wokwi.com. Paste the `.ino`, `diagram.json` and `libraries.txt`.
2. Set `MQTT_HOST`. The web simulator cannot reach your laptop, so use
   `test.mosquitto.org` for a first test (public: anyone can read it), and run the
   collector with `IOT_MQTT_HOST=test.mosquitto.org`.
3. To reach Mosquitto on your laptop, use **Wokwi for VS Code** with the private
   gateway (`wokwi.toml`). The board then reaches your PC as `host.wokwi.internal`.
   Check Wokwi's docs for whether your plan includes the private gateway.

**Expected serial output:** `boot_id=1`, `Wi-Fi: connected`, `MQTT: connecting ... ok`, `seq=0 sent`, `seq=1 sent`, …

**Limits to state honestly**
- PubSubClient can only publish at QoS 0.
- Wokwi's Wi-Fi and RSSI are simulated, so Wokwi data is not evidence about real wireless behaviour.

---

## Block 5: The device emulator

**File:** `src/iotmon/emulator.py`  **Status:** Implemented. **Emulated.**

**What it does.** Runs N virtual devices using the real `paho-mqtt` client.
Each one publishes telemetry every 0.5 s, pings every 2 s, and sends a UDP probe
every 0.5 s.

**Why.** Experiments need many devices behind a fault injector, run many
times. Wokwi gives you one simulated board.

**QoS behaviour is real client behaviour, not something we modelled:**
- QoS 0 published while disconnected → **dropped** (a gap appears)
- QoS 1 published while disconnected → **queued** and sent after reconnecting (late, not lost)

**Run:** `python -m iotmon.emulator --run-id demo --devices 3` (Ctrl+C to stop).

**Check:** with the collector running, `received=` should rise by about 6 per second for 3 devices.

---

## Block 6: Windowed metrics

**Files:** `src/iotmon/metrics.py`, `src/iotmon/windows.py`
**Tests:** `tests/test_metrics.py`, `tests/test_windows.py`  **Status:** Implemented

**What it does.** Groups raw rows into 5-second windows per device. There are
**two kinds**, and keeping them apart is what makes the ML honest:

| | `observed` | `truth` |
|---|---|---|
| grouped by | **receive** time | **send** time |
| uses | only what had arrived by the window's end | hindsight (a message arriving 20 s late still counts for when it was sent) |
| used for | model **features**, live dashboard | model **labels**, evaluation |

**Formulas** (each checked by hand in the tests):
- Delivery ratio: `PDR = unique received / expected`
- On-time delivery ratio: `OTDR = received with relative delay ≤ 1 s / expected`
- Jitter (RFC 3550): `J = J + (|D| − J) / 16`
- Relative delay: device and laptop clocks aren't synchronised, so we use
  `transit = receive_time − device_time` minus the smallest transit seen for that boot.
  That's "how much slower than the fastest message", which needs no clock sync.
  One-way latency itself is **not** claimed.
- Missing messages' send times are interpolated between their neighbours.

**Run:** `python -m iotmon.windows --run-id demo`

**Common mistake.** Grouping truth by receive time. A message sent during an
outage but delivered afterwards would then count as "fine" in the wrong window.

---

## Block 7: The fault injector

**Files:** `src/iotmon/impair/models.py`, `proxy.py`, `netem.py`, `schedule.py`
**Tests:** `tests/test_loss_models.py`, `tests/test_schedule_netem.py`

### 7a. Loss models (Implemented)
- **Random (Bernoulli):** every packet is lost with probability p.
- **Gilbert–Elliott (bursty):** a two-state chain. GOOD → BAD with probability p,
  BAD → GOOD with probability r, and every packet in BAD is lost.
  For a target loss L and mean burst length B: `r = 1/B`, `p = L·r/(1−L)`.
  The test checks both the mean loss and the burst length over 200,000 packets.

### 7b. Userspace impairment proxy (Implemented, **Emulated: userspace model**)
Devices connect to the proxy (port 18830); it forwards to Mosquitto (1883).

| Impairment | TCP (MQTT) path | UDP (probe) path |
|---|---|---|
| delay ± jitter | each chunk held for that long | each datagram delayed |
| loss | **modelled as TCP sees it**: data held for a retransmission timeout (≥ 200 ms, doubling on repeats), in order, head-of-line blocking | datagram really dropped |
| outage `stall` | data waits until the link is back | dropped |
| outage `drop` | connections closed, new ones refused | dropped |

**Say this in the paper:** the TCP loss behaviour is a *model* of TCP's reaction
to loss, not real IP loss. It works on Windows with no admin rights, which is why
it is the default.

### 7c. netem backend (Written, **not run**: the build machine's kernel had no netem)
Real kernel-level IP loss and delay on Linux/WSL2. Only traffic on port 1884 and
UDP 9999 is impaired, so the collector's own link stays clean. **Use netem for the
results you report in the paper** (see "Running with netem" in the README). Do a
small test run first.

### 7d. Schedules (Implemented)
A run is a list of phases: clean → impairment → clean → … Generated from a
**profile + seed**, so `(mixed, seed 3)` always gives the same schedule.
- `mixed`: random loss (1–30%), delay (100–1200 ms), stalls (3–10 s), drops (5–30 s), gradual ramps
- `burst`: the same, but loss is always bursty. **Held out from training** for the RQ3 test.

---

## Block 8: Experiment runner and validation

**Files:** `src/iotmon/experiment.py`, `src/iotmon/validate.py`, `experiments/plan.yaml`
**Status:** Implemented; 20 runs + 1 pilot executed (**Emulated**, userspace proxy)

**What it does.** For each run: create the `runs` row → start the proxy → start
the emulator through the proxy → apply each phase and write it to `impairments`
→ compute windows. `batch` runs several in parallel on different ports.

**Validation (the Phase 2 gate).** `python -m iotmon.validate --run-id <run>`
prints, for every phase, what was injected next to what was measured.
`results/validation_*.md` has examples, and `python -m iotmon.validate --summary 'mix-%'` averages over many
runs (`results/rq1_summary.md`). What they show:
- UDP probe loss tracks the injected loss (for example 28.4% measured at 30% injected).
- MQTT message loss stays near 0 during random loss, because TCP hides it as delay (RTT goes up).
- A 6 s stall: QoS 0 lost about 30% of messages (keepalive timeout → disconnect → offline publishes dropped). QoS 1 lost 0%.
- Connection drops: QoS 0 lost 100% of messages during the drop; QoS 1 lost 0% (delivered late).
- Long loss bursts (held-out runs) broke connections: QoS 0 lost about 17%, QoS 1 lost 0%.

If these didn't line up, the testbed would be broken and no ML result could be trusted.

---

## Block 9: Machine learning

**Files:** `src/iotmon/ml/dataset.py`, `src/iotmon/ml/train.py`
**Tests:** `tests/test_no_leakage.py`  **Status:** Implemented; results in `results/ml_report.md` (**Emulated**)

**Task.** At each 5 s window end t, for each device: will the on-time delivery
ratio over the next 30 s fall below 0.9? (`y_degraded`)

**Features** (only from `observed` windows up to t): receive ratio, revealed
loss, relative delay mean/p95/max, inter-arrival stats, jitter, silence time,
RTT, lost pings, connection events, UDP probe ratio. Each is given as now,
previous window, 30 s mean/max/min, and slope. Plus QoS.

**Leakage protection** (the most important part):
1. Features use receive-time windows only. `test_no_leakage.py` deletes all data after t and checks the features don't change.
2. Labels use only the future horizon.
3. Splits are by **schedule seed**: the QoS 0 and QoS 1 runs of the same seed share a schedule, so they never land on opposite sides of a split.
4. Decision thresholds are chosen on training data only.
5. The impairment schedule is never a feature.

**Models, simplest first:** persistence ("next 30 s = last 5 s"), a threshold
rule, logistic regression, random forest, LightGBM.

**Metrics:** PR-AUC (main: works when bad windows are rare), ROC-AUC,
precision/recall/F1, **early-warning rate** (share of degradation onsets that
got an alarm before they began), median lead time, false alarms per device-hour.

**Run:** `python -m iotmon.ml.train` → `results/ml_report.md`, `results/ml_results.json`, `models/degradation_model.joblib`

The model with the best cross-validated PR-AUC is refitted on all mixed runs and saved. In the results here that
was the **random forest** (PR-AUC 0.363 vs 0.245 for persistence; prevalence 0.107). Its alert threshold (0.67)
was chosen on out-of-fold scores.

**What the results say (emulated):** the models beat the baselines, but only modestly. On unseen bursty loss,
ranking (PR-AUC) still beats persistence, but the threshold doesn't transfer well, so recall drops. UDP probe
features added nothing over application-level features.

**Gradual vs sudden onsets.** `train.py` also tags every degradation onset with what the fault injector was doing
when it began, and reports the early-warning rate per kind. The finding is uncomfortable but important: the
deployed random forest warned **none** of the 88 gradual (ramp) onsets in cross-validation, even though delay
visibly rises before them (for example 660 → 950 ms over 15 s). The simple threshold rule caught 28%. So the
model mostly *reacts*; it does not yet *anticipate*. Its higher warning rates on "sudden" onsets come mainly from
alarms still active after an earlier fault, not from prediction. Adding trend-projection features (delay + 6 ×
slope) was tried and did not help. Treat "predict 30 s ahead" as an open problem in the report, not a result.

**How to read the results honestly.** If a baseline does about as well as the
model, report that. It's a finding, not a failure. The results here come from the
userspace proxy, so they are a pipeline check, not paper results. Rerun with netem
(and later real boards) before claiming anything.

---

## Block 9b: Loss-pattern analysis

**Files:** `src/iotmon/ml/patterns.py`, gap features in `src/iotmon/windows.py`
**Tests:** `tests/test_patterns.py`  **Status:** Implemented; results in `results/pattern_report.md` (**Emulated**)

**Task.** For every 5 s window and device: which pattern is the link showing *now*?
`clean`, `random_loss`, `bursty_loss`, `delay`, `stall` or `outage`. This is the "packet loss pattern analysis"
in the project title.

**Labels** come from the fault injector's ground truth: each window gets the phase that covers most of it
(ramp phases count as their kind). **Features** are the degradation features plus the *shape* of the sequence
gaps: number of gaps, mean and max gap length (= burst length), for MQTT messages and for UDP probes, with
30 s and 60 s history. Bursty loss needs the 60 s history: a Gilbert–Elliott link is clean most of the time.

**Why gap shape works.** The same 3 lost messages look different: random loss gives 3 gaps of length 1,
bursty loss gives 1 gap of length 3 (`test_gap_features_separate_random_from_bursty`).

**Evaluation:** GroupKFold by schedule seed over all 24 runs; baseline = a hand-written rule classifier;
every model with and without UDP probe features; macro F1 (all classes count equally) and a confusion matrix.

**Run:** `python -m iotmon.ml.patterns` → `results/pattern_report.md`, `results/pattern_results.json`,
`models/pattern_model.joblib`

**Results (emulated):** random forest macro F1 **0.83** (MQTT + UDP probes) and **0.79** (MQTT only), against
0.55 and 0.37 for the rule baselines and 0.12 for always guessing `clean`. Outage, delay and random loss are
recognised in over 90% of windows. Bursty loss (53%) and stall (59%) are the weak classes: bursty windows are
often loss-free, and stalls are short (3–10 s), so many stall windows are only partly stalled.

---

## Block 10: Alerts and live prediction

**Files:** `src/iotmon/alerts.py`, `src/iotmon/live.py`  **Tests:** `tests/test_alerts.py`

- **Rule alert:** receive ratio < 0.9 or delay p95 > 1 s.
- **Model alert:** risk score ≥ the model's threshold.
- **Hysteresis:** raise after 2 bad windows in a row, clear after 2 good ones.
  The model only "clears" below half the threshold, so alerts don't flicker.

`python -m iotmon.live` scores every active device every 5 s and writes
`predictions`, `alerts` and `patterns` (the loss pattern of each window). It prints a line whenever an alert
is raised or cleared, or a device's pattern changes. `python -m iotmon.live --replay <run>` does the same
for a finished run, so the dashboard can show it.

Live mode builds windows up to the current time, so a full outage (no messages at all) shows up as empty
windows while it is happening, and it scores every window it has not scored yet. The last few seconds of a
finished run (devices shutting down) are ignored.

---

## Block 11: Dashboard

**File:** `dashboard/app.py`  **Run:** `streamlit run dashboard/app.py`

| Tab | Shows |
|---|---|
| Fleet | a tile per device (health score, receive ratio, delay), fleet health over time |
| Device | receive ratio, delay p95, RTT p95 and jitter over time; injected phases shaded |
| Predictions & alerts | model risk score above the truth on-time ratio; the alert log |
| Loss patterns | the classifier's pattern per window next to the injected pattern (truth); time spent in each pattern |
| Experiments | per-phase validation table, the ML report, the loss-pattern report |

**Health score** = 100 × (25% delivery + 25% delay + 25% jitter + 25% connected).
It's a summary for people to read, not a research result.

---

## Where each research question is answered

| Question | Where |
|---|---|
| RQ1: how network loss/delay shows up in MQTT signals, QoS 0 vs 1 | `python -m iotmon.validate`, Block 8 |
| RQ2: can a model forecast degradation and beat baselines | `results/ml_report.md`, cross-validation table |
| RQ3: does it work on unseen conditions | `results/ml_report.md`, held-out burst table |
| RQ4: can the loss pattern be recognised from MQTT-level signals | `results/pattern_report.md`, Block 9b |

## Reusing the experiment data

`data/iot_monitor.dump` is the complete database behind the results (24 runs plus the pilot): the 20 runs of
`experiments/plan.yaml` and the 4 extra bursty runs of `experiments/plan_extra_burst.yaml`. Restore it with
`pg_restore` (command in the README) to retrain or explore the dashboard without rerunning the experiments.

## What to do next
1. Run everything once on your laptop (README quick start).
2. Run the Wokwi firmware against your local broker.
3. Set up WSL2 + netem and rerun `experiments/plan.yaml` with `--backend netem`. Those are the results worth writing up.
4. Use the literature review (related work in four areas, with a draft Related Work section) in the report.
5. More burst-profile runs would strengthen the weakest pattern class (bursty loss).
6. If you get ESP32 boards: run the firmware on real Wi-Fi as a validation (**Measured**).
