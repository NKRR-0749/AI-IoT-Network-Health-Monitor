# AI-IoT-Network-Health-Monitor

**AI-Driven IoT-Based Real-Time Packet Loss Pattern Analysis and Predictive Network Health Monitoring System**

B.Tech AIE (Artificial Intelligence Engineering) project. ESP32 / emulated IoT devices publish numbered telemetry over MQTT. A Python
collector stores everything in PostgreSQL, measures delivery, delay, jitter, RTT and network-layer loss,
a machine learning model tries to predict degradation 30 s ahead, and a second model names the
**loss pattern** of every device every 5 s (clean, random loss, bursty loss, delay, stall, outage).
A Streamlit dashboard shows device health, patterns and alerts.

> **Research question.** How early and how accurately can application-level MQTT signals forecast network
> degradation that the application cannot see directly?

**New here? Read [`docs/GUIDE.md`](docs/GUIDE.md).** It explains every block: what it does, why, how to run
it, how to test it, and the common mistakes.

## Status of each part

| Part | Status | Label |
|---|---|---|
| PostgreSQL schema, collector, gap detection | Implemented, tested | — |
| Metrics (PDR, on-time ratio, RFC 3550 jitter, RTT, UDP loss) | Implemented, unit-tested | — |
| Python device emulator (N devices, QoS 0/1) | Implemented | Emulated |
| Userspace impairment proxy (delay, jitter, TCP-loss model, outages, bursts) | Implemented | Emulated (userspace model) |
| netem backend (real IP-level loss, Linux/WSL2) | Written, **not yet run** | Emulated (kernel) |
| Experiments: 20 runs + 4 extra bursty runs + pilot | Run | Emulated (userspace proxy) |
| ML: baselines, logistic regression, random forest (deployed), LightGBM | Implemented, evaluated | Emulated |
| Loss-pattern classifier (6 patterns, gap-shape features) | Implemented, evaluated, tested | Emulated |
| Alerts with hysteresis, live predictor (detects full outages while they happen) | Implemented, tested end to end | — |
| Streamlit dashboard (incl. Loss patterns tab) | Implemented, checked headless | — |
| ESP32 firmware for Wokwi | Written, **not compiled in the build environment** | Simulated |
| Physical ESP32 on real Wi-Fi | Planned | Measured |

## Architecture

```
 Wokwi ESP32 ─────────────────────────────────────┐
 Python emulator ─► fault injector (proxy/netem) ─►│ Mosquitto ─► collector ─► PostgreSQL ─► windows ─► ML model
 Physical ESP32 (planned) ────────────────────────┘                                │              │
                     fault injector ─► impairments table (ground truth) ───────────┘   live predictor + alerts ─► dashboard
```

Three layers are measured separately, because MQTT runs on TCP and TCP hides most IP packet loss:

| Layer | Signal | Name used |
|---|---|---|
| Application | sequence-number gaps | message loss / delivery ratio |
| Application | ping → echo → pong | application RTT |
| Network | UDP probes (no retransmission) | network-layer packet loss |

## Repository layout

```
db/schema.sql                 tables
src/iotmon/
  config.py                   settings (env vars / .env)
  db.py                       python -m iotmon.db init
  protocol.py                 message format
  gaps.py                     sequence-gap detection
  collector.py                MQTT + UDP collector  (the backend)
  emulator.py                 N virtual devices
  metrics.py                  formulas
  windows.py                  observed / truth windows
  impair/                     loss models, proxy, netem, schedules
  experiment.py               experiment runner (single + batch)
  validate.py                 injected vs measured, per phase
  ml/dataset.py               features + labels (leakage-safe)
  ml/train.py                 degradation forecast: baselines + models + evaluation
  ml/patterns.py              loss-pattern classifier + evaluation
  alerts.py                   alert engine (hysteresis)
  live.py                     live predictor, loss patterns, alerts / replay
dashboard/app.py              Streamlit dashboard
firmware/esp32_telemetry/     ESP32 sketch + Wokwi files
experiments/                  plan.yaml, plan_extra_burst.yaml, schedules, mosquitto.conf, runs/
results/                      ML report, loss-pattern report, RQ1 summary, validation tables
data/iot_monitor.dump         the full experiment database (pg_restore)
models/                       trained models (degradation, loss pattern)
tests/                        pytest unit tests
docs/GUIDE.md                 block-by-block guide
```

## Quick start (Windows laptop)

**1. Install** (all free):
- Python 3.11+ from python.org (tick "Add python.exe to PATH")
- PostgreSQL 16 from postgresql.org (remember the password you set for `postgres`)
- Mosquitto from mosquitto.org/download
- Git

Or, if you have Docker Desktop, run `docker compose up -d` to get PostgreSQL and Mosquitto, and skip their installers.

**2. Create the database user** (skip with Docker). Open "SQL Shell (psql)":
```sql
CREATE USER iot WITH PASSWORD 'iot' CREATEDB;
CREATE DATABASE iot_monitor OWNER iot;
```

**3. Set up Python** (in the project folder):
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
copy .env.example .env
python -m iotmon.db init
pytest
```
All tests should pass.

**4. Run it** (one terminal each, with `.venv` activated):
```bat
:: terminal 1: broker (skip if Mosquitto runs as a Windows service or in Docker)
"C:\Program Files\mosquitto\mosquitto.exe" -c experiments\mosquitto.conf -v
:: terminal 2: collector
python -m iotmon.collector
:: terminal 3: three emulated devices
python -m iotmon.emulator --run-id demo --devices 3
:: terminal 4: dashboard (opens in the browser)
streamlit run dashboard/app.py
```

**5. Run an experiment** (collector still running):
```bat
python -m iotmon.experiment run --run-id pilot --schedule experiments/schedules/pilot.yaml --devices 5
python -m iotmon.validate --run-id pilot
python -m iotmon.live --replay pilot
```

**6. Full experiment set + ML** (about 40 minutes for the batch):
```bat
python -m iotmon.experiment batch --plan experiments/plan.yaml --parallel 4
python -m iotmon.experiment batch --plan experiments/plan_extra_burst.yaml --parallel 4
python -m iotmon.ml.train
python -m iotmon.ml.patterns
```

## Running with netem (real IP-level impairment, recommended for paper results)

netem is part of the Linux kernel. On Windows, use **WSL2** (Ubuntu):
1. `wsl --install -d Ubuntu`, then inside Ubuntu: `sudo apt install python3-venv postgresql mosquitto iproute2`
2. Set up the project inside WSL as in step 3, and start Mosquitto with `experiments/mosquitto.conf` (listeners 1883 and 1884).
3. Check netem works: `sudo tc qdisc add dev lo root netem delay 1ms && sudo tc qdisc del dev lo root`
4. Run a pilot: `sudo -E .venv/bin/python -m iotmon.experiment run --run-id netem-pilot --backend netem --schedule experiments/schedules/pilot.yaml --devices 5`
5. `python -m iotmon.validate --run-id netem-pilot`: UDP loss should match the injected loss.
6. Full set (one run at a time): `sudo -E .venv/bin/python -m iotmon.experiment batch --plan experiments/plan.yaml --parallel 1 --backend netem`

The netem backend was written to netem's documented syntax but could not be run in the build
environment. Test it with the pilot first.

## Results so far (emulated, userspace proxy)

All numbers below come from 24 runs (10 devices, 0.5 s interval, about 7.5 min each, about 215,000 messages)
using the **userspace proxy**: the 20 runs of `plan.yaml` (run on Windows) and 4 extra bursty-loss runs of
`plan_extra_burst.yaml` (same code, run on Linux). They show the pipeline works. They are **not** paper results yet.
Full tables: [`results/ml_report.md`](results/ml_report.md), [`results/pattern_report.md`](results/pattern_report.md),
[`results/rq1_summary.md`](results/rq1_summary.md), `results/validation_*.md`.

**RQ1: how impairments show up in MQTT** (16 mixed runs; mean per phase kind)

| Injected | UDP probe loss | MQTT message loss, QoS 0 | MQTT message loss, QoS 1 | Median RTT |
|---|---|---|---|---|
| random loss (1–30%) | ~17% | 0.9% | 0% | 116–133 ms (clean: ~30 ms) |
| connection drop (5–30 s) | 100% | 100% | 0% (delivered late) | — |
| 3–10 s stall | 100% | 22.7% | 0% | — |
| delay 100–1200 ms | 0% | 0% | 0% | ~600 ms |
| bursty loss (held-out runs) | ~16% | 16.8% | 0% | 24–33 ms |

TCP hides random loss as delay. Outages and long loss bursts break the connection, and then QoS decides
whether messages are lost (QoS 0) or only late (QoS 1).

**RQ2: predicting degradation in the next 30 s** (4-fold cross-validation grouped by schedule seed; 13,965
rows; 10.7% positive, so random guessing gives PR-AUC ≈ 0.11)

| Model | PR-AUC | F1 | Early-warning rate |
|---|---|---|---|
| persistence baseline | 0.245 | 0.279 | 0.20 |
| threshold rule | 0.161 | 0.287 | 0.36 |
| logistic regression | 0.257 | 0.316 | 0.46 |
| **random forest** (deployed) | **0.363** | 0.311 | 0.33 |
| LightGBM | 0.278 | 0.286 | 0.43 |

**RQ3: unseen bursty-loss conditions** (8 held-out runs): every ML model beats persistence on PR-AUC
(0.51–0.52 vs 0.45), but the decision thresholds transfer badly (random forest recall drops to 0.12).

**Gradual vs sudden onsets** (cross-validation, share of onsets with an alarm in the 30 s before):

| Onset kind | onsets | threshold rule | logistic regression | random forest (deployed) |
|---|---|---|---|---|
| gradual (ramp) | 88 | 0.28 | 0.14 | **0.00** |
| sudden loss | 57 | 0.28 | 0.28 | 0.16 |
| sudden stall | 120 | 0.38 | 0.73 | 0.58 |
| sudden outage | 100 | 0.43 | 0.54 | 0.41 |

The deployed model did not warn about a single gradual degradation, although delay visibly rises before
them. Its warnings before "sudden" onsets come mostly from alarms still active after an earlier fault.

**RQ4: recognising the loss pattern** (5-fold cross-validation grouped by schedule seed; 21,870 windows;
macro F1 counts all six patterns equally)

| Model | Macro F1 | clean | random loss | bursty loss | delay | stall | outage |
|---|---|---|---|---|---|---|---|
| always "clean" | 0.12 | 1.00 | 0 | 0 | 0 | 0 | 0 |
| rule baseline, MQTT only | 0.37 | 0.93 | 0.00 | 0.01 | 0.78 | 0.22 | 0.87 |
| rule baseline, with UDP probes | 0.55 | 0.92 | 0.68 | 0.06 | 0.75 | 0.22 | 0.87 |
| **random forest, MQTT only** | **0.79** | 0.93 | 0.88 | 0.41 | 0.91 | 0.53 | 0.93 |
| **random forest, with UDP probes** | **0.83** | 0.94 | 0.91 | 0.53 | 0.93 | 0.59 | 0.95 |

Pattern columns are recall (share of windows of that pattern recognised). The key features are the *shape*
of the sequence gaps: the same 3 lost messages are 3 gaps of length 1 under random loss and 1 gap of length 3
under bursty loss. Bursty loss is the weakest class, because a Gilbert–Elliott link is loss-free most of the time.

**Honest reading.** Recognising *what is happening now* works well (RQ4). Forecasting *what happens next* is
only modestly better than baselines (RQ2), and the deployed model does not anticipate gradual degradation at
all, so it mostly reacts. UDP probe features added nothing to the forecast, but help the pattern classifier a
little (0.83 vs 0.79). Next steps: rerun with netem, then work on the forecast (more ramp data, a model
trained for onsets, tuning the horizon and thresholds).

**Reuse the data without rerunning the 40-minute batch.** The full database is in `data/iot_monitor.dump`:
```bat
pg_restore -d postgresql://iot:iot@localhost:5432/iot_monitor --clean --if-exists data/iot_monitor.dump
python -m iotmon.ml.train
python -m iotmon.ml.patterns
```
With Docker, restore inside the container instead:
```bat
docker compose cp data/iot_monitor.dump postgres:/tmp/db.dump
docker compose exec postgres pg_restore -U iot -d iot_monitor --clean --if-exists /tmp/db.dump
```
The dump already holds predictions, alerts and loss patterns for `mix-s01-q0`, `mix-s08-q0`, `bur-s101-q0`,
`bur-s103-q0` and `bur-s104-q0`. For any other run: `python -m iotmon.live --replay <run>`.

## Honest limitations
- All experiment results so far come from the **userspace proxy**. Its TCP-loss behaviour is a model of
  TCP's reaction to loss, not real IP loss. Rerun with netem before reporting results.
- Devices are emulated in Python, not ESP32s: memory, timing and the TCP stack differ.
- Emulated loss models (Bernoulli, Gilbert–Elliott) are not real Wi-Fi interference.
- RSSI is synthetic in the emulator and in Wokwi, so it is stored but never used as a feature.
- One broker on one machine.
- The loss-pattern labels come from the injected schedule. A window inside a bursty phase that happens to
  have no loss is still labelled "bursty", which caps the measurable recall for that class.
- The 4 extra bursty runs were recorded on a different machine (Linux) from the first 20 (Windows).
- A literature review exists (four areas of related work), but no novelty is claimed until results are rerun with netem.

## Licence
MIT, see [`LICENSE`](LICENSE).
