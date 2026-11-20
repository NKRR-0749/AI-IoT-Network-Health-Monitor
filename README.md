# AI-IoT-Network-Health-Monitor

**AI-Driven IoT-Based Real-Time Packet Loss Pattern Analysis and Predictive Network Health Monitoring System**

B.Tech CSE (AI) project. ESP32 / emulated IoT devices publish numbered telemetry over MQTT. A Python
collector stores everything in PostgreSQL, measures delivery, delay, jitter, RTT and network-layer loss,
and a machine learning model predicts degradation 30 s ahead. A Streamlit dashboard shows device health
and alerts.

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
| Experiments: 20 runs + pilot | Run | Emulated (userspace proxy) |
| ML: baselines, logistic regression, random forest (deployed), LightGBM | Implemented, evaluated | Emulated |
| Alerts with hysteresis, live predictor | Implemented, tested | — |
| Streamlit dashboard | Implemented | — |
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
  ml/train.py                 baselines + models + evaluation
  alerts.py                   alert engine (hysteresis)
  live.py                     live predictor / replay
dashboard/app.py              Streamlit dashboard
firmware/esp32_telemetry/     ESP32 sketch + Wokwi files
experiments/                  plan.yaml, schedules, mosquitto.conf, runs/ (schedule of every run)
results/                      ML report, RQ1 summary, validation tables
data/iot_monitor.dump         the full experiment database (pg_restore)
models/                       trained model
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
python -m iotmon.ml.train
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

All numbers below come from 20 runs (10 devices, 0.5 s interval, about 7.5 min each, about 180,000 messages)
using the **userspace proxy**. They show the pipeline works. They are **not** paper results yet.
Full tables: [`results/ml_report.md`](results/ml_report.md), [`results/rq1_summary.md`](results/rq1_summary.md),
`results/validation_*.md`.

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

**RQ3: unseen bursty-loss conditions:** every ML model beats persistence on PR-AUC (0.53–0.54 vs 0.48),
but the decision thresholds transfer badly (random forest recall drops to 0.11).

**Honest reading.** The models do better than the baselines, but only modestly. Most degradations here
begin suddenly (scripted outages), and nothing in the past predicts those. UDP probe features added
nothing over application-level features. Next steps: rerun with netem, add more gradual (ramp)
degradations, and tune the horizon and thresholds.

**Reuse the data without rerunning the 40-minute batch.** The full database is in `data/iot_monitor.dump`:
```bat
pg_restore -d postgresql://iot:iot@localhost:5432/iot_monitor --clean --if-exists data/iot_monitor.dump
python -m iotmon.ml.train
```

## Honest limitations
- All experiment results so far come from the **userspace proxy**. Its TCP-loss behaviour is a model of
  TCP's reaction to loss, not real IP loss. Rerun with netem before reporting results.
- Devices are emulated in Python, not ESP32s: memory, timing and the TCP stack differ.
- Emulated loss models (Bernoulli, Gilbert–Elliott) are not real Wi-Fi interference.
- RSSI is synthetic in the emulator and in Wokwi, so it is stored but never used as a feature.
- One broker on one machine.
- No literature review yet, so no novelty is claimed.

## Licence
Add a licence of your choice (for example MIT) before making the repository public.
