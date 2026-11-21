-- =====================================================================
-- AI-IoT-Network-Health-Monitor : PostgreSQL schema
-- Raw data is written once and never edited. Everything else
-- (windows, predictions, alerts) is computed from raw data, so any
-- result can be rebuilt.
-- All timestamps are timestamptz (stored in UTC).
-- =====================================================================

-- One row per experiment run (or 'live' for the Wokwi / real device feed).
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    config      JSONB NOT NULL DEFAULT '{}'::jsonb,   -- factors: qos, interval, profile ...
    seed        INTEGER,
    git_commit  TEXT,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at    TIMESTAMPTZ
);

-- One row per device. kind = wokwi | emulated | physical
CREATE TABLE IF NOT EXISTS devices (
    device_id        TEXT PRIMARY KEY,
    kind             TEXT NOT NULL DEFAULT 'emulated',
    firmware_version TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Raw telemetry. One row per message the collector received.
-- (device_id, run_id, boot_id, seq) is unique: a repeat is a duplicate.
CREATE TABLE IF NOT EXISTS messages (
    id               BIGSERIAL PRIMARY KEY,
    run_id           TEXT NOT NULL REFERENCES runs(run_id),
    device_id        TEXT NOT NULL REFERENCES devices(device_id),
    boot_id          INTEGER NOT NULL,
    seq              BIGINT NOT NULL,
    qos              SMALLINT NOT NULL,
    interval_ms      INTEGER,
    device_uptime_ms BIGINT NOT NULL,
    rssi             SMALLINT,                       -- recorded only; synthetic in simulation
    received_at      TIMESTAMPTZ NOT NULL,
    payload_bytes    INTEGER,
    UNIQUE (run_id, device_id, boot_id, seq)
);
CREATE INDEX IF NOT EXISTS messages_dev_time ON messages (run_id, device_id, received_at);

-- Application-level RTT samples (device -> collector echo -> device).
-- rtt_ms is NULL when the echo did not come back before the timeout.
CREATE TABLE IF NOT EXISTS rtt_probes (
    id          BIGSERIAL PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    device_id   TEXT NOT NULL REFERENCES devices(device_id),
    probe_seq   BIGINT NOT NULL,
    rtt_ms      DOUBLE PRECISION,
    measured_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS rtt_dev_time ON rtt_probes (run_id, device_id, measured_at);

-- UDP probe packets that ARRIVED. Lost probes are the gaps in probe_seq.
-- This is the network-layer loss signal (UDP has no retransmission).
CREATE TABLE IF NOT EXISTS udp_probes (
    id          BIGSERIAL PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    device_id   TEXT NOT NULL REFERENCES devices(device_id),
    boot_id     INTEGER NOT NULL,
    probe_seq   BIGINT NOT NULL,
    device_uptime_ms BIGINT NOT NULL,
    received_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS udp_dev_time ON udp_probes (run_id, device_id, received_at);

-- connect / disconnect events (from the retained status topic + Last Will).
CREATE TABLE IF NOT EXISTS connection_events (
    id        BIGSERIAL PRIMARY KEY,
    run_id    TEXT NOT NULL REFERENCES runs(run_id),
    device_id TEXT NOT NULL REFERENCES devices(device_id),
    event     TEXT NOT NULL CHECK (event IN ('online', 'offline')),
    at        TIMESTAMPTZ NOT NULL
);

-- GROUND TRUTH: what the fault injector actually did. Never used as a feature.
CREATE TABLE IF NOT EXISTS impairments (
    id        BIGSERIAL PRIMARY KEY,
    run_id    TEXT NOT NULL REFERENCES runs(run_id),
    phase     TEXT NOT NULL,          -- e.g. clean, random_loss, burst_loss, delay, outage_drop
    start_at  TIMESTAMPTZ NOT NULL,
    end_at    TIMESTAMPTZ,
    params    JSONB NOT NULL          -- loss_pct, burst_len, delay_ms, jitter_ms, outage ...
);

-- Aggregated metrics per (run, device, window).
--   kind = 'observed' : what the collector had SEEN by the window end (receive-time based, leakage-safe features)
--   kind = 'truth'    : what happened to messages SENT in the window (send-time based, uses hindsight; labels only)
CREATE TABLE IF NOT EXISTS windows (
    run_id       TEXT NOT NULL,
    device_id    TEXT NOT NULL,
    kind         TEXT NOT NULL CHECK (kind IN ('observed', 'truth')),
    window_start TIMESTAMPTZ NOT NULL,
    window_s     INTEGER NOT NULL,
    metrics      JSONB NOT NULL,
    PRIMARY KEY (run_id, device_id, kind, window_start, window_s)
);

-- Model outputs, kept so they can be evaluated later.
CREATE TABLE IF NOT EXISTS predictions (
    id            BIGSERIAL PRIMARY KEY,
    run_id        TEXT NOT NULL,
    device_id     TEXT NOT NULL,
    predicted_at  TIMESTAMPTZ NOT NULL,
    horizon_s     INTEGER NOT NULL,
    model_name    TEXT NOT NULL,
    model_version TEXT NOT NULL,
    score         DOUBLE PRECISION NOT NULL,   -- probability of degradation in the horizon
    label         SMALLINT NOT NULL            -- 1 = predicted degraded
);
CREATE INDEX IF NOT EXISTS pred_dev_time ON predictions (run_id, device_id, predicted_at);

-- Alerts and their lifecycle (raised -> cleared).
CREATE TABLE IF NOT EXISTS alerts (
    id         BIGSERIAL PRIMARY KEY,
    run_id     TEXT NOT NULL,
    device_id  TEXT NOT NULL,
    raised_at  TIMESTAMPTZ NOT NULL,
    cleared_at TIMESTAMPTZ,
    severity   TEXT NOT NULL,
    reason     TEXT NOT NULL,
    source     TEXT NOT NULL CHECK (source IN ('rule', 'model'))
);

-- Loss pattern of each device per 5 s window (ml/patterns.py classifier, written by live.py)
CREATE TABLE IF NOT EXISTS patterns (
    id         BIGSERIAL PRIMARY KEY,
    run_id     TEXT NOT NULL,
    device_id  TEXT NOT NULL,
    at         TIMESTAMPTZ NOT NULL,                -- end of the window
    pattern    TEXT NOT NULL,                       -- clean / random_loss / bursty_loss / delay / stall / outage
    confidence DOUBLE PRECISION NOT NULL,           -- classifier probability of that pattern
    model      TEXT NOT NULL                        -- 'with_probe' or 'mqtt_only'
);
CREATE INDEX IF NOT EXISTS patterns_dev_time ON patterns (run_id, device_id, at);
