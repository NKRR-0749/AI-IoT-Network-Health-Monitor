"""Block: Streamlit dashboard.

    streamlit run dashboard/app.py

Tabs
  Fleet        one tile per device: health score, delivery, delay, alerts
  Device       recv ratio, delay p95, RTT p95, jitter over time (+ injected phases, experiments only)
  Predictions  the model's risk score next to what actually happened (truth), plus the alert log
  Experiments  testbed validation for the selected run and the ML report
"""
import math
import os
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iotmon import db  # noqa: E402
from iotmon.config import ROOT, settings  # noqa: E402
from iotmon.metrics import health_score  # noqa: E402
from iotmon.validate import phase_table  # noqa: E402
from iotmon.windows import _query, load_run, observed_windows  # noqa: E402

st.set_page_config(page_title="IoT Network Health Monitor", layout="wide")
W = settings.window_s
BLUE, RED, INK2 = "#2a78d6", "#e34948", "#52514e"
PHASE_FILL = "rgba(127,127,127,0.13)"


# ------------------------------------------------------------------ data
@st.cache_resource
def conn():
    return db.connect()


@st.cache_data(ttl=10)
def runs() -> pd.DataFrame:
    return _query(conn(), "SELECT run_id, config, started_at, ended_at FROM runs ORDER BY started_at DESC", [])


@st.cache_data(ttl=4)
def observed(run_id: str, live: bool) -> pd.DataFrame:
    data = load_run(conn(), run_id, since_epoch=time.time() - 900 if live else None)
    o = observed_windows(data, W)
    if not o.empty:
        o["time"] = pd.to_datetime(o["window_start"] + W, unit="s", utc=True)
        o["health"] = [health_score(r.recv_ratio, r.delay_p95_ms, r.jitter_ms, r.connected > 0 and r.n_received > 0)
                       for r in o.itertuples()]
    return o


@st.cache_data(ttl=10)
def truth(run_id: str) -> pd.DataFrame:
    t = _query(conn(), "SELECT device_id, extract(epoch from window_start) AS ts, (metrics->>'otdr')::float AS otdr,"
                       " (metrics->>'pdr')::float AS pdr FROM windows WHERE run_id=%s AND kind='truth'", [run_id])
    if not t.empty:
        t["time"] = pd.to_datetime(t["ts"].astype(float) + W / 2, unit="s", utc=True)
    return t


@st.cache_data(ttl=4)
def phases(run_id: str) -> pd.DataFrame:
    return _query(conn(), "SELECT phase, start_at, coalesce(end_at, now()) AS end_at, params FROM impairments"
                          " WHERE run_id=%s ORDER BY start_at", [run_id])


@st.cache_data(ttl=4)
def predictions(run_id: str) -> pd.DataFrame:
    return _query(conn(), "SELECT device_id, predicted_at, score, label FROM predictions WHERE run_id=%s"
                          " ORDER BY predicted_at", [run_id])


@st.cache_data(ttl=4)
def alerts(run_id: str) -> pd.DataFrame:
    return _query(conn(), "SELECT device_id, source, severity, raised_at, cleared_at, reason FROM alerts"
                          " WHERE run_id=%s ORDER BY raised_at DESC", [run_id])


# ------------------------------------------------------------------ charts
def shade_phases(fig, ph: pd.DataFrame, rows: int):
    for _, p in ph.iterrows():
        if p["phase"] == "clean":
            continue
        for r in range(1, rows + 1):
            fig.add_vrect(x0=p["start_at"], x1=p["end_at"], fillcolor=PHASE_FILL, line_width=0, row=r, col=1)
        fig.add_annotation(x=p["start_at"], y=1.0, yref="paper", text=p["phase"].replace("_", " "),
                           showarrow=False, xanchor="left", yanchor="bottom", font=dict(size=10, color=INK2))


def style(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                      hovermode="x unified", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(127,127,127,0.18)", zeroline=False)
    return fig


def device_chart(d: pd.DataFrame, ph: pd.DataFrame):
    panels = [("recv_ratio", "Messages received / expected"), ("delay_p95_ms", "Relative delay p95 (ms)"),
              ("rtt_p95_ms", "Application RTT p95 (ms)"), ("jitter_ms", "Jitter, RFC 3550 (ms)")]
    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=[t for _, t in panels])
    for i, (c, t) in enumerate(panels, start=1):
        fig.add_trace(go.Scatter(x=d["time"], y=d[c], mode="lines", line=dict(color=BLUE, width=2), name=t,
                                 connectgaps=False), row=i, col=1)
    shade_phases(fig, ph, len(panels))
    return style(fig, 760)


def prediction_chart(p: pd.DataFrame, t: pd.DataFrame, thr: float | None, ph: pd.DataFrame):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=["Model: risk of degradation in the next 30 s",
                                        "What happened: on-time delivery ratio (truth, per window)"])
    fig.add_trace(go.Scatter(x=p["predicted_at"], y=p["score"], mode="lines", line=dict(color=BLUE, width=2),
                             name="risk"), row=1, col=1)
    if thr is not None:
        fig.add_hline(y=thr, line=dict(color=INK2, dash="dot", width=1), row=1, col=1,
                      annotation_text=f"alert threshold {thr:.2f}", annotation_font_size=10)
    if not t.empty:
        fig.add_trace(go.Scatter(x=t["time"], y=t["otdr"], mode="lines", line=dict(color=BLUE, width=2),
                                 name="on-time ratio"), row=2, col=1)
        fig.add_hline(y=0.9, line=dict(color=RED, dash="dot", width=1), row=2, col=1,
                      annotation_text="degraded below 0.9", annotation_font_size=10)
    fig.update_yaxes(range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(range=[0, 1.05], row=2, col=1)
    shade_phases(fig, ph, 2)
    return style(fig, 520)


# ------------------------------------------------------------------ page
st.title("IoT Network Health Monitor")
r = runs()
if r.empty:
    st.info("No data yet. Start the collector and an emulator (see README).")
    st.stop()

with st.sidebar:
    run_id = st.selectbox("Run", r["run_id"].tolist())
    row = r[r["run_id"] == run_id].iloc[0]
    cfg = row["config"] or {}
    live = pd.isna(row["ended_at"])
    st.caption(("LIVE run" if live else "Finished run") + (f" · QoS {cfg.get('qos')}" if "qos" in cfg else "") +
               (f" · profile {cfg.get('profile')}" if "profile" in cfg else ""))
    label = "EMULATED" if cfg.get("backend") in ("proxy", "netem") else "SIMULATED / MEASURED (see device kind)"
    st.caption(f"Data label: {label}")
    auto = st.toggle("Auto-refresh every 5 s", value=live and not os.getenv("IOT_DASH_NO_AUTO"))
    if st.button("Refresh now"):
        st.cache_data.clear()

obs = observed(run_id, live)
if obs.empty:
    st.warning("This run has no messages yet.")
    st.stop()
ph = phases(run_id)
devs = sorted(obs["device_id"].unique())
tab_fleet, tab_dev, tab_pred, tab_exp = st.tabs(["Fleet", "Device", "Predictions & alerts", "Experiments"])

with tab_fleet:
    last_complete = obs[obs["window_start"] + W <= (time.time() - 1 if live else obs["window_start"].max() - W)]
    latest = last_complete.sort_values("window_start").groupby("device_id").tail(1).set_index("device_id")
    al = alerts(run_id)
    open_al = al[al["cleared_at"].isna()] if not al.empty else al
    cols = st.columns(5)
    for i, dev in enumerate(devs):
        if dev not in latest.index:
            continue
        x = latest.loc[dev]
        n_open = int((open_al["device_id"] == dev).sum()) if not open_al.empty else 0
        status = "ALERT" if n_open else ("offline" if x["connected"] == 0 or x["n_received"] == 0 else "ok")
        with cols[i % 5]:
            st.metric(f"{dev} · {status}", f"{x['health']:.0f} / 100",
                      help="Health score: 25% delivery, 25% delay, 25% jitter, 25% connected")
            dp = x["delay_p95_ms"]
            st.caption(f"recv {x['recv_ratio']:.2f} · delay p95 {'–' if math.isnan(dp) else f'{dp:.0f} ms'}")
    st.subheader("Fleet health over time")
    fleet = obs.groupby("time")["health"].agg(["mean", "min"]).reset_index()
    fig = go.Figure([go.Scatter(x=fleet["time"], y=fleet["mean"], mode="lines", line=dict(color=BLUE, width=2),
                                name="mean health"),
                     go.Scatter(x=fleet["time"], y=fleet["min"], mode="lines", line=dict(color=INK2, width=1, dash="dot"),
                                name="worst device")])
    shade_phases(fig, ph, 1)
    fig = style(fig, 300)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")

with tab_dev:
    dev = st.selectbox("Device", devs)
    d = obs[obs["device_id"] == dev]
    if not ph.empty:
        st.caption("Shaded bands = impairment phases injected by the experiment runner (ground truth).")
    st.plotly_chart(device_chart(d, ph), width="stretch")
    with st.expander("Window table"):
        st.dataframe(d.drop(columns=["window_start"]).set_index("time"), width="stretch")

with tab_pred:
    dev2 = st.selectbox("Device ", devs, key="pdev")
    p = predictions(run_id)
    p = p[p["device_id"] == dev2] if not p.empty else p
    if p.empty:
        st.info("No predictions for this run. Live runs: start `python -m iotmon.live`. "
                "Finished runs: `python -m iotmon.live --replay " + run_id + "`.")
    else:
        try:
            import joblib
            thr = joblib.load(ROOT / "models" / "degradation_model.joblib")["threshold"]
        except Exception:
            thr = None
        t = truth(run_id)
        t = t[t["device_id"] == dev2] if not t.empty else t
        st.plotly_chart(prediction_chart(p, t, thr, ph), width="stretch")
    st.subheader("Alerts")
    al = alerts(run_id)
    st.dataframe(al, width="stretch", hide_index=True) if not al.empty else st.caption("No alerts.")

with tab_exp:
    if not ph.empty:
        st.subheader("Testbed validation: injected vs measured, per phase")
        st.dataframe(phase_table(conn(), run_id), width="stretch", hide_index=True)
    report = ROOT / "results" / "ml_report.md"
    if report.exists():
        st.markdown(report.read_text())
    else:
        st.caption("No ML report yet: run `python -m iotmon.ml.train`.")

if auto:
    time.sleep(5)
    st.cache_data.clear()
    st.rerun()
