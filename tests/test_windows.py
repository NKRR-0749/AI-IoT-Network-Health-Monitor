"""Windows are checked on a hand-made stream where we know exactly what happened."""
import numpy as np
import pandas as pd
import pytest

from iotmon.windows import observed_windows, truth_windows

T0 = 1_000_000.0   # device boot, in collector epoch seconds (multiple of 5 for an easy grid)


def make_data(lost=(), late=None, n=40, interval=0.5, delay=0.05):
    """n messages every `interval` s, each arriving `delay` s later; `lost` seqs never arrive,
    `late` = {seq: extra_seconds} arrive later than usual."""
    late = late or {}
    rows = []
    for s in range(n):
        if s in lost:
            continue
        send = T0 + s * interval
        rows.append({"device_id": "d1", "boot_id": 1, "seq": s, "interval_ms": int(interval * 1000),
                     "uptime_ms": int(s * interval * 1000), "ts": send + delay + late.get(s, 0.0)})
    m = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    m["id"] = range(len(m))
    empty = lambda cols: pd.DataFrame(columns=cols)
    return {"messages": m,
            "rtt": empty(["device_id", "probe_seq", "rtt_ms", "ts"]),
            "udp": empty(["device_id", "boot_id", "probe_seq", "uptime_ms", "ts", "id"]),
            "events": empty(["device_id", "event", "ts"])}


def test_truth_counts_lost_messages_in_their_send_window():
    data = make_data(lost={12, 13, 14})          # sent at 6.0, 6.5, 7.0 s -> window 5-10 s
    t = truth_windows(data, window_s=5).set_index("window_start")
    w = t.loc[T0 + 5]
    assert w["expected"] == 10 and w["received"] == 7
    assert w["pdr"] == pytest.approx(0.7)
    assert t.loc[T0]["pdr"] == 1.0


def test_truth_late_message_is_received_but_not_on_time():
    data = make_data(late={4: 3.0})               # seq 4 sent at 2.0 s, arrives 3 s late
    t = truth_windows(data, window_s=5, ontime_ms=1000).set_index("window_start")
    w = t.loc[T0]
    assert w["pdr"] == 1.0 and w["otdr"] == pytest.approx(0.9)


def test_observed_reveals_gap_when_next_message_arrives():
    data = make_data(lost={12, 13, 14})
    o = observed_windows(data, window_s=5).set_index("window_start")
    # gap is revealed when seq 15 arrives (t = 7.55 s), i.e. in the 5-10 s receive window
    assert o.loc[T0 + 5]["missing_revealed"] == 3
    assert o.loc[T0 + 5]["n_received"] == 7
    assert o.loc[T0]["missing_revealed"] == 0


def test_observed_constant_delay_has_zero_relative_delay_and_jitter():
    o = observed_windows(make_data(), window_s=5)
    assert np.allclose(o["delay_mean_ms"].dropna(), 0.0, atol=1e-6)
    assert np.allclose(o["jitter_ms"].dropna(), 0.0, atol=1e-6)
    assert o["recv_ratio"].iloc[0] == pytest.approx(1.0)
