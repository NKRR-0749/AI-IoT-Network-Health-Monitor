"""Loss-pattern block: gap-shape features, ground-truth labels, long history."""
import numpy as np
import pandas as pd

from iotmon.ml.patterns import add_long_history, label_windows, phase_pattern
from iotmon.windows import observed_windows
from test_windows import T0, make_data


def test_gap_features_separate_random_from_bursty():
    # window 0-5 s: messages 0-9. Random: 3 single losses. Bursty: one run of 3 losses.
    random_ = observed_windows(make_data(lost={2, 5, 8}, n=20), 5).set_index("window_start").loc[T0]
    bursty = observed_windows(make_data(lost={4, 5, 6}, n=20), 5).set_index("window_start").loc[T0]
    assert random_["gap_events"] == 3 and random_["gap_len_max"] == 1 and random_["gap_len_mean"] == 1
    assert bursty["gap_events"] == 1 and bursty["gap_len_max"] == 3 and bursty["gap_len_mean"] == 3
    assert random_["missing_revealed"] == bursty["missing_revealed"] == 3      # same loss, different shape


def test_no_loss_means_zero_gap_features():
    w = observed_windows(make_data(n=20), 5)
    assert (w["gap_events"] == 0).all() and (w["gap_len_max"] == 0).all()


def test_phase_pattern_mapping():
    assert phase_pattern("random_loss", {"loss_pct": 10}) == "random_loss"
    assert phase_pattern("burst_loss", {"loss_pct": 10, "burst_len": 5}) == "bursty_loss"
    assert phase_pattern("ramp_loss", {"loss_pct": 5.0, "burst_len": 10}) == "bursty_loss"
    assert phase_pattern("ramp_delay", {"delay_ms": 133, "jitter_ms": 33}) == "delay"
    assert phase_pattern("outage_stall", {"outage": "stall"}) == "stall"
    assert phase_pattern("outage_drop", {"outage": "drop"}) == "outage"
    assert phase_pattern("clean", {}) == "clean"


def test_window_gets_the_phase_that_covers_most_of_it():
    ph = pd.DataFrame({"start": [0.0, 7.0], "end": [7.0, 20.0], "pattern": ["clean", "delay"],
                       "intensity": [0.0, 300.0]})
    lab = label_windows(np.array([0.0, 5.0, 10.0, 25.0]), ph, 5).set_index("window_start")
    assert lab.loc[0.0, "pattern"] == "clean" and lab.loc[0.0, "pure"]
    assert lab.loc[5.0, "pattern"] == "delay" and not lab.loc[5.0, "pure"]    # 3 s of delay beats 2 s of clean
    assert lab.loc[10.0, "pattern"] == "delay"
    assert pd.isna(lab.loc[25.0, "pattern"])                                     # outside the schedule


def test_long_history_only_looks_back():
    f = pd.DataFrame({"device_id": "d1", "window_start": np.arange(20) * 5.0, "gap_events_now": 0.0,
                      "gap_len_max_now": 0.0, "loss_revealed_now": 0.0, "udp_gap_events_now": 0.0,
                      "udp_gap_len_max_now": 0.0, "udp_loss_revealed_now": 0.0})
    f.loc[10, "udp_gap_len_max_now"] = 8.0
    h = add_long_history(f).set_index("window_start")
    assert h.loc[45.0, "udp_gap_len_max_max12"] == 0        # before the burst: nothing from the future
    assert h.loc[50.0, "udp_gap_len_max_max12"] == 8        # the burst itself
    assert h.loc[95.0, "udp_gap_len_max_max12"] == 8        # still remembered 45 s later (within 60 s)
