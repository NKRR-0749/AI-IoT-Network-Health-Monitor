"""The most important ML test: features at time t must not change when we delete everything after t."""
import numpy as np
import pandas as pd

from iotmon.ml.dataset import make_features, make_labels
from iotmon.windows import observed_windows, truth_windows
from tests.test_windows import T0, make_data


def truncate(data, t_cut):
    return {k: (v[v["ts"] <= t_cut].copy() if len(v) else v) for k, v in data.items()}


def test_features_only_use_the_past():
    data = make_data(lost={30, 31, 32, 50, 51}, late={60: 4.0}, n=80)
    full = make_features(observed_windows(data, window_s=5), qos=0).set_index("window_start")
    for w in (T0 + 10, T0 + 20, T0 + 25):
        cut = make_features(observed_windows(truncate(data, w + 5), window_s=5), qos=0).set_index("window_start")
        a, b = full.loc[w].drop("device_id"), cut.loc[w].drop("device_id")
        assert np.allclose(a.astype(float), b.astype(float), equal_nan=True), f"leak at window {w}"


def test_label_looks_only_at_the_future_horizon():
    data = make_data(lost=set(range(20, 30)), n=80)      # seqs 20-29 sent at 10.0-14.5 s are lost
    lab = make_labels(truth_windows(data, window_s=5), window_s=5, horizon_s=10).set_index("window_start")
    # row T0+0 describes data up to 5 s; its horizon is the windows starting at 5 s and 10 s -> includes the loss
    assert lab.loc[T0, "target_pdr"] == 0.5
    # row T0+10 describes data up to 15 s; its horizon (15-25 s) is clean
    assert lab.loc[T0 + 10, "target_pdr"] == 1.0
