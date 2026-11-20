import math

import pytest

from iotmon.metrics import (Rfc3550Jitter, delivery_ratio, health_score, interpolate_missing_times,
                            percentile, std)


def test_delivery_ratio_hand_calculation():
    assert delivery_ratio(19, 20) == pytest.approx(0.95)
    assert delivery_ratio(0, 0) == 1.0
    assert delivery_ratio(25, 20) == 1.0          # never above 1


def test_percentile_matches_numpy_linear():
    vals = [10, 20, 30, 40]
    assert percentile(vals, 50) == pytest.approx(25.0)
    assert percentile(vals, 95) == pytest.approx(38.5)
    assert math.isnan(percentile([], 95))


def test_std_sample():
    assert std([2, 4, 4, 4, 5, 5, 7, 9]) == pytest.approx(2.138, abs=1e-3)


def test_jitter_constant_delay_is_zero():
    j = Rfc3550Jitter()
    for _ in range(10):
        j.update(50.0)
    assert j.j == 0.0


def test_jitter_hand_calculation():
    j = Rfc3550Jitter()
    j.update(50.0)                 # first sample: no D yet
    assert j.update(66.0) == pytest.approx(1.0)        # |D|=16 -> 0 + (16-0)/16
    assert j.update(50.0) == pytest.approx(1.9375)     # |D|=16 -> 1 + (16-1)/16


def test_interpolate_missing_times():
    assert interpolate_missing_times(10, 5.0, 13, 6.5) == [(11, 5.5), (12, 6.0)]
    assert interpolate_missing_times(10, 5.0, 11, 5.5) == []


def test_health_score_bounds():
    assert health_score(1.0, 0.0, 0.0, True) == 100.0
    assert health_score(0.0, 5000.0, 900.0, False) == 0.0
    assert health_score(0.9, 500.0, 100.0, True) == pytest.approx(100 * (0.25 * 0.9 + 0.25 * 0.5 + 0.25 * 0.5 + 0.25))
