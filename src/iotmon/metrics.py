"""Block: metric formulas (pure functions, no database).

Each function here matches a formula in the project plan, so it can be
checked by hand in tests/test_metrics.py.
"""
import math
from typing import Iterable, Sequence


def delivery_ratio(received_unique: int, expected: int) -> float:
    """PDR = unique messages received / messages expected. 1.0 when nothing was expected."""
    if expected <= 0:
        return 1.0
    return min(1.0, received_unique / expected)


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile (same as numpy's default). NaN for an empty list."""
    vals = sorted(v for v in values if v is not None and not math.isnan(v))
    if not vals:
        return math.nan
    k = (len(vals) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return vals[int(k)]
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def mean(values: Iterable[float]) -> float:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return sum(vals) / len(vals) if vals else math.nan


def std(values: Iterable[float]) -> float:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if len(vals) < 2:
        return math.nan
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


class Rfc3550Jitter:
    """Interarrival jitter, RFC 3550 section 6.4.1.

    transit_i = arrival_i - send_i   (clock offset cancels out in the difference)
    D = transit_i - transit_{i-1}
    J = J + (|D| - J) / 16
    """

    def __init__(self) -> None:
        self.j = 0.0
        self._prev_transit = None

    def update(self, transit_ms: float) -> float:
        if self._prev_transit is not None:
            d = abs(transit_ms - self._prev_transit)
            self.j += (d - self.j) / 16.0
        self._prev_transit = transit_ms
        return self.j


def interpolate_missing_times(seq_a: int, t_a: float, seq_b: int, t_b: float) -> list[tuple[int, float]]:
    """Estimate send times of the seqs missing between two received messages.

    If seq 10 was sent at t=5.0 and seq 13 at t=6.5, then 11 -> 5.5 and 12 -> 6.0.
    """
    out = []
    span = seq_b - seq_a
    for s in range(seq_a + 1, seq_b):
        out.append((s, t_a + (t_b - t_a) * (s - seq_a) / span))
    return out


def health_score(recv_ratio: float, delay_p95_ms: float, jitter_ms: float, connected: bool,
                 weights=(0.25, 0.25, 0.25, 0.25), delay_limit_ms=1000.0, jitter_limit_ms=200.0) -> float:
    """H = 100 * (w1*PDR + w2*s_delay + w3*s_jitter + w4*s_uptime), each s in [0, 1], 1 = ideal.

    A summary for people to read, not a research result. Weights are documented and adjustable.
    """
    def clip01(x):
        return 0.0 if x is None or math.isnan(x) else max(0.0, min(1.0, x))

    s_pdr = clip01(recv_ratio)
    s_delay = clip01(1 - delay_p95_ms / delay_limit_ms) if not math.isnan(delay_p95_ms) else s_pdr
    s_jit = clip01(1 - jitter_ms / jitter_limit_ms) if not math.isnan(jitter_ms) else s_pdr
    s_up = 1.0 if connected else 0.0
    w1, w2, w3, w4 = weights
    return round(100 * (w1 * s_pdr + w2 * s_delay + w3 * s_jit + w4 * s_up), 1)
