"""Block: alert engine with hysteresis.

Two independent alert sources per device:
  rule  : last window recv_ratio < 0.9  OR  delay p95 > 1000 ms
  model : predicted degradation score >= model threshold

Hysteresis keeps alerts from flickering:
  raise only after RAISE_AFTER bad windows in a row,
  clear only after CLEAR_AFTER good windows in a row, and for the model the
  "good" level is lower than the raise level (score < threshold * 0.5).
"""
from dataclasses import dataclass, field

RAISE_AFTER = 2
CLEAR_AFTER = 2


@dataclass
class _Track:
    active: bool = False
    bad_run: int = 0
    good_run: int = 0


@dataclass
class AlertEvent:
    device_id: str
    source: str          # rule | model
    action: str          # raise | clear
    at: float
    reason: str = ""
    severity: str = "warning"


@dataclass
class AlertEngine:
    model_threshold: float = 0.5
    tracks: dict = field(default_factory=dict)

    def _step(self, key, bad: bool, good: bool) -> str | None:
        t = self.tracks.setdefault(key, _Track())
        t.bad_run = t.bad_run + 1 if bad else 0
        t.good_run = t.good_run + 1 if good else 0
        if not t.active and t.bad_run >= RAISE_AFTER:
            t.active = True
            return "raise"
        if t.active and t.good_run >= CLEAR_AFTER:
            t.active = False
            return "clear"
        return None

    def update_rule(self, device_id: str, at: float, recv_ratio: float, delay_p95_ms: float) -> AlertEvent | None:
        rr = 0.0 if recv_ratio is None or recv_ratio != recv_ratio else recv_ratio
        dp = 0.0 if delay_p95_ms is None or delay_p95_ms != delay_p95_ms else delay_p95_ms
        bad = rr < 0.9 or dp > 1000
        good = rr >= 0.97 and dp <= 500
        act = self._step((device_id, "rule"), bad, good)
        if act:
            sev = "critical" if rr < 0.5 else "warning"
            return AlertEvent(device_id, "rule", act, at, f"recv_ratio={rr:.2f}, delay_p95={dp:.0f} ms", sev)
        return None

    def update_model(self, device_id: str, at: float, score: float) -> AlertEvent | None:
        bad = score >= self.model_threshold
        good = score < self.model_threshold * 0.5
        act = self._step((device_id, "model"), bad, good)
        if act:
            return AlertEvent(device_id, "model", act, at, f"degradation risk {score:.2f} in next window horizon")
        return None
