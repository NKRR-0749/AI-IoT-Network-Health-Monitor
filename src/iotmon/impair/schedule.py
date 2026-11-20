"""Block: impairment schedules (what the fault injector does, and when).

A schedule is a list of phases. Each phase has a name, a duration and the
impairment parameters. Schedules are either loaded from YAML
(experiments/schedules/*.yaml) or generated from a profile + random seed, so
every run is reproducible from (profile, seed).

Profiles
  mixed : random loss, delay/jitter, stalls, connection drops, gradual ramps
  burst : same structure, but every loss phase uses Gilbert-Elliott BURSTS.
          Kept out of training for the unseen-condition test (RQ3).
"""
import random
from dataclasses import asdict, dataclass, field

import yaml


@dataclass
class Phase:
    name: str
    duration_s: float
    params: dict = field(default_factory=dict)   # loss_pct, burst_len, delay_ms, jitter_ms, outage


def clean(d):
    return Phase("clean", d, {})


def _loss_phase(rng, profile, intensity=None):
    loss = intensity if intensity is not None else rng.choice([1, 5, 10, 20, 30])
    if profile == "burst":
        return Phase("burst_loss", rng.uniform(25, 50), {"loss_pct": loss, "burst_len": rng.choice([5, 10, 20])})
    return Phase("random_loss", rng.uniform(25, 50), {"loss_pct": loss})


def _ramp(rng, profile, kind):
    """Gradual degradation: 6 steps getting worse, then recovery. Gives the model something to anticipate."""
    steps, step_s = 6, rng.uniform(8, 12)
    out = []
    for i in range(1, steps + 1):
        f = i / steps
        if kind == "loss":
            p = {"loss_pct": round(30 * f, 1)}
            if profile == "burst":
                p["burst_len"] = 10
            out.append(Phase("ramp_loss", step_s, p))
        else:
            out.append(Phase("ramp_delay", step_s, {"delay_ms": round(800 * f), "jitter_ms": round(200 * f)}))
    return out


def generate(profile: str, seed: int, total_s: float) -> list[Phase]:
    rng = random.Random(seed)
    phases = [clean(45)]
    t = 45
    while t < total_s - 40:
        kind = rng.choice(["loss", "loss", "delay", "stall", "drop", "ramp_loss", "ramp_delay"])
        if kind == "loss":
            block = [_loss_phase(rng, profile)]
        elif kind == "delay":
            d = rng.choice([100, 300, 600, 1200])
            block = [Phase("delay", rng.uniform(25, 50), {"delay_ms": d, "jitter_ms": d // 4})]
        elif kind == "stall":
            block = [Phase("outage_stall", rng.choice([3, 6, 10]), {"outage": "stall"})]
        elif kind == "drop":
            block = [Phase("outage_drop", rng.choice([5, 15, 30]), {"outage": "drop"})]
        else:
            block = _ramp(rng, profile, "loss" if kind == "ramp_loss" else "delay")
        block.append(clean(rng.uniform(20, 45)))
        phases += block
        t += sum(p.duration_s for p in block)
    phases.append(clean(30))
    return phases


def load_yaml(path) -> list[Phase]:
    with open(path) as f:
        doc = yaml.safe_load(f)
    return [Phase(p["name"], float(p["duration_s"]), p.get("params", {}) or {}) for p in doc["phases"]]


def to_yaml(phases: list[Phase]) -> str:
    return yaml.safe_dump({"phases": [asdict(p) for p in phases]}, sort_keys=False)
