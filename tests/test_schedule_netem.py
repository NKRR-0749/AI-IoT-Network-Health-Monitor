from iotmon.impair.netem import netem_args
from iotmon.impair.schedule import generate, load_yaml, to_yaml


def test_generate_is_reproducible():
    a = generate("mixed", seed=7, total_s=400)
    b = generate("mixed", seed=7, total_s=400)
    assert a == b
    assert a[0].name == "clean" and a[-1].name == "clean"


def test_burst_profile_only_uses_bursty_loss():
    for seed in range(20):
        for p in generate("burst", seed, 400):
            if p.params.get("loss_pct"):
                assert p.params.get("burst_len", 0) > 1


def test_yaml_roundtrip(tmp_path):
    phases = generate("mixed", 3, 300)
    f = tmp_path / "s.yaml"
    f.write_text(to_yaml(phases))
    assert load_yaml(f) == phases


def test_netem_args():
    assert netem_args({"loss_pct": 5}) == "loss 5%"
    assert netem_args({"delay_ms": 100, "jitter_ms": 25}) == "delay 100ms 25ms distribution normal"
    assert netem_args({"outage": "drop"}) == "loss 100%"
    assert netem_args({}) == "delay 0ms"
    assert netem_args({"loss_pct": 5, "burst_len": 10}).startswith("loss gemodel")
