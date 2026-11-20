from iotmon.alerts import AlertEngine


def test_rule_alert_needs_two_bad_windows_and_two_good_to_clear():
    e = AlertEngine()
    assert e.update_rule("d", 0, 0.4, 0) is None              # 1st bad: not yet
    ev = e.update_rule("d", 5, 0.4, 0)
    assert ev.action == "raise" and ev.severity == "critical"
    assert e.update_rule("d", 10, 0.95, 0) is None            # in-between: neither bad nor good enough
    assert e.update_rule("d", 15, 1.0, 10) is None            # 1st good
    assert e.update_rule("d", 20, 1.0, 10).action == "clear"


def test_single_spike_does_not_alert():
    e = AlertEngine()
    events = [e.update_rule("d", i, r, 0) for i, r in enumerate([1, 0.2, 1, 0.2, 1, 1])]
    assert all(ev is None for ev in events)


def test_model_hysteresis():
    e = AlertEngine(model_threshold=0.4)
    assert e.update_model("d", 0, 0.6) is None
    assert e.update_model("d", 5, 0.5).action == "raise"
    assert e.update_model("d", 10, 0.3) is None               # below raise level but not below 0.2
    assert e.update_model("d", 15, 0.3) is None
    assert e.update_model("d", 20, 0.1) is None
    assert e.update_model("d", 25, 0.1).action == "clear"
