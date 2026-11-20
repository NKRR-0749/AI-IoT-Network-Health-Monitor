from iotmon.gaps import SequenceTracker

K = ("run", "dev-01")


def test_in_order_stream_has_no_loss():
    t = SequenceTracker()
    kinds = [t.observe(K, 1, s).kind for s in range(5)]
    assert kinds == ["first", "in_order", "in_order", "in_order", "in_order"]
    assert t.outstanding_missing(K) == 0


def test_gap_counts_missing_messages():
    t = SequenceTracker()
    t.observe(K, 1, 41)
    obs = t.observe(K, 1, 44)
    assert obs.kind == "gap" and obs.missing == 2
    assert t.outstanding_missing(K) == 2


def test_late_message_fills_gap():
    t = SequenceTracker()
    t.observe(K, 1, 41)
    t.observe(K, 1, 44)
    assert t.observe(K, 1, 42).kind == "late"
    assert t.outstanding_missing(K) == 1


def test_duplicate_detected():
    t = SequenceTracker()
    t.observe(K, 1, 5)
    t.observe(K, 1, 6)
    assert t.observe(K, 1, 6).kind == "duplicate"


def test_reboot_is_not_loss():
    t = SequenceTracker()
    for s in range(100, 103):
        t.observe(K, 1, s)
    assert t.observe(K, 2, 0).kind == "reboot"
    assert t.outstanding_missing(K) == 0
    assert t.observe(K, 1, 103).kind == "stale_boot"


def test_devices_are_independent():
    t = SequenceTracker()
    t.observe(("run", "a"), 1, 0)
    t.observe(("run", "b"), 1, 0)
    assert t.observe(("run", "a"), 1, 3).missing == 2
    assert t.observe(("run", "b"), 1, 1).kind == "in_order"
