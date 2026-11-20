"""Block: sequence-gap detection (application-level message loss).

Every device numbers its messages 0, 1, 2, ... If the collector sees 41 and
then 44, messages 42 and 43 are MISSING (so far). If 42 turns up later, it
was LATE (reordered / delayed), not lost. If the device reboots, seq restarts
at 0 with a higher boot_id, which must NOT be counted as loss.

This module is pure Python (no MQTT, no database) so it is easy to unit-test.
"""
from dataclasses import dataclass, field

MAX_TRACKED_MISSING = 10_000  # cap memory if a device sends garbage seq numbers


@dataclass
class Observation:
    kind: str          # first | in_order | gap | duplicate | late | reboot | stale_boot
    missing: int = 0   # how many seqs were skipped (only for 'gap')


@dataclass
class _DeviceState:
    boot_id: int
    highest_seq: int
    missing: set = field(default_factory=set)
    seen_count: int = 0


class SequenceTracker:
    """Tracks the latest boot of every (run, device) and classifies each new message."""

    def __init__(self) -> None:
        self._state: dict[tuple, _DeviceState] = {}

    def observe(self, key: tuple, boot_id: int, seq: int) -> Observation:
        st = self._state.get(key)
        if st is None:
            self._state[key] = _DeviceState(boot_id, seq, seen_count=1)
            return Observation("first")

        if boot_id > st.boot_id:            # device rebooted: start fresh, no loss counted
            self._state[key] = _DeviceState(boot_id, seq, seen_count=1)
            return Observation("reboot")
        if boot_id < st.boot_id:            # a delayed message from before the reboot
            return Observation("stale_boot")

        st.seen_count += 1
        if seq == st.highest_seq + 1:
            st.highest_seq = seq
            return Observation("in_order")
        if seq > st.highest_seq + 1:
            skipped = range(st.highest_seq + 1, seq)
            if len(st.missing) + len(skipped) <= MAX_TRACKED_MISSING:
                st.missing.update(skipped)
            st.highest_seq = seq
            return Observation("gap", missing=len(skipped))
        # seq <= highest_seq
        if seq in st.missing:
            st.missing.discard(seq)
            return Observation("late")
        st.seen_count -= 1                  # a repeat does not count as a new delivery
        return Observation("duplicate")

    def outstanding_missing(self, key: tuple) -> int:
        """Messages currently believed lost for this device (may still arrive late)."""
        st = self._state.get(key)
        return len(st.missing) if st else 0
