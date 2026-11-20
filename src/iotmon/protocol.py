"""The message formats shared by the ESP32 firmware, the emulator and the collector.

Telemetry (topic iot/<run>/<device>/telemetry), JSON:
    {"device_id": "dev-01", "seq": 1042, "boot_id": 7, "uptime_ms": 1042331,
     "qos": 1, "interval_ms": 500, "rssi": -61}

  seq        increases by 1 for every message since boot (starts at 0)
  boot_id    increases by 1 on every reboot -> a seq reset is a reboot, not loss
  uptime_ms  device clock (ms since boot); used to place messages in time

RTT ping  (device -> iot/<run>/<device>/ping):  {"probe_seq": 12, "t": <device ms>}
RTT pong  (collector -> iot/<run>/<device>/pong): the same payload echoed back
RTT result (device -> iot/<run>/<device>/rtt):  {"probe_seq": 12, "rtt_ms": 34.2}  (rtt_ms null = lost)
Status    (retained, iot/<run>/<device>/status): "online" / "offline" (Last Will)

UDP probe (device -> collector UDP port), JSON:
    {"run_id": "r1", "device_id": "dev-01", "boot_id": 7, "probe_seq": 55, "uptime_ms": 1042331}
"""
import json

REQUIRED_TELEMETRY = ("device_id", "seq", "boot_id", "uptime_ms")


class BadMessage(ValueError):
    pass


def encode(obj: dict) -> bytes:
    return json.dumps(obj, separators=(",", ":")).encode()


def decode_telemetry(raw: bytes) -> dict:
    """Parse and validate a telemetry payload. Raises BadMessage if it is unusable."""
    try:
        msg = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as e:
        raise BadMessage(f"not JSON: {e}") from e
    if not isinstance(msg, dict):
        raise BadMessage("payload is not a JSON object")
    missing = [k for k in REQUIRED_TELEMETRY if k not in msg]
    if missing:
        raise BadMessage(f"missing fields: {missing}")
    try:
        msg["seq"] = int(msg["seq"])
        msg["boot_id"] = int(msg["boot_id"])
        msg["uptime_ms"] = int(msg["uptime_ms"])
        msg["qos"] = int(msg.get("qos", 0))
    except (TypeError, ValueError) as e:
        raise BadMessage(f"bad number: {e}") from e
    if msg["seq"] < 0:
        raise BadMessage("negative seq")
    return msg
