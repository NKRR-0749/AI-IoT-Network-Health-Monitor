"""Central settings, read from environment variables (or a .env file).

Every other module imports settings from here, so you change a port or a
password in ONE place. Defaults work for a local setup on one laptop.
"""
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # the repository folder


def _load_dotenv() -> None:
    """Tiny .env reader: KEY=VALUE lines. Real environment variables win."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    # PostgreSQL
    db_dsn: str = os.getenv("IOT_DB_DSN", "postgresql://iot:iot@localhost:5432/iot_monitor")
    # MQTT broker the COLLECTOR talks to (directly, never through the fault injector)
    mqtt_host: str = os.getenv("IOT_MQTT_HOST", "localhost")
    mqtt_port: int = int(os.getenv("IOT_MQTT_PORT", "1883"))
    # UDP port where the collector receives network-layer probe packets
    udp_probe_port: int = int(os.getenv("IOT_UDP_PROBE_PORT", "9999"))
    # Analysis settings
    window_s: int = int(os.getenv("IOT_WINDOW_S", "5"))          # metric window length
    ontime_ms: float = float(os.getenv("IOT_ONTIME_MS", "1000"))  # a message is "on time" if its relative delay <= this
    horizon_s: int = int(os.getenv("IOT_HORIZON_S", "30"))       # how far ahead the model predicts


settings = Settings()

# MQTT topic layout:  iot/<run_id>/<device_id>/<kind>
TOPIC_ROOT = "iot"


def topic(run_id: str, device_id: str, kind: str) -> str:
    return f"{TOPIC_ROOT}/{run_id}/{device_id}/{kind}"


def parse_topic(t: str):
    """'iot/run1/dev-01/telemetry' -> ('run1', 'dev-01', 'telemetry'); None if not ours."""
    parts = t.split("/")
    if len(parts) != 4 or parts[0] != TOPIC_ROOT:
        return None
    return parts[1], parts[2], parts[3]
