"""Database helpers.

Usage from the command line:
    python -m iotmon.db init     # create all tables (safe to run twice)
    python -m iotmon.db reset    # DROP everything and recreate (asks first)
"""
import sys

import psycopg
from psycopg.types.json import Jsonb

from .config import ROOT, settings

SCHEMA_FILE = ROOT / "db" / "schema.sql"
TABLES = ["alerts", "predictions", "windows", "impairments", "connection_events",
          "udp_probes", "rtt_probes", "messages", "devices", "runs"]


def connect(autocommit: bool = True) -> psycopg.Connection:
    return psycopg.connect(settings.db_dsn, autocommit=autocommit)


def init_schema(conn=None) -> None:
    own = conn is None
    conn = conn or connect()
    conn.execute(SCHEMA_FILE.read_text())
    if own:
        conn.close()


def reset_schema() -> None:
    with connect() as conn:
        for t in TABLES:
            conn.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
        init_schema(conn)


def ensure_run(conn, run_id: str, config: dict | None = None, seed: int | None = None) -> None:
    """Create the run row if it does not exist yet (the collector calls this for unknown runs)."""
    conn.execute(
        "INSERT INTO runs (run_id, config, seed) VALUES (%s, %s, %s) ON CONFLICT (run_id) DO NOTHING",
        (run_id, Jsonb(config or {}), seed),
    )


def ensure_device(conn, device_id: str, kind: str = "emulated") -> None:
    conn.execute(
        "INSERT INTO devices (device_id, kind) VALUES (%s, %s) ON CONFLICT (device_id) DO NOTHING",
        (device_id, kind),
    )


def main(argv: list[str]) -> None:
    cmd = argv[1] if len(argv) > 1 else "init"
    if cmd == "init":
        init_schema()
        print("Schema ready.")
    elif cmd == "reset":
        if "--yes" not in argv and input("Delete ALL data and recreate tables? [y/N] ").lower() != "y":
            print("Cancelled.")
            return
        reset_schema()
        print("All tables dropped and recreated.")
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv)
