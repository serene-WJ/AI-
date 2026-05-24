from __future__ import annotations

import sqlite3
from pathlib import Path

from app.config import BASE_DIR

DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "assistant.db"


def get_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    _init_schema(connection)
    return connection


def _init_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS healthkit_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            session_id TEXT,
            device_id TEXT,
            sample_type TEXT NOT NULL,
            value_text TEXT NOT NULL,
            value_number REAL,
            unit TEXT,
            start_time REAL,
            end_time REAL,
            source TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            imported_at REAL NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_healthkit_samples_user_type_time
        ON healthkit_samples(user_id, sample_type, end_time)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS watch_health_latest (
            user_id TEXT PRIMARY KEY,
            session_id TEXT,
            age INTEGER NOT NULL,
            heart_rate INTEGER,
            sleep_quality TEXT NOT NULL,
            sleep_hours REAL,
            heart_rate_recovery_seconds INTEGER,
            timestamp REAL NOT NULL
        )
        """
    )
    connection.commit()
