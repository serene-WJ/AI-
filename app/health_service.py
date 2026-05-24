from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from sqlite3 import Row

from app.database import get_connection
from app.schemas import (
    HealthKitImportRequest,
    HealthKitImportResponse,
    HealthKitSample,
    ShortcutHeartRateRequest,
    ShortcutHeartRateResponse,
    WatchHealthData,
)


def ingest_watch_health(data: WatchHealthData) -> WatchHealthData:
    if data.timestamp is None:
        data.timestamp = time.time()
    _save_latest_health(data)
    return data


def latest_watch_health(user_id: str = "default") -> WatchHealthData | None:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT user_id, session_id, age, heart_rate, sleep_quality, sleep_hours,
                   heart_rate_recovery_seconds, timestamp
            FROM watch_health_latest
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
    return _row_to_watch_health(row) if row else None


def import_healthkit_samples(request: HealthKitImportRequest) -> HealthKitImportResponse:
    imported = [
        _normalize_sample(sample, request.user_id, request.session_id, request.device_id)
        for sample in request.samples
    ]
    imported_at = time.time()

    with get_connection() as connection:
        connection.executemany(
            """
            INSERT INTO healthkit_samples (
                user_id, session_id, device_id, sample_type, value_text, value_number,
                unit, start_time, end_time, source, metadata_json, imported_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [_sample_to_db_tuple(sample, imported_at) for sample in imported],
        )
        connection.commit()

    health = _summarize_healthkit_samples(
        user_id=request.user_id,
        session_id=request.session_id,
        age=request.age,
        samples=imported,
        timestamp=request.timestamp,
    )
    ingest_watch_health(health)

    return HealthKitImportResponse(
        imported_count=len(imported),
        health=health,
        samples=imported,
    )


def ingest_shortcut_heart_rate(request: ShortcutHeartRateRequest) -> ShortcutHeartRateResponse:
    timestamp = _parse_timestamp(request.timestamp) or time.time()
    sample = HealthKitSample(
        user_id=request.user_id,
        session_id=request.session_id,
        device_id=request.device_id,
        sample_type="heart_rate",
        value=request.heart_rate,
        unit=request.unit,
        start_time=timestamp,
        end_time=timestamp,
        source=request.source,
        metadata=request.metadata,
    )
    response = import_healthkit_samples(
        HealthKitImportRequest(
            user_id=request.user_id,
            session_id=request.session_id,
            device_id=request.device_id,
            age=request.age,
            samples=[sample],
            timestamp=timestamp,
        )
    )
    return ShortcutHeartRateResponse(health=response.health, sample=response.samples[0])


def latest_healthkit_samples(
    limit: int = 50,
    sample_type: str | None = None,
    user_id: str = "default",
    session_id: str | None = None,
) -> list[HealthKitSample]:
    query = """
        SELECT user_id, session_id, device_id, sample_type, value_text, value_number,
               unit, start_time, end_time, source, metadata_json
        FROM healthkit_samples
        WHERE user_id = ?
    """
    params: list[object] = [user_id]
    if sample_type is not None:
        query += " AND sample_type = ?"
        params.append(sample_type)
    if session_id is not None:
        query += " AND session_id = ?"
        params.append(session_id)
    query += " ORDER BY COALESCE(end_time, start_time, imported_at) DESC LIMIT ?"
    params.append(limit)

    with get_connection() as connection:
        rows = connection.execute(query, params).fetchall()

    return [_row_to_sample(row) for row in rows]


def _save_latest_health(data: WatchHealthData) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO watch_health_latest (
                user_id, session_id, age, heart_rate, sleep_quality, sleep_hours,
                heart_rate_recovery_seconds, timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                session_id = excluded.session_id,
                age = excluded.age,
                heart_rate = excluded.heart_rate,
                sleep_quality = excluded.sleep_quality,
                sleep_hours = excluded.sleep_hours,
                heart_rate_recovery_seconds = excluded.heart_rate_recovery_seconds,
                timestamp = excluded.timestamp
            """,
            (
                data.user_id,
                data.session_id,
                data.age,
                data.heart_rate,
                data.sleep_quality,
                data.sleep_hours,
                data.heart_rate_recovery_seconds,
                data.timestamp,
            ),
        )
        connection.commit()


def _normalize_sample(
    sample: HealthKitSample,
    user_id: str,
    session_id: str | None,
    device_id: str | None,
) -> HealthKitSample:
    now = time.time()
    sample_user_id = user_id if sample.user_id == "default" and user_id != "default" else sample.user_id
    update = {
        "user_id": sample_user_id,
        "session_id": sample.session_id if sample.session_id is not None else session_id,
        "device_id": sample.device_id if sample.device_id is not None else device_id,
    }
    if sample.start_time is None:
        update["start_time"] = now
    if sample.end_time is None:
        update["end_time"] = update.get("start_time", sample.start_time) or now
    return sample.model_copy(update=update)


def _parse_timestamp(value: float | str | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _sample_to_db_tuple(sample: HealthKitSample, imported_at: float) -> tuple[object, ...]:
    numeric_value = float(sample.value) if isinstance(sample.value, (int, float)) else None
    return (
        sample.user_id,
        sample.session_id,
        sample.device_id,
        sample.sample_type,
        str(sample.value),
        numeric_value,
        sample.unit,
        sample.start_time,
        sample.end_time,
        sample.source,
        json.dumps(sample.metadata, ensure_ascii=False),
        imported_at,
    )


def _summarize_healthkit_samples(
    user_id: str,
    session_id: str | None,
    age: int,
    samples: list[HealthKitSample],
    timestamp: float | None,
) -> WatchHealthData:
    previous = latest_watch_health(user_id)
    heart_rate = _latest_numeric_value(samples, "heart_rate")
    recovery = _latest_numeric_value(samples, "heart_rate_recovery")
    sleep_hours = _calculate_sleep_hours(samples)
    explicit_sleep_quality = _latest_text_value(samples, "sleep_quality")
    sleep_quality = _normalize_sleep_quality(
        explicit_quality=explicit_sleep_quality,
        sleep_hours=sleep_hours,
        previous=previous.sleep_quality if previous else None,
    )

    return WatchHealthData(
        user_id=user_id,
        session_id=session_id,
        age=age,
        heart_rate=round(heart_rate) if heart_rate is not None else (previous.heart_rate if previous else None),
        sleep_quality=sleep_quality,
        sleep_hours=round(sleep_hours, 2) if sleep_hours is not None else (previous.sleep_hours if previous else None),
        heart_rate_recovery_seconds=round(recovery)
        if recovery is not None
        else (previous.heart_rate_recovery_seconds if previous else None),
        timestamp=timestamp or time.time(),
    )


def _latest_numeric_value(samples: list[HealthKitSample], sample_type: str) -> float | None:
    candidates = [
        sample
        for sample in samples
        if sample.sample_type == sample_type and isinstance(sample.value, (int, float))
    ]
    if not candidates:
        return None
    latest = max(candidates, key=lambda sample: sample.end_time or sample.start_time or 0)
    return float(latest.value)


def _latest_text_value(samples: list[HealthKitSample], sample_type: str) -> str | None:
    candidates = [
        sample
        for sample in samples
        if sample.sample_type == sample_type and isinstance(sample.value, str)
    ]
    if not candidates:
        return None
    latest = max(candidates, key=lambda sample: sample.end_time or sample.start_time or 0)
    return latest.value


def _calculate_sleep_hours(samples: list[HealthKitSample]) -> float | None:
    sleep_samples = [sample for sample in samples if sample.sample_type == "sleep"]
    if not sleep_samples:
        return None

    total_hours = 0.0
    for sample in sleep_samples:
        if isinstance(sample.value, (int, float)):
            unit = (sample.unit or "hours").lower()
            if unit in {"s", "sec", "second", "seconds"}:
                total_hours += float(sample.value) / 3600
            elif unit in {"min", "minute", "minutes"}:
                total_hours += float(sample.value) / 60
            else:
                total_hours += float(sample.value)
            continue

        if sample.start_time is not None and sample.end_time is not None:
            total_hours += max(0.0, sample.end_time - sample.start_time) / 3600

    return total_hours


def _normalize_sleep_quality(
    explicit_quality: str | None,
    sleep_hours: float | None,
    previous: str | None,
) -> str:
    if explicit_quality in {"poor", "fair", "good"}:
        return explicit_quality
    if sleep_hours is None:
        return previous or "fair"
    if sleep_hours < 6:
        return "poor"
    if sleep_hours < 7:
        return "fair"
    return "good"


def _row_to_watch_health(row: Row) -> WatchHealthData:
    return WatchHealthData(
        user_id=row["user_id"],
        session_id=row["session_id"],
        age=row["age"],
        heart_rate=row["heart_rate"],
        sleep_quality=row["sleep_quality"],
        sleep_hours=row["sleep_hours"],
        heart_rate_recovery_seconds=row["heart_rate_recovery_seconds"],
        timestamp=row["timestamp"],
    )


def _row_to_sample(row: Row) -> HealthKitSample:
    value: float | str = row["value_number"] if row["value_number"] is not None else row["value_text"]
    return HealthKitSample(
        user_id=row["user_id"],
        session_id=row["session_id"],
        device_id=row["device_id"],
        sample_type=row["sample_type"],
        value=value,
        unit=row["unit"],
        start_time=row["start_time"],
        end_time=row["end_time"],
        source=row["source"],
        metadata=json.loads(row["metadata_json"]),
    )
