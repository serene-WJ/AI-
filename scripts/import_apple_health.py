from __future__ import annotations

import argparse
import json
import time
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator
from xml.etree.ElementTree import iterparse

HEART_RATE_TYPES = {
    "HKQuantityTypeIdentifierHeartRate": "heart_rate",
    "HKQuantityTypeIdentifierRestingHeartRate": "resting_heart_rate",
}
SLEEP_TYPE = "HKCategoryTypeIdentifierSleepAnalysis"
ASLEEP_VALUES = {
    "HKCategoryValueSleepAnalysisAsleep",
    "HKCategoryValueSleepAnalysisAsleepUnspecified",
    "HKCategoryValueSleepAnalysisAsleepCore",
    "HKCategoryValueSleepAnalysisAsleepDeep",
    "HKCategoryValueSleepAnalysisAsleepREM",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Apple Health export data into the backend.")
    parser.add_argument("export_path", help="Path to Apple Health export.zip or export.xml.")
    parser.add_argument("--user-id", default="default", help="Backend user id.")
    parser.add_argument("--session-id", default=None, help="Optional training session id.")
    parser.add_argument("--device-id", default="apple_watch", help="Optional device id.")
    parser.add_argument("--age", type=int, default=20, help="User age for heart-rate safety rules.")
    parser.add_argument("--days", type=int, default=7, help="Only import samples from the last N days.")
    parser.add_argument("--backend", default=None, help="Backend base URL, e.g. http://127.0.0.1:8000.")
    parser.add_argument("--out", default=None, help="Write converted JSON to this file.")
    args = parser.parse_args()

    export_xml, temp_dir = _resolve_export_xml(Path(args.export_path))
    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    samples = list(_parse_samples(export_xml, since=since))
    payload = {
        "user_id": args.user_id,
        "session_id": args.session_id,
        "device_id": args.device_id,
        "age": args.age,
        "timestamp": time.time(),
        "samples": samples,
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Wrote {len(samples)} samples to {args.out}")
    else:
        print(text)

    if args.backend:
        response = _post_json(f"{args.backend.rstrip('/')}/ingest/healthkit", payload)
        print(json.dumps(response, ensure_ascii=False, indent=2))

    if temp_dir is not None:
        temp_dir.cleanup()


def _resolve_export_xml(path: Path) -> tuple[Path, TemporaryDirectory[str] | None]:
    if path.suffix.lower() == ".xml":
        return path, None
    if path.suffix.lower() != ".zip":
        raise ValueError("Expected export.zip or export.xml")

    temp_dir = TemporaryDirectory()
    with zipfile.ZipFile(path) as archive:
        candidates = [name for name in archive.namelist() if name.endswith("export.xml")]
        if not candidates:
            raise ValueError("No export.xml found in the zip file.")
        archive.extract(candidates[0], temp_dir.name)

    return Path(temp_dir.name) / candidates[0], temp_dir


def _parse_samples(export_xml: Path, since: datetime) -> Iterator[dict[str, Any]]:
    for _, element in iterparse(export_xml, events=("end",)):
        if element.tag != "Record":
            element.clear()
            continue

        record_type = element.attrib.get("type")
        if record_type in HEART_RATE_TYPES:
            sample = _parse_heart_rate(element.attrib, HEART_RATE_TYPES[record_type], since)
            if sample:
                yield sample
        elif record_type == SLEEP_TYPE:
            sample = _parse_sleep(element.attrib, since)
            if sample:
                yield sample

        element.clear()


def _parse_heart_rate(
    attrs: dict[str, str],
    sample_type: str,
    since: datetime,
) -> dict[str, Any] | None:
    end_time = _parse_apple_time(attrs.get("endDate"))
    if end_time is None or end_time < since:
        return None

    try:
        value = float(attrs["value"])
    except (KeyError, ValueError):
        return None

    return {
        "sample_type": sample_type,
        "value": value,
        "unit": "bpm",
        "start_time": _to_unix(attrs.get("startDate")),
        "end_time": end_time.timestamp(),
        "source": attrs.get("sourceName", "apple_health"),
    }


def _parse_sleep(attrs: dict[str, str], since: datetime) -> dict[str, Any] | None:
    value = attrs.get("value")
    if value not in ASLEEP_VALUES:
        return None

    start_time = _parse_apple_time(attrs.get("startDate"))
    end_time = _parse_apple_time(attrs.get("endDate"))
    if start_time is None or end_time is None or end_time < since:
        return None

    hours = max(0.0, (end_time - start_time).total_seconds() / 3600)
    return {
        "sample_type": "sleep",
        "value": round(hours, 4),
        "unit": "hours",
        "start_time": start_time.timestamp(),
        "end_time": end_time.timestamp(),
        "source": attrs.get("sourceName", "apple_health"),
        "metadata": {"apple_sleep_value": value},
    }


def _parse_apple_time(value: str | None) -> datetime | None:
    if not value:
        return None
    for date_format in ("%Y-%m-%d %H:%M:%S %z", "%Y-%m-%d %H:%M:%S %Z"):
        try:
            return datetime.strptime(value, date_format).astimezone(timezone.utc)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(value).astimezone(timezone.utc)
    except ValueError:
        return None


def _to_unix(value: str | None) -> float | None:
    parsed = _parse_apple_time(value)
    return parsed.timestamp() if parsed else None


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    main()
