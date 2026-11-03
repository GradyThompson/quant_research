"""Bounded, resumable market-wide daily-file download/import orchestration."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any


def process_daily_window(
    *,
    security_manifest: Path,
    import_root: Path,
    max_sessions: int,
    process_session: Callable[[date, str], dict[str, object]],
    on_progress: Callable[[str], None] | None = None,
    validate_existing: Callable[[dict[str, object]], bool] | None = None,
) -> dict[str, int]:
    """Process dated snapshots with explicit pins; retry only incomplete sessions.

    `process_session` downloads a verified source and imports it using the exact
    snapshot UUID. A completed import manifest is authoritative only if the
    caller validates its underlying pinned dataset when resuming.
    """
    if max_sessions < 1:
        raise ValueError("max_sessions must be positive")
    raw: dict[str, Any] = json.loads(security_manifest.read_text(encoding="utf-8"))
    dates = raw.get("dates")
    results = raw.get("results")
    if not isinstance(dates, list) or not isinstance(results, dict):
        raise ValueError("Invalid security master window manifest")
    if dates != sorted(set(dates)):
        raise ValueError("Manifest dates must be unique and sorted")
    if any(results.get(day, {}).get("status") != "ok" for day in dates):
        raise ValueError("All security-master sessions must be complete before importing")

    completed = failed = attempted = 0
    for day_text in dates:
        if not isinstance(day_text, str):
            raise ValueError("Invalid session date in manifest")
        snapshot = results[day_text].get("snapshot_version")
        if not isinstance(snapshot, str) or not snapshot:
            raise ValueError(f"Missing snapshot version for {day_text}")
        day = date.fromisoformat(day_text)
        existing_path = import_root / f"{day_text}.json"
        if existing_path.exists():
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
            if existing.get("snapshot_version") != snapshot or existing.get("session") != day_text:
                raise ValueError(f"Conflicting pinned import for {day_text}; refusing to overwrite")
            if not existing.get("dataset_version") or not existing.get("sha256"):
                raise ValueError(f"Malformed completed import for {day_text}")
            if validate_existing is not None and not validate_existing(existing):
                raise ValueError(f"Pinned dataset for {day_text} is missing or invalid")
            completed += 1
            continue
        if attempted >= max_sessions:
            continue
        attempted += 1
        try:
            result = process_session(day, snapshot)
            if result.get("session") != day_text or result.get("snapshot_version") != snapshot:
                raise ValueError("Importer returned mismatched provenance")
            completed += 1
            message = f"OK {day_text}: {result.get('dataset_id')} / {result.get('dataset_version')}"
        except Exception as exc:
            failed += 1
            message = f"FAILED {day_text}: {type(exc).__name__}: {exc}"
        if on_progress is not None:
            on_progress(message)
    return {
        "total": len(dates),
        "completed": completed,
        "attempted": attempted,
        "failed_this_run": failed,
        "remaining": len(dates) - completed,
    }
