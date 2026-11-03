"""Resumable dated security-master snapshots; no inferred listing continuity."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any
from uuid import uuid4


def session_dates(start: date, end: date) -> list[date]:
    """Return NYSE sessions, inclusive, including exchange holidays/closures."""
    if start > end:
        raise ValueError("start must not exceed end")
    import exchange_calendars as xcals

    calendar = xcals.get_calendar("XNYS")
    return [
        stamp.date() for stamp in calendar.sessions_in_range(start.isoformat(), end.isoformat())
    ]


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def reconstruct_dated_window(
    *,
    dates: Sequence[date],
    manifest_path: Path,
    store_root: Path,
    discover_and_save: Callable[[date], tuple[str, dict[str, Any]]],
    max_sessions: int,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Save each day as its own immutable master, with an atomic resume manifest.

    Previously successful dates are never re-requested. Failed dates can be
    retried. The manifest is scoped to one exact date sequence and store root.
    """
    if max_sessions < 1:
        raise ValueError("max_sessions must be positive")
    keys = [day.isoformat() for day in dates]
    if keys != sorted(set(keys)):
        raise ValueError("dates must be sorted and unique")
    resolved_root = str(store_root.resolve())
    if manifest_path.exists():
        payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            payload.get("format") != 1
            or payload.get("dates") != keys
            or payload.get("store_root") != resolved_root
            or not isinstance(payload.get("results"), dict)
        ):
            raise ValueError("Existing manifest has different dates, format, or store root")
    else:
        payload = {
            "format": 1,
            "dates": keys,
            "store_root": resolved_root,
            "results": {},
            "note": "Retrospective provider assertions, one-day validity; not historical knowledge.",
        }
    attempted = 0
    for key in keys:
        previous = payload["results"].get(key)
        if previous is not None and previous.get("status") == "ok":
            continue
        if attempted >= max_sessions:
            break
        attempted += 1
        try:
            snapshot_version, report = discover_and_save(date.fromisoformat(key))
            if not snapshot_version:
                raise ValueError("Empty snapshot version")
            payload["results"][key] = {
                "status": "ok",
                "snapshot_version": snapshot_version,
                "report": report,
            }
            message = f"OK {key} -> {snapshot_version}"
        except Exception as exc:
            payload["results"][key] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            message = f"FAILED {key}: {type(exc).__name__}: {exc}"
        _write_json_atomic(manifest_path, payload)
        if on_progress is not None:
            on_progress(message)
    return payload
