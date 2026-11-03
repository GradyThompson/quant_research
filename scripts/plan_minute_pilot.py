"""Create a deterministic, immutable minute-ingestion pilot plan (no API requests)."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from quant_research.paths import resolve_project_path
from quant_research.security_master.minute_pilot import minute_session_plan, select_minute_pilot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--universe", type=Path, default=Path("data/dev/liquid_universe_market_20250113.json")
    )
    parser.add_argument("--output", type=Path, default=Path("data/dev/minute_pilot_20250113.json"))
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--initialization-sessions", type=int, default=5)
    args = parser.parse_args()

    source = resolve_project_path(args.universe)
    screen = json.loads(source.read_text(encoding="utf-8"))
    evaluation_date = date.fromisoformat(screen["as_of"])
    if screen.get("reference_data_point_in_time_verified") is not False:
        parser.error("Expected explicit retrospective-reference-data warning")
    members = screen["members"]
    if screen.get("included") != len(members):
        parser.error("Screened-universe member count mismatch")
    selected, exclusions = select_minute_pilot(members, count=args.count)
    sessions = minute_session_plan(
        evaluation_date=evaluation_date, initialization_sessions=args.initialization_sessions
    )
    result = {
        "schema_version": 1,
        "evaluation_date": evaluation_date.isoformat(),
        "initialization_sessions": [d.isoformat() for d in sessions[:-1]],
        "evaluation_session": sessions[-1].isoformat(),
        "source_universe": str(source),
        "source_daily_datasets": screen["source_daily_datasets"],
        "security_master_snapshot_version": screen["security_master_snapshot_version"],
        "reference_data_point_in_time_verified": False,
        "minute_data_publication_time_verified": False,
        "members": selected,
        "unsupported_exclusions": exclusions,
        "planned_requests": len(selected) * len(sessions),
        "status": "planned_only_no_data_downloaded",
    }
    output = resolve_project_path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing plan: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(f"Selected: {len(selected)} supported securities from {len(members)}")
    print(f"Excluded unsupported: {exclusions}")
    print(f"Initialization: {sessions[0]} through {sessions[-2]}")
    print(f"Evaluation: {sessions[-1]}")
    print(f"Planned minute requests: {result['planned_requests']} (not executed)")
    print(f"Output: {output}")
    print("NOTICE: retrospective reference mapping; historical availability unverified")


if __name__ == "__main__":
    main()
