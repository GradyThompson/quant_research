"""Multi-year ORB aggregation, validation isolation, and stable candidate rules."""

from datetime import date, timedelta

import pytest
from test_opening_range import _sample

from quant_research.model.opening_range_history import (
    evaluate_opening_range_history,
    protected_research_days,
)


def test_extended_validation_excludes_warmup_trade_outcomes() -> None:
    assets, factor = _sample(17)
    start = assets[11].day
    report = evaluate_opening_range_history(
        asset_sessions=assets,
        factor_sessions=factor,
        validation_start=start,
        bootstrap_replicates=10,
    )
    assert report["warmup_sessions"] == 11
    assert report["validation_sessions"] == 6
    for experiment in report["experiments"]:
        results = experiment["variants"]["breakout"]
        assert results["cost_sensitivity"][1]["trades"] == 6
        assert all(date.fromisoformat(row["day"]) >= start for row in results["trade_examples"])
        assert results["by_year"][0]["sessions"] == 6
        assert (
            results["cost_sensitivity"][0]["net_bps_per_session"]
            > (results["cost_sensitivity"][1]["net_bps_per_session"])
        )


def test_validation_requires_earlier_sessions_and_factor_alignment() -> None:
    assets, factors = _sample(15)
    with pytest.raises(ValueError, match="warmup"):
        evaluate_opening_range_history(
            asset_sessions=assets,
            factor_sessions=factors,
            validation_start=assets[4].day,
        )
    with pytest.raises(ValueError, match="align"):
        evaluate_opening_range_history(
            asset_sessions=assets,
            factor_sessions=factors[:-1],
            validation_start=assets[11].day,
        )


def test_original_protected_days_are_calculated_from_frozen_anchor() -> None:
    rows = {}
    for idx in range(60):
        day = date(2024, 10, 1) + timedelta(days=idx)
        rows[str(idx)] = {"ticker": "AAPL", "session": day.isoformat(), "security_id": "x"}
    original = {"schema_version": 1, "entries": rows}
    protected = protected_research_days(original)
    assert date(2025, 1, 13) in protected
    assert len(protected) == 12
    assert date(2024, 10, 1) not in protected
