"""Deterministic timing, accounting, and failure-mode tests for P5-A."""

from __future__ import annotations

from datetime import date, timedelta
from math import exp

import pytest

from quant_research.model.conditional_residual import (
    DailyPrice,
    ResidualEvent,
    generate_residual_events,
    score_residual_events,
)


def _fixture() -> tuple[list[date], dict[date, dict[str, DailyPrice]]]:
    days = [date(2023, 1, 1) + timedelta(days=i) for i in range(135)]
    panel: dict[date, dict[str, DailyPrice]] = {}
    spy = 100.0
    xl = 100.0
    asset = 100.0
    for i, day in enumerate(days):
        spy *= exp(0.0004 + 0.0002 * ((i % 7) - 3))
        xl *= exp(0.0005 + 0.0003 * ((i % 11) - 5))
        shock = -0.012 if 85 <= i <= 89 else 0.0004 * ((i % 9) - 4)
        asset *= exp(0.0005 + 0.65 * (0.0003 * ((i % 11) - 5)) + shock)
        panel[day] = {
            "SPY": DailyPrice(spy, spy, 800_000_000),
            "XLK": DailyPrice(xl, xl, 200_000_000),
            "AAA": DailyPrice(asset, asset, 110_000_000),
        }
    return days, panel


def _run(
    days: list[date],
    panel: dict[date, dict[str, DailyPrice]],
) -> tuple[list[ResidualEvent], dict[str, int]]:
    return generate_residual_events(
        days=days,
        panel=panel,
        factor_tickers=("SPY", "XLK"),
        validation_start=days[75],
        shock_z=2.0,
        max_universe=1,
        holding_sessions=(3, 5),
        max_per_side=1,
    )


def test_no_future_price_enters_factor_fit_or_selection() -> None:
    days, panel = _fixture()
    original, _ = _run(days, panel)
    target = next(event for event in original if event.ticker == "AAA")
    changed = {day: dict(rows) for day, rows in panel.items()}
    exit_day = target.exit_day
    row = changed[exit_day]["AAA"]
    changed[exit_day]["AAA"] = DailyPrice(
        row.opening * 1.10,
        row.closing,
        row.dollar_volume,
    )
    modified, _ = _run(days, changed)
    matching = next(
        event
        for event in modified
        if event.formation_day == target.formation_day
        and event.holding_sessions == target.holding_sessions
    )
    assert matching.z_score == pytest.approx(target.z_score)
    assert matching.beta == pytest.approx(target.beta)
    assert matching.stock_return_bps != pytest.approx(target.stock_return_bps)
    assert matching.entry_day > matching.formation_day
    assert matching.exit_day > matching.entry_day


def test_missing_protected_session_prevents_crossing_lookback() -> None:
    days, panel = _fixture()
    panel[days[85]] = {}
    results, diagnostics = _run(days, panel)
    assert diagnostics.get("missing_market_regime", 0) > 0
    assert all(event.formation_day != days[85] for event in results)


def test_split_like_discontinuity_is_excluded() -> None:
    days, panel = _fixture()
    original = panel[days[85]]["AAA"]
    panel[days[85]]["AAA"] = DailyPrice(
        original.opening * 0.5,
        original.closing * 0.5,
        original.dollar_volume,
    )
    _, counts = _run(days, panel)
    assert counts["invalid_stock_prices_or_split_risk"] > 0


def test_score_hedge_cost_and_cash_sessions() -> None:
    days = [date(2024, 1, 2) + timedelta(days=i) for i in range(3)]
    event = ResidualEvent(
        formation_day=days[0],
        ticker="AAA",
        factor="XLK",
        beta=1.25,
        z_score=-3.0,
        regime_efficiency=0.1,
        holding_sessions=3,
        entry_day=days[1],
        exit_day=days[2],
        stock_return_bps=30.0,
        factor_return_bps=8.0,
    )
    result = score_residual_events(
        events=[event],
        eligible_days=days,
        horizon=3,
        side="negative_long",
        regime="range",
        roundtrip_cost_bps=4,
        bootstrap_repeats=30,
        slots=1,
    )
    # Gross 30 - 1.25*8=20; cost (1+1.25)*4=9; 3 entry sessions.
    assert result["net_bps_per_trade"] == pytest.approx(11.0)
    assert result["net_bps_per_entry_session"] == pytest.approx(11.0 / 3)
    assert result["trades"] == 1
    no_trade = score_residual_events(
        events=[event],
        eligible_days=days,
        horizon=3,
        side="positive_short",
        regime="all",
        roundtrip_cost_bps=4,
    )
    assert no_trade["net_bps_per_entry_session"] == 0
    assert no_trade["trades"] == 0


def test_regime_and_positive_shock_are_distinct() -> None:
    days = [date(2024, 1, 2), date(2024, 1, 3)]
    event = ResidualEvent(
        formation_day=days[0],
        ticker="AAA",
        factor="SPY",
        beta=1.0,
        z_score=3.0,
        regime_efficiency=0.7,
        holding_sessions=5,
        entry_day=days[0],
        exit_day=days[1],
        stock_return_bps=10,
        factor_return_bps=1,
    )
    result = score_residual_events(
        events=[event],
        eligible_days=days,
        horizon=5,
        side="positive_short",
        regime="trend",
        roundtrip_cost_bps=2,
        bootstrap_repeats=30,
    )
    assert result["net_bps_per_trade"] == pytest.approx(-13)
    absent = score_residual_events(
        events=[event],
        eligible_days=days,
        horizon=5,
        side="positive_short",
        regime="range",
        roundtrip_cost_bps=2,
    )
    assert absent["trades"] == 0


def test_reject_invalid_dates_and_costs() -> None:
    days, panel = _fixture()
    with pytest.raises(ValueError, match="invalid conditional"):
        generate_residual_events(
            days=list(reversed(days)),
            panel=panel,
            factor_tickers=("SPY",),
            validation_start=days[70],
        )
    with pytest.raises(ValueError, match="invalid residual"):
        score_residual_events(
            events=[],
            eligible_days=days,
            horizon=5,
            side="negative_long",
            regime="all",
            roundtrip_cost_bps=-1,
        )


def test_future_holding_window_does_not_cross_missing_protected_day() -> None:
    days, panel = _fixture()
    panel[days[90]] = {}
    events, counts = _run(days, panel)
    assert counts["missing_or_discontinuous_exit"] > 0
    for event in events:
        assert not (event.entry_day <= days[90] <= event.exit_day)
