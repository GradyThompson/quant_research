"""P4.5 causal universe, event timing and fixed-capital portfolio tests."""

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from quant_research.data.ingestion.minute_market_summary import DailySummary
from quant_research.model.cross_sectional_orb import (
    Candidate,
    Event,
    RollingUniverse,
    first_breakout_event,
    score_events,
)
from quant_research.model.first_passage_validation import MinuteBar, Session


def _summary(*, volume=100, dollars=100_000_000, price=100, high=101, low=99):
    return DailySummary(
        opening_volume=volume,
        opening_high=high,
        opening_low=low,
        opening_open=price,
        opening_close=price,
        last_close=price,
        dollar_volume=dollars,
    )


def test_selector_only_uses_prior_sessions_and_liquidity_matches_control():
    selector = RollingUniverse(lookback=2, count=1, min_rvol=1.5)
    old = {"AAA": _summary(), "BBB": _summary()}
    assert selector.observe(old) == []
    assert selector.observe(old) == []
    selected = selector.observe(
        {
            "AAA": _summary(volume=300),
            "BBB": _summary(volume=100),
        }
    )
    assert [(c.ticker, c.cohort) for c in selected] == [
        ("AAA", "high_rvol"),
        ("BBB", "normal_rvol_control"),
    ]
    assert selected[0].rvol == 3
    # The new high-volume day is part of the *next* prior history, not this
    # day's denominator. No five-minute or daily future volume leaks in.
    selected2 = selector.observe(
        {
            "AAA": _summary(volume=300),
            "BBB": _summary(volume=100),
        }
    )
    assert selected2[0].rvol == pytest.approx(1.5)


def test_selector_rejects_missing_intermediate_session():
    selector = RollingUniverse(lookback=2)
    selector.observe({"AAA": _summary()})
    selector.observe({})
    selector.observe({"AAA": _summary(volume=1000)})
    assert selector.observe({"AAA": _summary(volume=2000)}) == []


def _session(day, ticker, *, breakout, qqq=False):
    del ticker
    stamp = datetime(
        day.year, day.month, day.day, 9, 31, tzinfo=ZoneInfo("America/New_York")
    ).astimezone(UTC)
    bars = []
    for i in range(200):
        price = 100.0
        if breakout and i >= 16:
            price = 102.0 + i * 0.001
        if qqq and i >= 16:
            price = 103.0
        bars.append(
            MinuteBar(
                timestamp=stamp + timedelta(minutes=i),
                available_at=stamp + timedelta(minutes=i + 1),
                open=price,
                high=price + 0.1,
                low=price - 0.1,
                close=price,
                volume=100,
            )
        )
    return Session(day, tuple(bars))


def test_first_breakout_event_strictly_future_entry():
    day = date(2024, 6, 10)
    candidate = Candidate("AAA", "high_rvol", 2.0, 100_000_000, 100.1, 99.9)
    asset = _session(day, "AAA", breakout=True)
    factor = _session(day, "QQQ", breakout=False, qqq=True)
    found = first_breakout_event(
        candidate=candidate,
        asset=asset,
        factor=factor,
        holding_minutes=30,
    )
    assert found is not None
    assert found.market_aligned
    assert found.decision_at < found.entry_at < found.exit_at
    assert found.side == 1
    assert found.day == day


def test_first_breakout_returns_none_without_move():
    day = date(2024, 6, 10)
    asset = _session(day, "AAA", breakout=False)
    candidate = Candidate("AAA", "high_rvol", 2.0, 100_000_000, 100.1, 99.9)
    assert (
        first_breakout_event(
            candidate=candidate,
            asset=asset,
            factor=asset,
            holding_minutes=30,
        )
        is None
    )


def test_portfolio_cost_scenarios_use_identical_trades_and_fixed_cash_slots():
    d = date(2024, 6, 10)
    stamp = datetime(2024, 6, 10, 13, 50, tzinfo=UTC)
    events = [
        Event(
            d,
            "AAA",
            "high_rvol",
            2.0,
            1,
            stamp,
            stamp + timedelta(minutes=1),
            stamp + timedelta(minutes=31),
            20.0,
            True,
        ),
        Event(
            d,
            "BBB",
            "high_rvol",
            1.7,
            -1,
            stamp,
            stamp + timedelta(minutes=1),
            stamp + timedelta(minutes=31),
            -5.0,
            False,
        ),
    ]
    all_ = score_events(
        events=events,
        eligible_days=[d, date(2024, 6, 11)],
        cohort="high_rvol",
        confirmed=False,
        positions=2,
        costs_bps=(8, 12),
        bootstrap_replicates=10,
    )
    c8, c12 = all_["cost_sensitivity"]
    assert c8["trades"] == c12["trades"] == 2
    assert c8["net_bps_per_session"] == pytest.approx((20 - 8 - 5 - 8) / 4)
    assert c12["net_bps_per_session"] == c8["net_bps_per_session"] - 2
    confirmed = score_events(
        events=events,
        eligible_days=[d, date(2024, 6, 11)],
        cohort="high_rvol",
        confirmed=True,
        positions=2,
        costs_bps=(8,),
        bootstrap_replicates=10,
    )
    assert confirmed["cost_sensitivity"][0]["trades"] == 1
    assert confirmed["cost_sensitivity"][0]["net_bps_per_session"] == 3


def test_no_events_is_zero_not_profitable():
    score = score_events(
        events=[],
        eligible_days=[date(2024, 6, 10), date(2024, 6, 11)],
        cohort="high_rvol",
        confirmed=False,
        positions=3,
        costs_bps=(12,),
        bootstrap_replicates=10,
    )
    assert score["cost_sensitivity"][0]["net_bps_per_session"] == 0


def test_event_not_filtered_by_later_session_length():
    day = date(2024, 6, 10)
    asset = _session(day, "AAA", breakout=True)
    short = Session(day, asset.bars[:80])
    candidate = Candidate("AAA", "high_rvol", 2, 100_000_000, 100.1, 99.9)
    event = first_breakout_event(
        candidate=candidate,
        asset=short,
        factor=_session(day, "QQQ", breakout=False, qqq=True),
        holding_minutes=30,
    )
    # A future halt after the trade would have closed must not invalidate it.
    assert event is not None
