"""Timing, economics and exclusion tests for P4.6 ETF late-session research."""

from datetime import UTC, date, datetime, timedelta
from math import log

import pytest

from quant_research.model.first_passage_validation import MinuteBar, Session
from quant_research.model.late_session_momentum import (
    build_late_session_event,
    position,
    score_late_session,
)


def _session(
    day: date,
    *,
    lag_minutes: int = 1,
    skip_index: int | None = None,
    last_price: float = 101.0,
) -> tuple[Session, datetime, datetime]:
    opening = datetime(2024, 3, 11, 13, 30, tzinfo=UTC)
    close = opening + timedelta(minutes=390)
    bars = []
    for i in range(390):
        if i == skip_index:
            continue
        stamp = opening + timedelta(minutes=i + 1)
        price = 100.0 if i < 30 else 100.5
        if i == 29:
            price = 100.6
        if i == 361:
            price = 100.7
        if i == 389:
            price = last_price
        bars.append(
            MinuteBar(
                timestamp=stamp,
                available_at=stamp + timedelta(minutes=lag_minutes),
                open=price,
                high=price + 0.1,
                low=price - 0.1,
                close=price,
                volume=1000.0,
            )
        )
    return Session(day, tuple(bars)), opening, close


def test_late_entry_uses_open_not_close_auction() -> None:
    day = date(2024, 3, 11)
    session, opening, closing = _session(day)
    result = build_late_session_event(
        ticker="SPY",
        session=session,
        previous_close=99.0,
        opening=opening,
        closing=closing,
    )
    assert result is not None
    assert result.entry_at == closing - timedelta(minutes=29)
    assert result.exit_at == closing - timedelta(minutes=1)
    assert result.first_half_hour_available_at == opening + timedelta(minutes=31)
    assert result.early_prevclose_bps > result.early_open_bps > 0
    assert result.forward_bps == pytest.approx(log(101.0 / 100.7) * 10_000)
    assert position(result, "prev_close_momentum") == 1
    assert position(result, "open_momentum") == 1
    assert position(result, "always_long") == 1


@pytest.mark.parametrize("missing", [0, 29, 361, 389])
def test_missing_required_minute_excludes_event(missing: int) -> None:
    session, opening, closing = _session(date(2024, 3, 11), skip_index=missing)
    assert (
        build_late_session_event(
            ticker="QQQ",
            session=session,
            previous_close=99.0,
            opening=opening,
            closing=closing,
        )
        is None
    )


def test_delayed_unpublished_feature_is_not_allowed() -> None:
    session, opening, closing = _session(date(2024, 3, 11), lag_minutes=400)
    assert (
        build_late_session_event(
            ticker="QQQ",
            session=session,
            previous_close=99.0,
            opening=opening,
            closing=closing,
        )
        is None
    )


def test_early_close_is_excluded() -> None:
    session, opening, closing = _session(date(2024, 3, 11))
    assert (
        build_late_session_event(
            ticker="SPY",
            session=session,
            previous_close=99.0,
            opening=opening,
            closing=closing - timedelta(minutes=180),
        )
        is None
    )


def test_invalid_or_duplicate_bars_rejected() -> None:
    session, opening, closing = _session(date(2024, 3, 11))
    repeated = Session(session.day, session.bars + session.bars[-1:])
    with pytest.raises(ValueError, match="duplicate"):
        build_late_session_event(
            ticker="SPY",
            session=repeated,
            previous_close=99.0,
            opening=opening,
            closing=closing,
        )


def test_costs_keep_trades_fixed_and_include_cash_sessions() -> None:
    day = date(2024, 3, 11)
    session, opening, closing = _session(day)
    event = build_late_session_event(
        ticker="SPY",
        session=session,
        previous_close=99.0,
        opening=opening,
        closing=closing,
    )
    assert event is not None
    other = date(2024, 3, 12)
    result = score_late_session(
        events=[event],
        eligible_days=[day, other],
        signal="prev_close_momentum",
        costs_bps=(4.0, 12.0),
        replications=50,
    )
    cheaper, expensive = result["cost_sensitivity"]
    assert cheaper["trades"] == expensive["trades"] == 1
    assert cheaper["net_bps_per_session"] - expensive["net_bps_per_session"] == pytest.approx(4)
    assert result["long_trades"] == 1
    assert len(cheaper["session_bootstrap_95ci"]) == 2
    assert cheaper["by_month_net_bps_per_session"]["2024-03"] == pytest.approx(
        cheaper["net_bps_per_session"]
    )


def test_score_rejects_duplicate_events() -> None:
    session, opening, closing = _session(date(2024, 3, 11))
    event = build_late_session_event(
        ticker="SPY",
        session=session,
        previous_close=99.0,
        opening=opening,
        closing=closing,
    )
    assert event is not None
    with pytest.raises(ValueError, match="multiple events"):
        score_late_session(
            events=[event, event],
            eligible_days=[event.day],
            signal="always_long",
            costs_bps=(4.0,),
        )


def test_short_direction_and_paired_always_long_comparison() -> None:
    from dataclasses import replace

    session, opening, closing = _session(date(2024, 3, 11))
    base = build_late_session_event(
        ticker="SPY",
        session=session,
        previous_close=99.0,
        opening=opening,
        closing=closing,
    )
    assert base is not None
    event = replace(base, early_prevclose_bps=-15.0, early_open_bps=-20.0)
    assert position(event, "prev_close_momentum") == -1
    mom = score_late_session(
        events=[event],
        eligible_days=[event.day],
        signal="prev_close_momentum",
        costs_bps=(4.0,),
        replications=5,
    )
    benchmark = score_late_session(
        events=[event],
        eligible_days=[event.day],
        signal="always_long",
        costs_bps=(4.0,),
        replications=5,
    )
    momentum_net = mom["cost_sensitivity"][0]
    long_net = benchmark["cost_sensitivity"][0]
    assert mom["short_trades"] == 1
    assert momentum_net["net_bps_vs_always_long"] == pytest.approx(
        momentum_net["net_bps_per_session"] - long_net["net_bps_per_session"]
    )
    assert long_net["net_bps_vs_always_long"] == pytest.approx(0)
