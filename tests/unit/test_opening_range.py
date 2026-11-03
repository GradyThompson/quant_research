"""Pure synthetic opening-range tests: no market-data network or broker access."""

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from math import isclose
from zoneinfo import ZoneInfo

import pytest

from quant_research.model.first_passage_validation import MinuteBar, Session
from quant_research.model.opening_range import (
    _entry_window,
    build_opening_range_trades,
    score_breakout_trades,
)


def _sample(
    count: int = 17,
    *,
    negative_qqq: bool = False,
    lag_seconds: int = 60,
) -> tuple[list[Session], list[Session]]:
    assets, factors = [], []
    eastern = ZoneInfo("America/New_York")
    for d in range(count):
        day = date(2025, 1, 13) + timedelta(days=d)
        first = datetime(day.year, day.month, day.day, 9, 31, tzinfo=eastern).astimezone(UTC)
        a_bars, f_bars = [], []
        for i in range(150):
            stamp = first + timedelta(minutes=i)
            a_close = 100.0 if i < 18 else 100.30 + 0.04 * (i - 18)
            f_close = 100.0 * (1 + (-1 if negative_qqq else 1) * i * 0.00005)
            # current session opening volume spikes on last sample; earlier
            # sessions provide 10-session baseline without future-day leakage.
            volume = 500.0 if d < 11 else 1800.0
            a_bars.append(
                MinuteBar(
                    stamp,
                    stamp + timedelta(seconds=lag_seconds),
                    a_close,
                    a_close + 0.06,
                    a_close - 0.06,
                    a_close,
                    volume,
                )
            )
            f_bars.append(
                MinuteBar(
                    stamp,
                    stamp + timedelta(seconds=lag_seconds),
                    f_close,
                    f_close + 0.05,
                    f_close - 0.05,
                    f_close,
                    100.0,
                )
            )
        assets.append(Session(day, tuple(a_bars)))
        factors.append(Session(day, tuple(f_bars)))
    return assets, factors


def test_breakout_market_and_past_volume_variants() -> None:
    assets, factors = _sample()
    diagnostics: dict[str, int] = {}
    result = build_opening_range_trades(
        asset_sessions=assets,
        factor_sessions=factors,
        horizon_minutes=30,
        diagnostics=diagnostics,
    )
    assert len(result["breakout"]) == 17
    assert len(result["market_confirmed"]) == 17
    assert {trade.day for trade in result["market_volume_confirmed"]} == {
        s.day for s in assets[11:]
    }
    assert diagnostics["high_volume_sessions"] == 6
    for group in result.values():
        for trade in group:
            assert trade.side == 1
            assert trade.decision_at < trade.entry_at < trade.exit_available_at
            assert trade.gross_bps > 0
    scores = score_breakout_trades(
        result["market_volume_confirmed"],
        eligible_days=[s.day for s in assets[-6:]],
        roundtrip_cost_bps=8,
        bootstrap_replicates=10,
    )
    assert scores["trades"] == 6
    assert scores["net_bps_per_session"] > 0
    assert scores["proxy_not_executable"] is True


def test_no_future_volume_or_unpublished_factor_information() -> None:
    assets, factors = _sample(12)
    before = build_opening_range_trades(
        asset_sessions=assets,
        factor_sessions=factors,
        horizon_minutes=30,
    )
    assert before["market_volume_confirmed"]
    # Alter late-day volume after the breakout; opening volume and signal stay fixed.
    last = list(assets[-1].bars)
    last[100] = replace(last[100], volume=9999999.0)
    assets[-1] = Session(assets[-1].day, tuple(last))
    after = build_opening_range_trades(
        asset_sessions=assets,
        factor_sessions=factors,
        horizon_minutes=30,
    )
    assert before["market_volume_confirmed"] == after["market_volume_confirmed"]
    # If QQQ never confirms breakout direction, market variants abstain.
    negative_factor = _sample(12, negative_qqq=True)[1]
    no_confirm = build_opening_range_trades(
        asset_sessions=assets,
        factor_sessions=negative_factor,
        horizon_minutes=30,
    )
    assert no_confirm["breakout"]
    assert not no_confirm["market_confirmed"]
    assert not no_confirm["market_volume_confirmed"]


def test_lag_entry_avoids_same_instant_zero_latency_fill() -> None:
    assets, factors = _sample(11)
    result = build_opening_range_trades(
        asset_sessions=assets,
        factor_sessions=factors,
        horizon_minutes=60,
    )
    first = result["breakout"][0]
    assert first.entry_at > first.decision_at
    day = assets[0]
    bar = next(b for b in day.bars if b.timestamp - timedelta(minutes=1) == first.entry_at)
    assert first.entry_price == bar.open
    assert first.exit_available_at > first.entry_at
    assert len(result["market_volume_confirmed"]) == 0  # Not enough history/volume spikes


def test_reject_gapped_or_late_start_and_wrong_session_or_cost() -> None:
    assets, factors = _sample(11)
    original = list(assets[0].bars)
    original = [
        replace(
            bar,
            timestamp=bar.timestamp + timedelta(minutes=3),
            available_at=bar.available_at + timedelta(minutes=3),
        )
        for bar in original
    ]
    assets[0] = Session(assets[0].day, tuple(original))
    original = list(assets[1].bars)
    original[6] = replace(original[6], timestamp=original[6].timestamp + timedelta(seconds=5))
    assets[1] = Session(assets[1].day, tuple(original))
    diag: dict[str, int] = {}
    trades = build_opening_range_trades(
        asset_sessions=assets, factor_sessions=factors, horizon_minutes=30, diagnostics=diag
    )
    assert diag["missing_or_misaligned_opening_range"] == 2
    assert all(t.day not in (assets[0].day, assets[1].day) for t in trades["breakout"])
    with pytest.raises(ValueError, match="invalid cost"):
        score_breakout_trades([], eligible_days=[assets[0].day], roundtrip_cost_bps=-1)
    with pytest.raises(ValueError, match="outside"):
        score_breakout_trades(
            trades["breakout"], eligible_days=[assets[0].day], roundtrip_cost_bps=4
        )


def test_entry_window_rejects_incomplete_or_noncontiguous_minutes() -> None:
    assets, _ = _sample(1)
    bars = list(assets[0].bars)
    assert _entry_window(bars, decision=bars[-3].available_at, horizon=30) is None
    entry = _entry_window(bars, decision=bars[30].available_at, horizon=30)
    assert entry is not None
    broken = bars.copy()
    broken[36] = replace(broken[36], timestamp=broken[36].timestamp + timedelta(seconds=2))
    assert _entry_window(broken, decision=bars[30].available_at, horizon=30) is None


def test_cash_days_included_in_score_and_cost_sensitivity() -> None:
    assets, factors = _sample(12)
    one_trade = build_opening_range_trades(
        asset_sessions=assets,
        factor_sessions=factors,
        horizon_minutes=30,
    )["market_volume_confirmed"]
    assert len(one_trade) == 1
    days = [asset.day for asset in assets]
    low = score_breakout_trades(
        one_trade, eligible_days=days, roundtrip_cost_bps=4, bootstrap_replicates=10
    )
    high = score_breakout_trades(
        one_trade, eligible_days=days, roundtrip_cost_bps=12, bootstrap_replicates=10
    )
    assert low["scored_sessions"] == 12
    assert low["trades"] == 1
    assert isclose(low["net_bps_per_session"] - high["net_bps_per_session"], 8 / 12)
    with pytest.raises(ValueError, match="more than one"):
        score_breakout_trades(one_trade * 2, eligible_days=days, roundtrip_cost_bps=8)
