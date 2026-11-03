from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import exp, isclose

import pytest

from quant_research.features.market_residual import compute_market_residual

T0 = datetime(2026, 10, 5, 13, 30, tzinfo=UTC)


@dataclass(frozen=True)
class Close:
    timestamp: datetime
    available_at: datetime
    close: float


def _prices(
    *,
    steps: int = 180,
    offset: int = 0,
    delay: timedelta = timedelta(0),
) -> tuple[list[Close], list[Close]]:
    stock = 100.0
    factor = 100.0
    assets: list[Close] = []
    factors: list[Close] = []
    for idx in range(steps):
        stamp = T0 + timedelta(minutes=idx + offset)
        if idx:
            factor *= exp((0.001 if idx % 2 else -0.001) + 0.0002)
            stock *= exp(
                (0.0018 if idx % 2 else -0.0018) + 0.0001 + (0.0004 if idx % 3 else -0.0008)
            )
        assets.append(Close(stamp, stamp + delay, stock))
        factors.append(Close(stamp, stamp + delay, factor))
    return assets, factors


def test_residual_is_past_only_and_beta_is_excluded_from_signal_window():
    assets, factors = _prices()
    at = assets[-1].timestamp
    signal = compute_market_residual(assets, factors, decision_at=at, beta_lookback=90)
    assert signal is not None
    assert signal.beta_observations == 90
    assert 1.3 < signal.beta < 2.4
    assert isclose(signal.reversal_score, -signal.z_score)
    assert signal.as_of == at
    # A future price cannot affect the forecast made at `at`.
    future = Close(at + timedelta(minutes=1), at + timedelta(minutes=1), 100_000.0)
    other = compute_market_residual(assets + [future], factors, decision_at=at, beta_lookback=90)
    assert signal == other


def test_not_yet_published_asset_or_factor_fails_closed():
    assets, factors = _prices()
    at = assets[-1].timestamp
    delayed = [
        *factors[:-1],
        Close(
            factors[-1].timestamp, factors[-1].timestamp + timedelta(minutes=2), factors[-1].close
        ),
    ]
    assert compute_market_residual(assets, delayed, decision_at=at) is None


def test_fails_closed_on_staleness_missing_signal_window_and_insufficient_history():
    assets, factors = _prices()
    at = assets[-1].timestamp
    assert compute_market_residual(assets, factors, decision_at=at + timedelta(minutes=3)) is None
    assert (
        compute_market_residual(assets[:30], factors[:30], decision_at=assets[29].timestamp) is None
    )
    assert compute_market_residual(assets, factors[:-3] + factors[-2:], decision_at=at) is None


def test_skips_overnight_return_for_beta_but_uses_older_sessions():
    old_asset, old_factor = _prices(steps=100)
    new_start = 390
    new_asset, new_factor = _prices(steps=30, offset=new_start)
    assets = old_asset + new_asset
    factors = old_factor + new_factor
    signal = compute_market_residual(
        assets, factors, decision_at=assets[-1].timestamp, min_beta_observations=60
    )
    assert signal is not None
    assert signal.beta_observations >= 60
    # A discontinuous signal window must not cross an overnight gap.
    assert (
        compute_market_residual(
            assets[:100] + new_asset[:3],
            factors[:100] + new_factor[:3],
            decision_at=new_asset[2].timestamp,
        )
        is None
    )


def test_malformed_and_unordered_bars_rejected():
    assets, factors = _prices()
    at = assets[-1].timestamp
    with pytest.raises(ValueError, match="chronological"):
        compute_market_residual(assets[::-1], factors, decision_at=at)
    with pytest.raises(ValueError, match="positive"):
        compute_market_residual([*assets[:-1], Close(at, at, -1)], factors, decision_at=at)
    with pytest.raises(ValueError, match="UTC"):
        compute_market_residual(assets, factors, decision_at=at.replace(tzinfo=None))
    with pytest.raises(ValueError, match="min_beta"):
        compute_market_residual(assets, factors, decision_at=at, min_beta_observations=999)
    with pytest.raises(ValueError, match="available"):
        compute_market_residual(
            [Close(at, at - timedelta(seconds=1), 100)], factors, decision_at=at
        )


def test_flat_factor_fails_closed():
    assets, factors = _prices()
    factors = [Close(bar.timestamp, bar.available_at, 100.0) for bar in factors]
    assert compute_market_residual(assets, factors, decision_at=assets[-1].timestamp) is None
