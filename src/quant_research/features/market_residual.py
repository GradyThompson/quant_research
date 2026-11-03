"""Point-in-time stock-versus-factor residual signal from published minute closes.

This is a pure feature, not a strategy. It does not read a market-data provider,
learn from future outcomes, or treat a stale/missing synchronized quote as zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import isfinite, log, sqrt
from typing import Protocol, Sequence

MINUTE = timedelta(minutes=1)


class PublishedClose(Protocol):
    """Structural contract satisfied by the existing MinuteBar type."""

    @property
    def timestamp(self) -> datetime: ...

    @property
    def available_at(self) -> datetime: ...

    @property
    def close(self) -> float: ...


@dataclass(frozen=True, slots=True)
class ResidualSignal:
    as_of: datetime
    asset_return: float
    factor_return: float
    beta: float
    residual_return: float
    residual_volatility: float
    z_score: float
    reversal_score: float
    beta_observations: int


def _is_utc(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() == UTC.utcoffset(value)


def _published(
    bars: Sequence[PublishedClose], *, decision_at: datetime
) -> dict[datetime, PublishedClose]:
    result: dict[datetime, PublishedClose] = {}
    previous: datetime | None = None
    for bar in bars:
        if not _is_utc(bar.timestamp) or not _is_utc(bar.available_at):
            raise ValueError("price timestamps must be UTC-aware")
        if bar.available_at < bar.timestamp:
            raise ValueError("price available before observation")
        if not isfinite(bar.close) or bar.close <= 0:
            raise ValueError("close must be positive and finite")
        if previous is not None and bar.timestamp <= previous:
            raise ValueError("price bars must be strictly chronological")
        previous = bar.timestamp
        if bar.timestamp <= decision_at and bar.available_at <= decision_at:
            result[bar.timestamp] = bar
    return result


def compute_market_residual(
    asset_bars: Sequence[PublishedClose],
    factor_bars: Sequence[PublishedClose],
    *,
    decision_at: datetime,
    return_minutes: int = 5,
    beta_lookback: int = 120,
    min_beta_observations: int = 60,
    max_staleness: timedelta = timedelta(seconds=90),
) -> ResidualSignal | None:
    """Compute a historical-beta residual with information published by decision_at.

    Beta and residual scale use paired adjacent-minute returns *strictly before*
    the signal window. The signal uses consecutive synchronized minute closes.
    Historical beta samples may span sessions, but overnight returns are skipped.
    Insufficient, stale, or degenerate data returns None (no signal).
    """
    if not _is_utc(decision_at):
        raise ValueError("decision_at must be UTC-aware")
    if type(return_minutes) is not int or return_minutes < 1:
        raise ValueError("return_minutes must be positive")
    if type(beta_lookback) is not int or beta_lookback < 2:
        raise ValueError("beta_lookback must be >= 2")
    if type(min_beta_observations) is not int or not 2 <= min_beta_observations <= beta_lookback:
        raise ValueError("min_beta_observations must be in [2, beta_lookback]")
    if max_staleness < timedelta(0):
        raise ValueError("max_staleness must be nonnegative")

    assets = _published(asset_bars, decision_at=decision_at)
    factors = _published(factor_bars, decision_at=decision_at)
    # Reject lagging feeds rather than silently using the last older shared bar.
    if not assets or not factors or max(assets) != max(factors):
        return None
    common = sorted(assets.keys() & factors.keys())
    if len(common) <= return_minutes:
        return None
    end = common[-1]
    if decision_at - end > max_staleness:
        return None

    # The signal window must be complete, and must not cross an overnight gap.
    start = end - return_minutes * MINUTE
    if any(
        start + i * MINUTE not in assets or start + i * MINUTE not in factors
        for i in range(return_minutes + 1)
    ):
        return None
    history: list[tuple[float, float]] = []
    previous: datetime | None = None
    for stamp in common:
        if stamp > start:
            break
        if previous is not None and stamp - previous == MINUTE:
            asset_ret = log(assets[stamp].close / assets[previous].close)
            factor_ret = log(factors[stamp].close / factors[previous].close)
            history.append((asset_ret, factor_ret))
        previous = stamp
    history = history[-beta_lookback:]
    if len(history) < min_beta_observations:
        return None

    count = len(history)
    avg_factor = sum(item[1] for item in history) / count
    avg_asset = sum(item[0] for item in history) / count
    factor_ss = sum((item[1] - avg_factor) ** 2 for item in history)
    if factor_ss <= 1e-18:
        return None
    beta = (
        sum(
            (asset_ret - avg_asset) * (factor_ret - avg_factor) for asset_ret, factor_ret in history
        )
        / factor_ss
    )
    residuals = [asset_ret - beta * factor_ret for asset_ret, factor_ret in history]
    average_residual = sum(residuals) / count
    residual_variance = sum((value - average_residual) ** 2 for value in residuals) / count
    scale = sqrt(residual_variance * return_minutes)
    if scale <= 1e-12 or not isfinite(scale):
        return None

    asset_return = log(assets[end].close / assets[start].close)
    factor_return = log(factors[end].close / factors[start].close)
    residual_return = asset_return - beta * factor_return
    z_score = residual_return / scale
    if not isfinite(z_score):
        return None
    return ResidualSignal(
        as_of=decision_at,
        asset_return=asset_return,
        factor_return=factor_return,
        beta=beta,
        residual_return=residual_return,
        residual_volatility=scale,
        z_score=z_score,
        reversal_score=-z_score,
        beta_observations=count,
    )
