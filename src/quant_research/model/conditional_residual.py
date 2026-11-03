"""P5-A: point-in-time conditional ETF-proxy residual shock diagnostics.

Research only: historical ticker/instrument identity and split adjustments are
not available from market-wide minute summaries. Do not trade these results.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from math import isfinite, log, sqrt

import numpy as np

BPS = 10_000.0


@dataclass(frozen=True, slots=True)
class DailyPrice:
    """Daily prices extracted from a verified minute-file summary."""

    opening: float
    closing: float
    dollar_volume: float
    close_eligible: bool = True


@dataclass(frozen=True, slots=True)
class ResidualEvent:
    """An entry determined at close of formation_day; held open-to-open."""

    formation_day: date
    ticker: str
    factor: str
    beta: float
    z_score: float
    regime_efficiency: float
    holding_sessions: int
    entry_day: date
    exit_day: date
    stock_return_bps: float
    factor_return_bps: float

    @property
    def long_residual_bps(self) -> float:
        return self.stock_return_bps - self.beta * self.factor_return_bps


def _safe_close_return(previous: DailyPrice | None, current: DailyPrice | None) -> float | None:
    if (
        previous is None
        or current is None
        or not previous.close_eligible
        or not current.close_eligible
    ):
        return None
    if min(previous.closing, current.closing) <= 0:
        return None
    value = log(current.closing / previous.closing)
    # Large discontinuities can indicate splits, ticker reuse or bad bars.
    if not isfinite(value) or abs(value) >= 0.30:
        return None
    return value


def _open_return(start: DailyPrice | None, end: DailyPrice | None) -> float | None:
    if start is None or end is None or min(start.opening, end.opening) <= 0:
        return None
    value = log(end.opening / start.opening)
    if not isfinite(value) or abs(value) >= 0.40:
        return None
    return value


def generate_residual_events(
    *,
    days: Sequence[date],
    panel: dict[date, dict[str, DailyPrice]],
    factor_tickers: Sequence[str],
    validation_start: date,
    formation_sessions: int = 5,
    estimation_sessions: int = 60,
    holding_sessions: tuple[int, ...] = (3, 5, 10),
    min_price: float = 20.0,
    min_dollar_volume: float = 50_000_000.0,
    liquidity_sessions: int = 20,
    max_universe: int = 300,
    shock_z: float = 2.0,
    max_per_side: int = 3,
    progress: Callable[[date, int, int], None] | None = None,
) -> tuple[list[ResidualEvent], dict[str, int]]:
    """Create OOS event proxies using only past data to fit shock scores.

    ETF factor choice, beta, and residual standard deviation are fitted on
    observations strictly BEFORE the formation window. Entry uses the next
    session's opening bar and exit uses the open after `holding_sessions`.
    Daily liquidity selection uses only earlier sessions. Missing data never
    gets imputed. Event selection never inspects realized returns.
    """
    if (
        len(set(days)) != len(days)
        or list(days) != sorted(days)
        or not factor_tickers
        or len(set(factor_tickers)) != len(factor_tickers)
        or formation_sessions < 1
        or estimation_sessions < 15
        or liquidity_sessions < 2
        or max_universe < 1
        or max_per_side < 1
        or not holding_sessions
        or len(set(holding_sessions)) != len(holding_sessions)
        or any(h < 1 for h in holding_sessions)
        or min_price <= 0
        or min_dollar_volume <= 0
        or shock_z <= 0
    ):
        raise ValueError("invalid conditional residual experiment settings")
    counts: defaultdict[str, int] = defaultdict(int)
    events: list[ResidualEvent] = []
    start_index = max(estimation_sessions + formation_sessions, liquidity_sessions)
    for i in range(start_index, len(days)):
        day = days[i]
        if progress is not None and i % 20 == 0:
            progress(day, i, len(days))
        if day < validation_start:
            continue
        present = panel.get(day, {})
        if "SPY" not in present:
            counts["missing_spy"] += 1
            continue
        # Market-efficiency ratio uses daily returns through today's close.
        spy_changes = [
            _safe_close_return(
                panel.get(days[j - 1], {}).get("SPY"), panel.get(days[j], {}).get("SPY")
            )
            for j in range(i - estimation_sessions + 1, i + 1)
        ]
        if any(value is None for value in spy_changes):
            counts["missing_market_regime"] += 1
            continue
        spy_values = np.asarray(spy_changes, dtype=np.float64)
        total_path = float(np.abs(spy_values).sum())
        efficiency = abs(float(spy_values.sum())) / total_path if total_path else 0.0
        # Pre-rank by *previous* dollar volume, never today's dollar volume.
        eligible: list[tuple[float, str]] = []
        excluded = set(factor_tickers) | {"SPY"}
        for ticker, current in present.items():
            if ticker in excluded or current.opening < min_price:
                continue
            history = [panel.get(days[j], {}).get(ticker) for j in range(i - liquidity_sessions, i)]
            if any(row is None or not row.close_eligible for row in history):
                continue
            trailing_dv = float(sum(row.dollar_volume for row in history if row is not None))
            trailing_dv /= liquidity_sessions
            if trailing_dv >= min_dollar_volume:
                eligible.append((trailing_dv, ticker))
        eligible.sort(key=lambda item: (-item[0], item[1]))
        candidates = [ticker for _, ticker in eligible[:max_universe]]
        counts["liquid_candidates"] += len(candidates)
        # Precompute candidate factors once per date rather than once per
        # stock; this dominates full-market scans when used without caching.
        factor_windows: dict[str, np.ndarray] = {}
        for factor in factor_tickers:
            history = [
                panel.get(days[j], {}).get(factor)
                for j in range(i - estimation_sessions - formation_sessions, i + 1)
            ]
            changes = [_safe_close_return(a, b) for a, b in zip(history, history[1:], strict=False)]
            if all(value is not None for value in changes):
                factor_windows[factor] = np.asarray(changes, dtype=np.float64)
        # The same decided-on-close candidates are evaluated for each horizon.
        signals: dict[str, tuple[float, str, float]] = {}
        for ticker in candidates:
            prices = [
                panel.get(days[j], {}).get(ticker)
                for j in range(i - estimation_sessions - formation_sessions, i + 1)
            ]
            y_values = [_safe_close_return(a, b) for a, b in zip(prices, prices[1:], strict=False)]
            if any(value is None for value in y_values):
                counts["invalid_stock_prices_or_split_risk"] += 1
                continue
            y = np.asarray(y_values, dtype=np.float64)
            training_y, recent_y = y[:-formation_sessions], y[-formation_sessions:]
            best: tuple[float, str, float, float, float] | None = None
            for factor, x in factor_windows.items():
                train_x = x[:-formation_sessions]
                xc = train_x - train_x.mean()
                yc = training_y - training_y.mean()
                sx = float(xc @ xc)
                sy = float(yc @ yc)
                if sx < 1e-10 or sy < 1e-10:
                    continue
                beta = float(xc @ yc / sx)
                if not 0.0 <= beta <= 3.0:
                    continue
                corr_squared = float((xc @ yc) ** 2 / (sx * sy))
                alpha = float(training_y.mean() - beta * train_x.mean())
                residual = training_y - alpha - beta * train_x
                residual_std = float(residual.std(ddof=1))
                if residual_std <= 1e-9:
                    continue
                result = (corr_squared, factor, beta, alpha, residual_std)
                if best is None or result[0] > best[0]:
                    best = result
            if best is None:
                counts["no_valid_factor_fit"] += 1
                continue
            _, factor, beta, alpha, std = best
            future_x = factor_windows[factor][-formation_sessions:]
            score = float((recent_y - alpha - beta * future_x).sum())
            score /= sqrt(formation_sessions) * std
            if isfinite(score) and abs(score) >= shock_z:
                signals[ticker] = (score, factor, beta)
        counts["shock_candidates"] += len(signals)
        # Select on z-score ONLY; realized future price is never used to rank.
        negative = sorted(
            (ticker for ticker in signals if signals[ticker][0] < 0),
            key=lambda ticker: (signals[ticker][0], ticker),
        )[:max_per_side]
        positive = sorted(
            (ticker for ticker in signals if signals[ticker][0] > 0),
            key=lambda ticker: (-signals[ticker][0], ticker),
        )[:max_per_side]
        for ticker in negative + positive:
            z, factor, beta = signals[ticker]
            for horizon in holding_sessions:
                if i + 1 + horizon >= len(days):
                    counts["end_of_sample_without_exit"] += 1
                    continue
                entry_day, exit_day = days[i + 1], days[i + horizon + 1]
                # A protected/missing session inside a future label must not
                # be implicitly crossed by an endpoint-to-endpoint return.
                intermediate = [panel.get(days[j], {}) for j in range(i + 1, i + horizon + 2)]
                if any(
                    "SPY" not in day_rows or ticker not in day_rows or factor not in day_rows
                    for day_rows in intermediate
                ):
                    counts["missing_or_discontinuous_exit"] += 1
                    continue
                if any(
                    _safe_close_return(panel[days[j - 1]].get(ticker), panel[days[j]].get(ticker))
                    is None
                    for j in range(i + 2, i + horizon + 1)
                ):
                    counts["missing_or_discontinuous_exit"] += 1
                    continue
                asset_entry = panel.get(entry_day, {}).get(ticker)
                asset_exit = panel.get(exit_day, {}).get(ticker)
                factor_entry = panel.get(entry_day, {}).get(factor)
                factor_exit = panel.get(exit_day, {}).get(factor)
                stock_ret = _open_return(asset_entry, asset_exit)
                factor_ret = _open_return(factor_entry, factor_exit)
                # Diagnostic only: absence of an exit is a potential delisting
                # censoring problem and must be disclosed, not silently zeroed.
                if stock_ret is None or factor_ret is None:
                    counts["missing_or_discontinuous_exit"] += 1
                    continue
                events.append(
                    ResidualEvent(
                        formation_day=day,
                        ticker=ticker,
                        factor=factor,
                        beta=beta,
                        z_score=z,
                        regime_efficiency=efficiency,
                        holding_sessions=horizon,
                        entry_day=entry_day,
                        exit_day=exit_day,
                        stock_return_bps=stock_ret * BPS,
                        factor_return_bps=factor_ret * BPS,
                    )
                )
    return events, dict(counts)


def _moving_block_interval(values: list[float], *, repeats: int, block: int = 10) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1 or repeats < 2:
        return [float(np.mean(values)), float(np.mean(values))]
    rng = np.random.default_rng(20261101)
    data = np.asarray(values)
    n = len(data)
    width = min(block, n)
    starts = rng.integers(0, n - width + 1, size=(repeats, (n + width - 1) // width))
    offsets = np.arange(width)
    indexes = (starts[:, :, None] + offsets[None, None, :]).reshape(repeats, -1)[:, :n]
    samples = data[indexes].mean(axis=1)
    return [float(v) for v in np.quantile(samples, [0.025, 0.975])]


def score_residual_events(
    *,
    events: Sequence[ResidualEvent],
    eligible_days: Sequence[date],
    horizon: int,
    side: str,
    regime: str,
    roundtrip_cost_bps: float,
    bootstrap_repeats: int = 1000,
    slots: int = 3,
) -> dict[str, object]:
    """Per-entry-day slot return, NOT overlapping-position portfolio P&L."""
    if (
        not eligible_days
        or list(eligible_days) != sorted(set(eligible_days))
        or side not in ("negative_long", "positive_short")
        or regime not in ("all", "range", "trend")
        or horizon < 1
        or slots < 1
        or roundtrip_cost_bps < 0
        or not isfinite(roundtrip_cost_bps)
    ):
        raise ValueError("invalid residual score configuration")
    day_return: dict[date, float] = {day: 0.0 for day in eligible_days}
    chosen: list[dict[str, object]] = []
    net_values: list[float] = []
    for event in events:
        if event.holding_sessions != horizon:
            continue
        if (side == "negative_long") != (event.z_score < 0):
            continue
        if regime == "range" and event.regime_efficiency > 0.25:
            continue
        if regime == "trend" and event.regime_efficiency <= 0.25:
            continue
        direction = 1 if side == "negative_long" else -1
        # Hedge ETF notional is beta times the stock position. Both legs pay
        # the stated roundtrip cost assumption, in stock-notional basis points.
        gross = direction * event.long_residual_bps
        total_cost = roundtrip_cost_bps * (1.0 + abs(event.beta))
        net = gross - total_cost
        net_values.append(net)
        # Three independent daily signal slots; overlapping holdings between
        # sessions are NOT netted into portfolio capital (reported explicitly).
        if event.formation_day not in day_return:
            raise ValueError("event formation day is outside eligible days")
        day_return[event.formation_day] += net / slots
        chosen.append(
            {
                "formation_day": event.formation_day.isoformat(),
                "entry_day": event.entry_day.isoformat(),
                "exit_day": event.exit_day.isoformat(),
                "ticker": event.ticker,
                "factor": event.factor,
                "z_score": event.z_score,
                "beta": event.beta,
                "gross_factor_hedged_bps": gross,
                "net_factor_hedged_bps": net,
                "unhedged_stock_bps": direction * event.stock_return_bps - roundtrip_cost_bps,
            }
        )
    daily = [day_return[day] for day in eligible_days]
    return {
        "trades": len(chosen),
        "active_entry_sessions": sum(v != 0 for v in daily),
        "net_bps_per_entry_session": float(np.mean(daily)),
        "net_bps_per_trade": float(np.mean(net_values)) if net_values else 0.0,
        "block_bootstrap_95_interval": _moving_block_interval(
            daily,
            repeats=bootstrap_repeats,
        ),
        "by_year": {
            str(year): {
                "entry_sessions": len([d for d in eligible_days if d.year == year]),
                "net_bps_per_entry_session": float(
                    np.mean([day_return[d] for d in eligible_days if d.year == year])
                ),
            }
            for year in sorted({d.year for d in eligible_days})
        },
        "trades_detail": chosen,
        "NOT_PORTFOLIO_PNL": True,
    }
