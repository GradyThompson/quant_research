"""Point-in-time cross-sectional abnormal-volume continuation research.

One common first-breakout event per security per day. Confirmation changes
which events are traded, not which event is detected. Price proxies only.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from math import isfinite, log
from statistics import mean
from typing import Literal, Sequence
from zoneinfo import ZoneInfo

from quant_research.data.ingestion.minute_market_summary import DailySummary
from quant_research.model.first_passage_diagnostics import session_block_interval
from quant_research.model.first_passage_validation import Session
from quant_research.model.opening_range import BPS, _entry_window, _factor_side

Cohort = Literal["high_rvol", "normal_rvol_control"]


@dataclass(frozen=True, slots=True)
class Candidate:
    ticker: str
    cohort: Cohort
    rvol: float
    historical_dollar_volume: float
    opening_high: float
    opening_low: float


@dataclass(frozen=True, slots=True)
class Event:
    day: date
    ticker: str
    cohort: Cohort
    rvol: float
    side: int
    decision_at: datetime
    entry_at: datetime
    exit_at: datetime
    gross_bps: float
    market_aligned: bool


class RollingUniverse:
    """Only previous *exchange sessions* may determine liquidity and RVOL.

    Candidate selection occurs before today's summary enters the state.
    Delisted names are not dropped by future membership, although instrument
    type and historical ticker identity remain unverified by flat files.
    """

    def __init__(
        self,
        *,
        lookback: int = 20,
        min_price: float = 10.0,
        min_dollar_volume: float = 20_000_000.0,
        min_rvol: float = 1.5,
        count: int = 20,
        factor_ticker: str = "QQQ",
    ) -> None:
        if lookback < 2 or count < 1 or min_price <= 0 or min_dollar_volume <= 0:
            raise ValueError("invalid universe selection configuration")
        if not isfinite(min_rvol) or min_rvol <= 0:
            raise ValueError("invalid relative-volume threshold")
        self.lookback = lookback
        self.min_price = min_price
        self.min_dollar_volume = min_dollar_volume
        self.min_rvol = min_rvol
        self.count = count
        self.factor_ticker = factor_ticker
        self.history: dict[str, deque[tuple[int, DailySummary]]] = {}
        self.index = 0

    def observe(self, today: dict[str, DailySummary]) -> list[Candidate]:
        high: list[Candidate] = []
        ordinary: list[Candidate] = []
        i = self.index
        for ticker, item in today.items():
            if ticker == self.factor_ticker:
                continue
            previous = self.history.get(ticker)
            if previous is None or len(previous) != self.lookback:
                continue
            if previous[0][0] != i - self.lookback or previous[-1][0] != i - 1:
                continue  # Missing trading day, ticker change, or protected gap.
            prior = [summary for _, summary in previous]
            baseline = mean(s.opening_volume for s in prior)
            if baseline <= 0 or item.opening_volume <= 0:
                continue
            average_dollars = mean(s.dollar_volume for s in prior)
            prev_close = prior[-1].last_close
            if (
                prev_close < self.min_price
                or item.opening_open < self.min_price
                or average_dollars < self.min_dollar_volume
            ):
                continue
            # Research safety: avoid obvious unadjusted split-like gaps.
            if abs(log(item.opening_open / prev_close)) > 0.3:
                continue
            rvol = item.opening_volume / baseline
            candidate = Candidate(
                ticker,
                "high_rvol",
                rvol,
                average_dollars,
                item.opening_high,
                item.opening_low,
            )
            if rvol >= self.min_rvol:
                high.append(candidate)
            elif 0.8 <= rvol <= 1.2:
                ordinary.append(candidate)
        high.sort(key=lambda c: (-c.rvol, c.ticker))
        selected: list[Candidate] = high[: self.count]
        # A liquidity-matched, ordinary-volume control; selection uses ONLY
        # historical dollar volume and the current published opening volume.
        controls: list[Candidate] = []
        for positive in selected:
            if not ordinary:
                break
            match_index = min(
                range(len(ordinary)),
                key=lambda j: (
                    abs(
                        log(
                            ordinary[j].historical_dollar_volume / positive.historical_dollar_volume
                        )
                    ),
                    ordinary[j].ticker,
                ),
            )
            c = ordinary.pop(match_index)
            controls.append(
                Candidate(
                    c.ticker,
                    "normal_rvol_control",
                    c.rvol,
                    c.historical_dollar_volume,
                    c.opening_high,
                    c.opening_low,
                )
            )
        # Apply update only AFTER constructing today's selection.
        for ticker in list(self.history):
            history = self.history[ticker]
            while history and history[0][0] <= i - self.lookback:
                history.popleft()
            if not history:
                del self.history[ticker]
        for ticker, summary in today.items():
            history = self.history.setdefault(ticker, deque())
            if summary.close_eligible:
                history.append((i, summary))
        self.index += 1
        return selected + controls


def first_breakout_event(
    *,
    candidate: Candidate,
    asset: Session,
    factor: Session,
    holding_minutes: int,
    opening_minutes: int = 15,
    scan_minutes: int = 120,
    breakout_buffer_bps: float = 5.0,
) -> Event | None:
    """Use the FIRST breakout for both confirmed and unconfirmed policies.

    The completed breakout bar must be published before the decision; a
    minute-open proxy strictly after that decision is used as entry.
    """
    if asset.day != factor.day or holding_minutes < 1:
        raise ValueError("asset/factor day or holding period invalid")
    bars = asset.bars
    if len(bars) < opening_minutes + 1 or opening_minutes < 1:
        return None
    if bars[0].timestamp.astimezone(ZoneInfo("America/New_York")).strftime("%H:%M") != "09:31":
        return None
    opening = bars[:opening_minutes]
    if any(
        b.timestamp - a.timestamp != timedelta(minutes=1)
        for a, b in zip(opening, opening[1:], strict=False)
    ):
        return None
    upper = candidate.opening_high * (1 + breakout_buffer_bps / BPS)
    lower = candidate.opening_low * (1 - breakout_buffer_bps / BPS)
    expected_start = bars[0].timestamp
    for bar in bars[opening_minutes:]:
        index = (bar.timestamp - expected_start) // timedelta(minutes=1)
        if index >= scan_minutes:
            break
        if bar.timestamp != expected_start + index * timedelta(minutes=1):
            continue
        side = 1 if bar.close > upper else (-1 if bar.close < lower else 0)
        if side == 0:
            continue
        decision = bar.available_at
        if any(prior.available_at > decision for prior in opening):
            return None
        future = _entry_window(bars, decision=decision, horizon=holding_minutes)
        if future is None:
            return None
        entry, exit_bar = future
        market_side = _factor_side(
            factor,
            decision=decision,
            max_staleness=timedelta(seconds=90),
        )
        return Event(
            day=asset.day,
            ticker=candidate.ticker,
            cohort=candidate.cohort,
            rvol=candidate.rvol,
            side=side,
            decision_at=decision,
            entry_at=entry.timestamp - timedelta(minutes=1),
            exit_at=exit_bar.available_at,
            gross_bps=side * log(exit_bar.close / entry.open) * BPS,
            market_aligned=market_side == side,
        )
    return None


def score_events(
    *,
    events: Sequence[Event],
    eligible_days: Sequence[date],
    cohort: Cohort,
    confirmed: bool,
    positions: int,
    costs_bps: tuple[float, ...],
    bootstrap_replicates: int = 2000,
) -> dict[str, object]:
    """Fixed capital slots, day-bootstrap uncertainty, fixed trades across costs.

    Independent one-trade-per-day capital slots are reserved; unused capacity
    earns zero. Not a broker-level execution or dynamic-capital simulation.
    """
    if positions < 1 or not eligible_days or len(set(eligible_days)) != len(eligible_days):
        raise ValueError("invalid eligible sessions/positions")
    if not costs_bps or any(not isfinite(c) or c < 0 for c in costs_bps):
        raise ValueError("invalid round-trip costs")
    by_day: dict[date, list[Event]] = {d: [] for d in eligible_days}
    for event in events:
        if event.day not in by_day:
            raise ValueError("event outside validation days")
        if event.cohort == cohort and (not confirmed or event.market_aligned):
            by_day[event.day].append(event)
    selected: list[Event] = []
    for day in eligible_days:
        picks = sorted(
            by_day[day],
            key=lambda event: (
                event.decision_at,
                -event.rvol,
                event.ticker,
            ),
        )[:positions]
        selected.extend(picks)
    scenarios: list[dict[str, object]] = []
    for cost in costs_bps:
        daily: dict[date, list[float]] = {day: [0.0] for day in eligible_days}
        for event in selected:
            daily[event.day][0] += (event.gross_bps - cost) / positions
        net = [event.gross_bps - cost for event in selected]
        scenarios.append(
            {
                "roundtrip_cost_bps": cost,
                "trades": len(selected),
                "active_sessions": sum(bool(by_day[d]) for d in eligible_days),
                "net_bps_per_trade": mean(net) if net else None,
                "net_bps_per_session": sum(v[0] for v in daily.values()) / len(daily),
                "net_bps_per_session_95pct_bootstrap": session_block_interval(
                    daily,
                    replications=bootstrap_replicates,
                    seed=20250113,
                ),
                "by_year": {
                    str(year): {
                        "sessions": sum(d.year == year for d in eligible_days),
                        "net_bps_per_session": (
                            sum(values[0] for d, values in daily.items() if d.year == year)
                            / sum(d.year == year for d in eligible_days)
                        ),
                    }
                    for year in sorted({d.year for d in eligible_days})
                },
            }
        )
    return {
        "cohort": cohort,
        "market_confirmed": confirmed,
        "capital_slots": positions,
        "long_trades": sum(event.side == 1 for event in selected),
        "short_trades": sum(event.side == -1 for event in selected),
        "traded_symbols": len({event.ticker for event in selected}),
        "cost_sensitivity": scenarios,
        "trades": [
            {
                "day": e.day.isoformat(),
                "ticker": e.ticker,
                "side": e.side,
                "decision_at": e.decision_at.isoformat(),
                "entry_at": e.entry_at.isoformat(),
                "exit_at": e.exit_at.isoformat(),
                "gross_bps": e.gross_bps,
                "rvol": e.rvol,
            }
            for e in selected
        ],
        "proxy_not_executable": True,
    }
