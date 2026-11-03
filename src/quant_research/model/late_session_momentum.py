"""Research-only early-to-late ETF momentum, with strict bar-time alignment.

Entry is the 15:31 minute OPEN and exit the 15:59 minute OPEN for a
standard 390-minute US equity session. Neither is an executable quotation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from math import isfinite, log
from typing import Literal, Sequence

from quant_research.model.first_passage_diagnostics import session_block_interval
from quant_research.model.first_passage_validation import MinuteBar, Session

BPS = 10_000.0
MINUTE = timedelta(minutes=1)
Signal = Literal["prev_close_momentum", "open_momentum", "always_long"]
SIGNALS: tuple[Signal, ...] = (
    "prev_close_momentum",
    "open_momentum",
    "always_long",
)


@dataclass(frozen=True, slots=True)
class LateSessionEvent:
    day: date
    ticker: str
    entry_at: datetime
    exit_at: datetime
    first_half_hour_available_at: datetime
    early_prevclose_bps: float
    early_open_bps: float
    forward_bps: float


def _sign(value: float) -> int:
    return (value > 0) - (value < 0)


def position(event: LateSessionEvent, signal: Signal) -> int:
    if signal == "prev_close_momentum":
        return _sign(event.early_prevclose_bps)
    if signal == "open_momentum":
        return _sign(event.early_open_bps)
    if signal == "always_long":
        return 1
    raise ValueError(f"unknown signal: {signal}")


def build_late_session_event(
    *,
    ticker: str,
    session: Session,
    previous_close: float,
    opening: datetime,
    closing: datetime,
) -> LateSessionEvent | None:
    """Complete first 30m and final 29m required; never use later bars as features.

    Standard sessions only: 390 regular trading minutes. A 1-minute delay
    from the 15:30 decision to entry at 15:31 is deliberately imposed.
    Both entry and exit prices are minute OPEN proxies, avoiding a 16:00 close
    auction assumption. Actual market orders have spread/slippage/latency.
    """
    if not isfinite(previous_close) or previous_close <= 0:
        return None
    if closing - opening != timedelta(minutes=390):
        return None
    if opening.tzinfo is None or closing.tzinfo is None:
        raise ValueError("exchange times must be timezone aware")
    by_end: dict[datetime, MinuteBar] = {}
    for bar in session.bars:
        if bar.timestamp.tzinfo is None or bar.available_at.tzinfo is None:
            raise ValueError("minute bar timestamps must be timezone aware")
        if bar.timestamp in by_end:
            raise ValueError("duplicate minute end timestamp")
        by_end[bar.timestamp] = bar
    first: list[MinuteBar] = []
    last: list[MinuteBar] = []
    for i in range(1, 31):
        first_bar = by_end.get(opening + i * MINUTE)
        if first_bar is None:
            return None
        first.append(first_bar)
    for i in range(362, 391):
        last_bar = by_end.get(opening + i * MINUTE)
        if last_bar is None:
            return None
        last.append(last_bar)
    # Entry at 15:31 (bar ending 15:32), exit at 15:59 (bar ending 16:00).
    entry_at = last[0].timestamp - MINUTE
    exit_at = last[-1].timestamp - MINUTE
    decision_at = closing - timedelta(minutes=30)
    if entry_at != closing - timedelta(minutes=29):
        raise ValueError("late-session entry misaligned")
    if exit_at != closing - MINUTE or not decision_at < entry_at < exit_at:
        raise ValueError("late-session exit misaligned")
    if any(bar.available_at > decision_at for bar in first):
        return None
    # Require published first-half-hour bar and usable prices at entry/exit.
    opening_price = first[0].open
    halfhour_close = first[-1].close
    entry_price = last[0].open
    exit_price = last[-1].open
    if any(
        not isfinite(p) or p <= 0 for p in (opening_price, halfhour_close, entry_price, exit_price)
    ):
        return None
    return LateSessionEvent(
        day=session.day,
        ticker=ticker,
        entry_at=entry_at,
        exit_at=exit_at,
        first_half_hour_available_at=first[-1].available_at,
        early_prevclose_bps=log(halfhour_close / previous_close) * BPS,
        early_open_bps=log(halfhour_close / opening_price) * BPS,
        forward_bps=log(exit_price / entry_price) * BPS,
    )


def score_late_session(
    *,
    events: Sequence[LateSessionEvent],
    eligible_days: Sequence[date],
    signal: Signal,
    costs_bps: Sequence[float],
    replications: int = 1000,
) -> dict[str, object]:
    """Session-equally weighted, zero on skipped sessions, fixed trades for costs."""
    if len(set(eligible_days)) != len(eligible_days) or not eligible_days:
        raise ValueError("eligible session dates must be nonempty and unique")
    if (
        replications < 1
        or not costs_bps
        or any(not isfinite(cost) or cost < 0 for cost in costs_bps)
    ):
        raise ValueError("invalid cost sensitivity or bootstrap configuration")
    allowed = set(eligible_days)
    if any(e.day not in allowed for e in events):
        raise ValueError("event outside eligible dates")
    if len({e.day for e in events}) != len(events):
        raise ValueError("multiple events for one ticker/day")
    positions = {e.day: position(e, signal) for e in events}
    gross = {e.day: positions[e.day] * e.forward_bps for e in events}
    traded = sum(pos != 0 for pos in positions.values())
    by_cost: list[dict[str, object]] = []
    forward_by_day = {e.day: e.forward_bps for e in events}
    for cost in costs_bps:
        net = {
            day: gross.get(day, 0.0) - cost * int(positions.get(day, 0) != 0)
            for day in eligible_days
        }
        benchmark_net = {
            day: (forward_by_day[day] - cost) if day in positions else 0.0 for day in eligible_days
        }
        paired = {day: net[day] - benchmark_net[day] for day in eligible_days}
        paired_interval = session_block_interval(
            {day: [difference] for day, difference in paired.items()},
            replications=replications,
            seed=20261101,
        )
        values = list(net.values())
        traded_values = [net[day] for day in eligible_days if positions.get(day, 0)]
        interval = session_block_interval(
            {d: [net[d]] for d in eligible_days},
            replications=replications,
            seed=20261101,
        )
        by_month: dict[str, list[float]] = {}
        by_year: dict[str, list[float]] = {}
        for day, value in net.items():
            by_month.setdefault(day.strftime("%Y-%m"), []).append(value)
            by_year.setdefault(str(day.year), []).append(value)
        by_cost.append(
            {
                "roundtrip_cost_bps": cost,
                "trades": traded,
                "net_bps_per_session": sum(values) / len(values),
                "net_bps_per_trade": sum(traded_values) / traded if traded else None,
                "net_bps_vs_always_long": sum(paired.values()) / len(eligible_days),
                "paired_session_bootstrap_95ci": paired_interval,
                "long_net_bps_per_trade": (
                    sum(net[d] for d in eligible_days if positions.get(d) == 1)
                    / sum(p == 1 for p in positions.values())
                    if any(p == 1 for p in positions.values())
                    else None
                ),
                "short_net_bps_per_trade": (
                    sum(net[d] for d in eligible_days if positions.get(d) == -1)
                    / sum(p == -1 for p in positions.values())
                    if any(p == -1 for p in positions.values())
                    else None
                ),
                "session_bootstrap_95ci": interval,
                "by_month_net_bps_per_session": {
                    month: sum(vals) / len(vals) for month, vals in sorted(by_month.items())
                },
                "by_year_net_bps_per_session": {
                    year: sum(vals) / len(vals) for year, vals in sorted(by_year.items())
                },
            }
        )
    return {
        "eligible_sessions": len(eligible_days),
        "signal": signal,
        "trades": traded,
        "long_trades": sum(p > 0 for p in positions.values()),
        "short_trades": sum(p < 0 for p in positions.values()),
        "gross_bps_per_session": sum(gross.values()) / len(eligible_days),
        "cost_sensitivity": by_cost,
        "event_details": [
            {
                "date": e.day.isoformat(),
                "side": position(e, signal),
                "entry_at": e.entry_at.isoformat(),
                "exit_at": e.exit_at.isoformat(),
                "early_prevclose_bps": e.early_prevclose_bps,
                "early_open_bps": e.early_open_bps,
                "forward_bps": e.forward_bps,
                "gross_bps": position(e, signal) * e.forward_bps,
            }
            for e in events
        ],
    }
