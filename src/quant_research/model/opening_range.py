"""Point-in-time opening-range breakout hypotheses and price-only trade proxies.

One possible trade per strategy and day; no optimization or order placement.
All entry windows start STRICTLY after decision availability, without instant fills.
Market confirmation only uses published factor data. Prior-volume conditioning
uses *earlier* market sessions; no same-day future volume enters a decision.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from math import isfinite, log
from statistics import mean
from typing import Literal, Sequence
from zoneinfo import ZoneInfo

from quant_research.model.first_passage_diagnostics import session_block_interval
from quant_research.model.first_passage_validation import MinuteBar, Session

MINUTE = timedelta(minutes=1)
BPS = 10_000.0
NEW_YORK = ZoneInfo("America/New_York")
Variant = Literal["breakout", "market_confirmed", "market_volume_confirmed"]
VARIANTS: tuple[Variant, ...] = (
    "breakout",
    "market_confirmed",
    "market_volume_confirmed",
)


@dataclass(frozen=True, slots=True)
class BreakoutTrade:
    day: date
    variant: Variant
    side: int
    decision_at: datetime
    entry_at: datetime
    exit_available_at: datetime
    entry_price: float
    exit_price: float

    @property
    def gross_bps(self) -> float:
        return self.side * log(self.exit_price / self.entry_price) * BPS


def _is_opening_session(session: Session, *, minimum_bars: int) -> bool:
    """Do not mistake a late-starting or gapped session for a real opening range."""
    bars = session.bars
    if len(bars) < minimum_bars:
        return False
    if bars[0].timestamp.tzinfo is None:
        return False
    if bars[0].timestamp.astimezone(NEW_YORK).strftime("%H:%M") != "09:31":
        return False
    return all(
        newer.timestamp - older.timestamp == MINUTE
        for older, newer in zip(bars[: minimum_bars - 1], bars[1:minimum_bars], strict=True)
    )


def _entry_window(
    bars: Sequence[MinuteBar], *, decision: datetime, horizon: int
) -> tuple[MinuteBar, MinuteBar] | None:
    stamps = [bar.timestamp for bar in bars]
    start_idx = bisect_right(stamps, decision + MINUTE)
    window = bars[start_idx : start_idx + horizon]
    if len(window) != horizon:
        return None
    start = window[0].timestamp - MINUTE
    if not decision < start <= decision + MINUTE:
        return None
    if any(b.timestamp - a.timestamp != MINUTE for a, b in zip(window, window[1:], strict=False)):
        return None
    if window[-1].available_at <= decision:
        return None
    return window[0], window[-1]


def _past_open_volume(
    prior_sessions: Sequence[Session], *, opening_minutes: int, history_sessions: int
) -> float | None:
    if len(prior_sessions) < history_sessions:
        return None
    volumes: list[float] = []
    for session in prior_sessions[-history_sessions:]:
        if not _is_opening_session(session, minimum_bars=opening_minutes):
            return None
        first = session.bars[:opening_minutes]
        if any(bar.volume is None or not isfinite(bar.volume) or bar.volume < 0 for bar in first):
            return None
        volumes.append(sum(bar.volume for bar in first if bar.volume is not None))
    average = mean(volumes)
    return average if average > 0 else None


def _factor_side(factor: Session, *, decision: datetime, max_staleness: timedelta) -> int:
    """Return sign of QQQ session-to-date return using only published bars."""
    if not _is_opening_session(factor, minimum_bars=1):
        return 0
    first = factor.bars[0]
    available = [
        bar for bar in factor.bars if bar.timestamp <= decision and bar.available_at <= decision
    ]
    if not available:
        return 0
    last = available[-1]
    if decision - last.available_at > max_staleness:
        return 0
    if last.close > first.open:
        return 1
    if last.close < first.open:
        return -1
    return 0


def build_opening_range_trades(
    *,
    asset_sessions: Sequence[Session],
    factor_sessions: Sequence[Session],
    horizon_minutes: int,
    opening_minutes: int = 15,
    scan_minutes: int = 120,
    breakout_buffer_bps: float = 5.0,
    history_sessions: int = 10,
    unusual_volume_ratio: float = 1.5,
    diagnostics: dict[str, int] | None = None,
) -> dict[Variant, tuple[BreakoutTrade, ...]]:
    """Fixed signal family. Each variant can take at most one trade per session.

    No model is fit, so previous sessions are used only for a point-in-time
    volume baseline. Caller must supply only approved pre-holdout sessions.
    No record of a target/exit beyond session end is ever used.
    """
    if (
        horizon_minutes < 1
        or opening_minutes < 1
        or scan_minutes < opening_minutes + 1
        or history_sessions < 1
    ):
        raise ValueError("invalid minute/history configuration")
    if not isfinite(breakout_buffer_bps) or breakout_buffer_bps < 0:
        raise ValueError("breakout buffer must be nonnegative finite")
    if not isfinite(unusual_volume_ratio) or unusual_volume_ratio <= 0:
        raise ValueError("volume ratio must be positive finite")
    days = [s.day for s in asset_sessions]
    if days != sorted(set(days)) or days != [s.day for s in factor_sessions]:
        raise ValueError("asset/factor sessions must have matching sorted unique days")
    events: dict[Variant, list[BreakoutTrade]] = {variant: [] for variant in VARIANTS}
    counts: defaultdict[str, int] = defaultdict(int)
    for idx, (asset, factor) in enumerate(zip(asset_sessions, factor_sessions, strict=True)):
        counts["sessions"] += 1
        for session in (asset, factor):
            if any(
                newer.timestamp <= older.timestamp or newer.available_at < older.available_at
                for older, newer in zip(session.bars, session.bars[1:], strict=False)
            ):
                raise ValueError("bar timestamps and availability must be ordered")
        if not _is_opening_session(asset, minimum_bars=opening_minutes):
            counts["missing_or_misaligned_opening_range"] += 1
            continue
        first = asset.bars[:opening_minutes]
        upper = max(bar.high for bar in first)
        lower = min(bar.low for bar in first)
        if lower <= 0 or upper <= 0:
            raise ValueError("opening range has invalid prices")
        reference_open_volume = _past_open_volume(
            asset_sessions[:idx],
            opening_minutes=opening_minutes,
            history_sessions=history_sessions,
        )
        same_day_volume = (
            sum(bar.volume for bar in first if bar.volume is not None)
            if all(
                bar.volume is not None and isfinite(bar.volume) and bar.volume >= 0 for bar in first
            )
            else None
        )
        unusual_volume = (
            reference_open_volume is not None
            and same_day_volume is not None
            and same_day_volume / reference_open_volume >= unusual_volume_ratio
        )
        if unusual_volume:
            counts["high_volume_sessions"] += 1
        elif reference_open_volume is None:
            counts["insufficient_prior_volume_history"] += 1
        seen: set[Variant] = set()
        for bar in asset.bars[opening_minutes:scan_minutes]:
            if seen == set(VARIANTS):
                break
            if bar.available_at.tzinfo is None or bar.available_at.utcoffset() != UTC.utcoffset(
                bar.available_at
            ):
                raise ValueError("decision times must be UTC-aware")
            if bar.close > upper * (1 + breakout_buffer_bps / BPS):
                side = 1
            elif bar.close < lower * (1 - breakout_buffer_bps / BPS):
                side = -1
            else:
                continue
            decision = bar.available_at
            # Every opening-range and volume input must have been published
            # by decision time. Gaps or delayed bars cannot leak future bars.
            if any(open_bar.available_at > decision for open_bar in first):
                counts["opening_range_unpublished"] += 1
                continue
            entry = _entry_window(asset.bars, decision=decision, horizon=horizon_minutes)
            if entry is None:
                counts["ineligible_entry_or_exit"] += 1
                continue
            market_side = _factor_side(
                factor, decision=decision, max_staleness=timedelta(seconds=90)
            )
            for variant in VARIANTS:
                if variant in seen:
                    continue
                if variant != "breakout" and market_side != side:
                    continue
                if variant == "market_volume_confirmed" and not unusual_volume:
                    continue
                first_future, last_future = entry
                events[variant].append(
                    BreakoutTrade(
                        day=asset.day,
                        variant=variant,
                        side=side,
                        decision_at=decision,
                        entry_at=first_future.timestamp - MINUTE,
                        exit_available_at=last_future.available_at,
                        entry_price=first_future.open,
                        exit_price=last_future.close,
                    )
                )
                seen.add(variant)
                counts[f"trades_{variant}"] += 1
    if diagnostics is not None:
        diagnostics.update(counts)
    return {variant: tuple(events[variant]) for variant in VARIANTS}


def score_breakout_trades(
    trades: Sequence[BreakoutTrade],
    *,
    eligible_days: Sequence[date],
    roundtrip_cost_bps: float,
    bootstrap_replicates: int = 2000,
) -> dict[str, object]:
    """Session-level net P&L for one strategy and one ticker (proxy only)."""
    if not isfinite(roundtrip_cost_bps) or roundtrip_cost_bps < 0:
        raise ValueError("invalid cost")
    if bootstrap_replicates < 1:
        raise ValueError("invalid bootstrap count")
    days = tuple(eligible_days)
    if not days or len(set(days)) != len(days):
        raise ValueError("eligible days must be nonempty and unique")
    if any(trade.day not in days for trade in trades):
        raise ValueError("trade falls outside supplied validation days")
    if len(set(trade.day for trade in trades)) != len(trades):
        raise ValueError("more than one trade per day/variant")
    daily = {day: [0.0] for day in days}
    net_returns = []
    gross_returns = []
    for trade in trades:
        if (
            trade.side not in (-1, 1)
            or not trade.decision_at < trade.entry_at < trade.exit_available_at
        ):
            raise ValueError("invalid trade timing or side")
        if trade.entry_price <= 0 or trade.exit_price <= 0:
            raise ValueError("invalid trade prices")
        gross = trade.gross_bps
        net = gross - roundtrip_cost_bps
        daily[trade.day] = [net]
        gross_returns.append(gross)
        net_returns.append(net)
    count = len(net_returns)
    return {
        "scored_sessions": len(days),
        "trades": count,
        "active_sessions": count,
        "long_trades": sum(t.side == 1 for t in trades),
        "short_trades": sum(t.side == -1 for t in trades),
        "roundtrip_cost_bps": roundtrip_cost_bps,
        "gross_bps_per_trade": mean(gross_returns) if count else None,
        "net_bps_per_trade": mean(net_returns) if count else None,
        "net_bps_per_session": sum(net_returns) / len(days),
        "net_bps_total_one_unit_notional": sum(net_returns),
        "positive_trade_fraction": sum(r > 0 for r in net_returns) / count if count else None,
        "net_bps_per_session_95pct_bootstrap": session_block_interval(
            daily, replications=bootstrap_replicates, seed=20250113
        ),
        "proxy_not_executable": True,
    }
