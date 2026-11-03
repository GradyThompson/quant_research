"""Research-only point-in-time intraday barrier labels and features.

Minute bars use end timestamps. A published 10:00 bar with 60-second latency
cannot be used until 10:01; hypothetical entry is STRICTLY after decision.
Unknown high/low ordering is resolved as a LOWER-first loss for long positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from itertools import chain
from math import isfinite, log, sqrt
from typing import Any, Sequence

from quant_research.model.first_passage import first_passage_probabilities

MINUTE = timedelta(minutes=1)
BPS = 10_000.0
FEATURE_NAMES = (
    "ret_5m",
    "ret_15m",
    "ret_30m",
    "range_30m",
    "close_location",
    "sigma_1m",
    "volume_15m_relative",
    "opening_dollar_volume_log",
    "prior_adv_log",
    "market_ret_30m",
    "market_ret_5m",
    "relative_ret_30m",
    "math_upper",
    "math_lower",
    "math_neither",
)


@dataclass(frozen=True, slots=True)
class BarrierRow:
    day: date
    ticker: str
    decision_at: datetime
    entry_at: datetime
    resolved_at: datetime
    features: tuple[float, ...]
    baseline: tuple[float, float, float]
    outcome: int  # 0 upper, 1 lower, 2 neither
    gross_return_bps: float
    timeout_return_bps: float  # zero unless timeout
    entry_price: float
    ambiguous: bool


def _number(row: dict[str, object], key: str) -> float:
    value = row[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"invalid {key}")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"nonfinite {key}")
    return result


def _timestamp(row: dict[str, object], field: str) -> datetime:
    value = row[field]
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"invalid {field}")
    return value


def construct_barrier_row(
    *,
    day: date,
    ticker: str,
    asset: Sequence[dict[str, object]],
    market: Sequence[dict[str, object]],
    opening_volume_baseline: float,
    prior_adv: float,
    decision_minutes: int = 30,
    horizon_minutes: int = 60,
    upper_pct: float = 0.005,
    lower_pct: float = 0.02,
) -> BarrierRow | None:
    """Only completed/published opening minutes enter features.

    First post-decision bar must start strictly later than decision timestamp.
    An exit-open is required for every timeout; halted/missing bars reject row.
    No same-day close, post-entry volume, or future liquidity is an input.
    """
    if not (1 <= decision_minutes <= 100 and 1 <= horizon_minutes <= 240):
        raise ValueError("invalid horizon")
    if not (0 < upper_pct < 1 and 0 < lower_pct < 1):
        raise ValueError("invalid barriers")
    if opening_volume_baseline <= 0 or prior_adv <= 0:
        return None
    if len(asset) < decision_minutes + horizon_minutes + 3 or len(market) < decision_minutes:
        return None
    initial = asset[:decision_minutes]
    factor = market[:decision_minutes]
    first_end = _timestamp(initial[0], "timestamp")
    for j, row in enumerate(initial):
        if _timestamp(row, "timestamp") != first_end + j * MINUTE:
            return None
    for j, row in enumerate(factor):
        if _timestamp(row, "timestamp") != first_end + j * MINUTE:
            return None
    # Do not shift the decision to accommodate a late bar: that would
    # silently change the entry clock and the available information set.
    decision = _timestamp(initial[-1], "timestamp") + MINUTE
    if any(_timestamp(row, "available_at") > decision for row in chain(initial, factor)):
        return None
    starts = [_timestamp(row, "timestamp") - MINUTE for row in asset]
    idx = next((j for j, start in enumerate(starts) if start > decision), None)
    if idx is None:
        return None
    entry_at = starts[idx]
    if entry_at > decision + 2 * MINUTE:
        return None  # no assumed fill after long delay
    if idx + horizon_minutes >= len(asset):
        return None
    for j in range(idx, idx + horizon_minutes + 1):
        if starts[j] != entry_at + (j - idx) * MINUTE:
            return None
    entry_price = _number(asset[idx], "open")
    if entry_price <= 0:
        return None
    upper = entry_price * (1 + upper_pct)
    lower = entry_price * (1 - lower_pct)
    observed = 2
    gross = 0.0
    ambiguous = False
    resolved = entry_at + horizon_minutes * MINUTE
    for j in range(idx, idx + horizon_minutes):
        row = asset[j]
        opening = _number(row, "open")
        high = _number(row, "high")
        low = _number(row, "low")
        if min(opening, high, low) <= 0 or low > opening or high < opening:
            return None
        if opening <= lower:
            observed = 1
            gross = BPS * (opening / entry_price - 1)
        elif opening >= upper:
            observed = 0
            gross = upper_pct * BPS  # do not assume favorable price improvement
        elif low <= lower:
            # With both barriers reached, pessimistically count the stop first.
            observed = 1
            gross = -lower_pct * BPS
            ambiguous = high >= upper
        elif high >= upper:
            observed = 0
            gross = upper_pct * BPS
        if observed != 2:
            resolved = _timestamp(row, "timestamp")
            break
    timeout = 0.0
    if observed == 2:
        exit_price = _number(asset[idx + horizon_minutes], "open")
        timeout = BPS * (exit_price / entry_price - 1)
        gross = timeout
    first_close = _number(initial[0], "open")
    closes = [_number(row, "close") for row in initial]
    highs = [_number(row, "high") for row in initial]
    lows = [_number(row, "low") for row in initial]
    returns = [log(b / a) for a, b in zip(closes, closes[1:], strict=False)]
    sigma = sqrt(sum(ret * ret for ret in returns) / len(returns))
    if sigma == 0:
        sigma = 1e-8
    math = first_passage_probabilities(
        upper_barrier_pct=upper_pct,
        lower_barrier_pct=lower_pct,
        horizon_minutes=horizon_minutes,
        volatility_per_sqrt_minute=sigma,
        drift_mode="price_zero",
    )
    first_15 = sum(_number(row, "volume") for row in initial[:15])
    volume_relative = first_15 / opening_volume_baseline
    market_closes = [_number(row, "close") for row in factor]
    stock_30 = log(closes[-1] / first_close)
    market_30 = log(market_closes[-1] / _number(factor[0], "open"))
    hi, lo = max(highs), min(lows)
    features = (
        log(closes[-1] / closes[-6]),
        log(closes[-1] / closes[-16]),
        stock_30,
        log(hi / lo),
        (closes[-1] - lo) / (hi - lo) if hi > lo else 0.5,
        sigma,
        volume_relative,
        log(max(1.0, first_15 * first_close)),
        log(prior_adv),
        market_30,
        log(market_closes[-1] / market_closes[-6]),
        stock_30 - market_30,
        math.upper_first,
        math.lower_first,
        math.neither,
    )
    if not all(isfinite(value) for value in features):
        return None
    return BarrierRow(
        day=day,
        ticker=ticker,
        decision_at=decision,
        entry_at=entry_at,
        resolved_at=resolved,
        features=features,
        baseline=(math.upper_first, math.lower_first, math.neither),
        outcome=observed,
        gross_return_bps=gross,
        timeout_return_bps=timeout,
        entry_price=entry_price,
        ambiguous=ambiguous,
    )


def row_to_json(row: BarrierRow) -> dict[str, Any]:
    return {
        "day": row.day.isoformat(),
        "ticker": row.ticker,
        "decision_at": row.decision_at.isoformat(),
        "entry_at": row.entry_at.isoformat(),
        "resolved_at": row.resolved_at.isoformat(),
        "features": list(row.features),
        "baseline": list(row.baseline),
        "outcome": row.outcome,
        "gross_return_bps": row.gross_return_bps,
        "timeout_return_bps": row.timeout_return_bps,
        "entry_price": row.entry_price,
        "ambiguous": row.ambiguous,
    }


def row_from_json(item: dict[str, Any]) -> BarrierRow:
    raw_baseline = item["baseline"]
    if not isinstance(raw_baseline, (list, tuple)) or len(raw_baseline) != 3:
        raise ValueError("incorrect saved feature schema")
    baseline = (float(raw_baseline[0]), float(raw_baseline[1]), float(raw_baseline[2]))
    row = BarrierRow(
        day=date.fromisoformat(item["day"]),
        ticker=str(item["ticker"]),
        decision_at=datetime.fromisoformat(item["decision_at"]),
        entry_at=datetime.fromisoformat(item["entry_at"]),
        resolved_at=datetime.fromisoformat(item["resolved_at"]),
        features=tuple(float(x) for x in item["features"]),
        baseline=baseline,
        outcome=int(item["outcome"]),
        gross_return_bps=float(item["gross_return_bps"]),
        timeout_return_bps=float(item["timeout_return_bps"]),
        entry_price=float(item["entry_price"]),
        ambiguous=bool(item["ambiguous"]),
    )
    if len(row.features) != len(FEATURE_NAMES) or len(row.baseline) != 3:
        raise ValueError("incorrect saved feature schema")
    if row.outcome not in (0, 1, 2):
        raise ValueError("incorrect saved outcome")
    if row.resolved_at <= row.entry_at or row.decision_at >= row.entry_at:
        raise ValueError("invalid saved label chronology")
    return row
