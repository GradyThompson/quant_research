"""Audit timestamp and barrier-reference conventions without changing backtests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pyarrow as pa

from quant_research.model.barrier import BarrierForecast


@dataclass(frozen=True)
class AlignmentExample:
    forecast_at: datetime
    last_known_end: datetime | None
    last_known_close: float | None
    calibration_entry_at: datetime | None
    calibration_open: float | None
    execution_entry_at: datetime | None
    execution_open: float | None
    reference_gap_pct: float | None


@dataclass(frozen=True)
class AlignmentAudit:
    checked: int
    differing_entry_times: int
    differing_reference_prices: int
    missing_calibration_entry: int
    missing_execution_entry: int
    examples: tuple[AlignmentExample, ...]


def audit_barrier_alignment(
    *, forecasts: list[BarrierForecast], bars: pa.Table, sample_count: int = 8
) -> AlignmentAudit:
    """Compare calibration's >= decision bar-start to execution's strict >.

    A bar's timestamp denotes its end. Only bars with available_at <=
    forecast time can be used as prior inputs. This audit cannot inspect
    internals of private models; it audits observable inputs and references.
    """
    required = {"timestamp", "available_at", "open", "close"}
    if not required.issubset(bars.column_names):
        raise ValueError("Audit requires timestamp, available_at, open and close")
    rows = bars.select(sorted(required)).to_pylist()
    if sample_count < 0:
        raise ValueError("sample_count must be nonnegative")
    if any(rows[i]["timestamp"] >= rows[i + 1]["timestamp"] for i in range(len(rows) - 1)):
        raise ValueError("Bars must be strictly sorted")
    examples: list[AlignmentExample] = []
    timing = prices = missing_cal = missing_exec = 0
    for forecast in forecasts:
        at = forecast.generated_at
        known = [r for r in rows if r["available_at"] <= at]
        cal = next((r for r in rows if r["timestamp"] - timedelta(minutes=1) >= at), None)
        exe = next((r for r in rows if r["timestamp"] - timedelta(minutes=1) > at), None)
        last = known[-1] if known else None
        cal_time = cal["timestamp"] - timedelta(minutes=1) if cal else None
        exe_time = exe["timestamp"] - timedelta(minutes=1) if exe else None
        gap = (
            100 * (float(cal["open"]) / float(last["close"]) - 1)
            if cal and last and float(last["close"]) > 0
            else None
        )
        missing_cal += int(cal is None)
        missing_exec += int(exe is None)
        timing += int(cal_time != exe_time)
        prices += int(cal is not None and exe is not None and cal["open"] != exe["open"])
        if len(examples) < sample_count:
            examples.append(
                AlignmentExample(
                    forecast_at=at,
                    last_known_end=last["timestamp"] if last else None,
                    last_known_close=float(last["close"]) if last else None,
                    calibration_entry_at=cal_time,
                    calibration_open=float(cal["open"]) if cal else None,
                    execution_entry_at=exe_time,
                    execution_open=float(exe["open"]) if exe else None,
                    reference_gap_pct=gap,
                )
            )
    return AlignmentAudit(
        len(forecasts), timing, prices, missing_cal, missing_exec, tuple(examples)
    )
