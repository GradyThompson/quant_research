"""Tests for the first alpha hypothesis: short-horizon mean reversion."""

from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest

from quant_research.data.schemas import BarFrequency, ohlcv_schema
from quant_research.signals import MeanReversionSignal


def make_bars(prices: dict[str, list[float | None]], *, availability_delay: int = 60) -> pa.Table:
    days = [datetime(2025, 1, day, 21, tzinfo=UTC) for day in (2, 3, 6, 7)]
    rows = []
    for security_id, closes in prices.items():
        for timestamp, close in zip(days, closes, strict=True):
            rows.append(
                {
                    "security_id": security_id,
                    "timestamp": timestamp,
                    "available_at": timestamp + timedelta(seconds=availability_delay),
                    "ingested_at": timestamp + timedelta(seconds=availability_delay + 1),
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 100,
                    "currency": "USD",
                }
            )
    return pa.Table.from_pylist(rows, schema=ohlcv_schema(frequency=BarFrequency.DAILY))


def test_scores_and_ranks():
    bars = make_bars(
        {
            "SEC-000001": [100, 100, 90, 90],
            "SEC-000002": [100, 100, 110, 110],
            "SEC-000003": [100, 100, 100, 100],
        }
    )
    rows = MeanReversionSignal(lookback=2).compute(bars).to_pylist()
    dated = {r["security_id"]: r for r in rows if r["timestamp"].day == 6}
    assert dated["SEC-000001"]["signal"] == pytest.approx(0.1)
    assert dated["SEC-000001"]["cross_sectional_rank"] == pytest.approx(1.0)
    assert dated["SEC-000002"]["signal"] == pytest.approx(-0.1)
    assert dated["SEC-000002"]["cross_sectional_rank"] == pytest.approx(0.0)
    assert dated["SEC-000003"]["cross_sectional_rank"] == pytest.approx(0.5)
    assert dated["SEC-000001"]["available_at"] == dated["SEC-000001"]["timestamp"] + timedelta(
        seconds=60
    )
    assert all(r["signal"] is None for r in rows if r["timestamp"].day in (2, 3))


def test_ties_receive_average_rank():
    bars = make_bars(
        {
            "SEC-000001": [100, 100, 90, 90],
            "SEC-000002": [100, 100, 90, 90],
            "SEC-000003": [100, 100, 110, 110],
        }
    )
    rows = MeanReversionSignal(2).compute(bars).to_pylist()
    ranked = {r["security_id"]: r["cross_sectional_rank"] for r in rows if r["timestamp"].day == 6}
    assert ranked["SEC-000001"] == pytest.approx(0.75)
    assert ranked["SEC-000002"] == pytest.approx(0.75)
    assert ranked["SEC-000003"] == pytest.approx(0)


def test_single_security_has_score_but_no_cross_sectional_rank():
    rows = MeanReversionSignal(2).compute(make_bars({"SEC-000001": [100, 100, 90, 90]})).to_pylist()
    row = next(r for r in rows if r["timestamp"].day == 6)
    assert row["signal"] == pytest.approx(0.1)
    assert row["cross_sectional_rank"] is None


def test_missing_close_resets_rolling_history():
    rows = (
        MeanReversionSignal(2).compute(make_bars({"SEC-000001": [100, None, 90, 95]})).to_pylist()
    )
    assert all(r["signal"] is None for r in rows)


def test_duplicate_bars_are_rejected():
    bars = make_bars({"SEC-000001": [100, 100, 90, 90]})
    with pytest.raises(Exception, match="Duplicate OHLCV"):
        MeanReversionSignal(2).compute(pa.concat_tables([bars, bars]))


@pytest.mark.parametrize("lookback", [0, -1, True])
def test_invalid_lookbacks(lookback):
    with pytest.raises(ValueError):
        MeanReversionSignal(lookback=lookback)


def test_output_metadata():
    output = MeanReversionSignal(2).compute(make_bars({"SEC-000001": [100, 100, 90, 90]}))
    assert output.schema.metadata[b"signal_name"] == b"mean_reversion"
    assert output.schema.metadata[b"lookback_observations"] == b"2"
