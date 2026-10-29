"""Tests for canonical OHLCV data validation."""

from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest

from quant_research.data.schemas import (
    BarFrequency,
    PriceAdjustment,
    ohlcv_schema,
)
from quant_research.data.validation import OHLCVValidator
from quant_research.exceptions import DataValidationError

BAR_END = datetime(2025, 1, 2, 21, 0, tzinfo=UTC)


@pytest.fixture
def validator():
    return OHLCVValidator(frequency=BarFrequency.DAILY)


@pytest.fixture
def valid_row():
    return {
        "security_id": "SEC-000001",
        "timestamp": BAR_END,
        "available_at": BAR_END + timedelta(seconds=5),
        "ingested_at": BAR_END + timedelta(seconds=10),
        "open": 100.0,
        "high": 105.0,
        "low": 98.0,
        "close": 103.0,
        "volume": 1000000,
        "currency": "USD",
    }


def make_table(
    rows,
    *,
    adjustment=PriceAdjustment.RAW,
):
    schema = ohlcv_schema(
        frequency=BarFrequency.DAILY,
        adjustment=adjustment,
    )

    return pa.Table.from_pylist(rows, schema=schema)


def test_valid_ohlcv(validator, valid_row):
    table = make_table([valid_row])

    validator.validate(table)


def test_empty_table_is_valid(validator):
    table = make_table([])

    validator.validate(table)


def test_invalid_security_id(validator, valid_row):
    valid_row["security_id"] = "AAPL"

    table = make_table([valid_row])

    with pytest.raises(
        DataValidationError,
        match="Invalid security_id",
    ):
        validator.validate(table)


def test_duplicate_observations(validator, valid_row):
    table = make_table([valid_row, valid_row.copy()])

    with pytest.raises(
        DataValidationError,
        match="Duplicate OHLCV observation",
    ):
        validator.validate(table)


def test_same_timestamp_different_securities(
    validator,
    valid_row,
):
    second = valid_row.copy()
    second["security_id"] = "SEC-000002"

    table = make_table([valid_row, second])

    validator.validate(table)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("open", -1.0),
        ("high", -1.0),
        ("low", 0.0),
        ("close", 0.0),
        ("open", float("nan")),
        ("high", float("inf")),
        ("low", float("-inf")),
    ],
)
def test_invalid_prices(
    validator,
    valid_row,
    column,
    value,
):
    valid_row[column] = value

    table = make_table([valid_row])

    with pytest.raises(
        DataValidationError,
        match="Invalid .* price",
    ):
        validator.validate(table)


def test_high_below_low(validator, valid_row):
    valid_row["high"] = 90.0

    table = make_table([valid_row])

    with pytest.raises(
        DataValidationError,
        match="High is below low",
    ):
        validator.validate(table)


def test_open_above_high(validator, valid_row):
    valid_row["open"] = 110.0

    table = make_table([valid_row])

    with pytest.raises(
        DataValidationError,
        match="open exceeds high",
    ):
        validator.validate(table)


def test_close_above_high(validator, valid_row):
    valid_row["close"] = 110.0

    table = make_table([valid_row])

    with pytest.raises(
        DataValidationError,
        match="close exceeds high",
    ):
        validator.validate(table)


def test_open_below_low(validator, valid_row):
    valid_row["open"] = 90.0

    table = make_table([valid_row])

    with pytest.raises(
        DataValidationError,
        match="open is below low",
    ):
        validator.validate(table)


def test_close_below_low(validator, valid_row):
    valid_row["close"] = 90.0

    table = make_table([valid_row])

    with pytest.raises(
        DataValidationError,
        match="close is below low",
    ):
        validator.validate(table)


def test_missing_prices_allowed(validator, valid_row):
    for column in ("open", "high", "low", "close"):
        valid_row[column] = None

    table = make_table([valid_row])

    validator.validate(table)


def test_partial_price_relationships(validator, valid_row):
    valid_row["open"] = 110.0
    valid_row["close"] = None
    valid_row["low"] = None

    table = make_table([valid_row])

    with pytest.raises(
        DataValidationError,
        match="open exceeds high",
    ):
        validator.validate(table)


def test_negative_volume(validator, valid_row):
    valid_row["volume"] = -100

    table = make_table([valid_row])

    with pytest.raises(
        DataValidationError,
        match="Invalid volume",
    ):
        validator.validate(table)


def test_zero_volume_allowed(validator, valid_row):
    valid_row["volume"] = 0

    table = make_table([valid_row])

    validator.validate(table)


def test_missing_volume_allowed(validator, valid_row):
    valid_row["volume"] = None

    table = make_table([valid_row])

    validator.validate(table)


@pytest.mark.parametrize(
    "currency",
    ["usd", "US", "USDD", "", "12A"],
)
def test_invalid_currency(
    validator,
    valid_row,
    currency,
):
    valid_row["currency"] = currency

    table = make_table([valid_row])

    with pytest.raises(
        DataValidationError,
        match="Invalid currency",
    ):
        validator.validate(table)


def test_missing_currency_rejected(validator, valid_row):
    valid_row["currency"] = None

    table = make_table([valid_row])

    with pytest.raises(
        DataValidationError,
        match="Required column 'currency'",
    ):
        validator.validate(table)


def test_availability_before_bar_end(validator, valid_row):
    valid_row["available_at"] = BAR_END - timedelta(seconds=1)

    table = make_table([valid_row])

    with pytest.raises(
        DataValidationError,
        match="Availability precedes bar end",
    ):
        validator.validate(table)


def test_ingestion_before_availability(validator, valid_row):
    valid_row["ingested_at"] = BAR_END + timedelta(seconds=2)

    table = make_table([valid_row])

    with pytest.raises(
        DataValidationError,
        match="Ingestion precedes availability",
    ):
        validator.validate(table)


def test_equal_temporal_boundaries_allowed(validator, valid_row):
    valid_row["available_at"] = BAR_END
    valid_row["ingested_at"] = BAR_END

    table = make_table([valid_row])

    validator.validate(table)


def test_wrong_price_adjustment_metadata(validator, valid_row):
    table = make_table(
        [valid_row],
        adjustment=PriceAdjustment.SPLIT_ADJUSTED,
    )

    with pytest.raises(
        DataValidationError,
        match="OHLCV schema mismatch",
    ):
        validator.validate(table)


def test_missing_schema_metadata(validator, valid_row):
    table = make_table([valid_row])
    table = table.replace_schema_metadata(None)

    with pytest.raises(
        DataValidationError,
        match="OHLCV schema mismatch",
    ):
        validator.validate(table)


def test_extra_column_rejected(validator, valid_row):
    table = make_table([valid_row])

    table = table.append_column(
        "unexpected",
        pa.array([123]),
    )

    with pytest.raises(
        DataValidationError,
        match="OHLCV schema mismatch",
    ):
        validator.validate(table)


def test_incorrect_column_type(validator, valid_row):
    table = make_table([valid_row])

    index = table.schema.get_field_index("volume")

    table = table.set_column(
        index,
        "volume",
        pa.array([1000000.0]),
    )

    with pytest.raises(
        DataValidationError,
        match="OHLCV schema mismatch",
    ):
        validator.validate(table)


def test_invalid_input_type(validator):
    with pytest.raises(
        DataValidationError,
        match="pyarrow.Table",
    ):
        validator.validate(None)


@pytest.mark.parametrize(
    "column",
    ["security_id", "timestamp", "available_at", "ingested_at"],
)
def test_required_field_cannot_be_null(
    validator,
    valid_row,
    column,
):
    valid_row[column] = None

    table = make_table([valid_row])

    with pytest.raises(
        DataValidationError,
        match="Required column",
    ):
        validator.validate(table)
