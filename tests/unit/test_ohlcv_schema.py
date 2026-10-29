"""Tests for canonical OHLCV schema definitions."""

from datetime import UTC, datetime

import pyarrow as pa
import pytest

from quant_research.data.schemas import (
    BarFrequency,
    PriceAdjustment,
    VolumeUnit,
    ohlcv_schema,
)


def test_daily_ohlcv_schema():
    schema = ohlcv_schema(
        frequency=BarFrequency.DAILY,
    )

    assert isinstance(schema, pa.Schema)

    assert schema.names == [
        "security_id",
        "timestamp",
        "available_at",
        "ingested_at",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "currency",
    ]


def test_column_types():
    schema = ohlcv_schema(
        frequency=BarFrequency.DAILY,
    )

    assert schema.field("security_id").type == pa.string()

    assert schema.field("timestamp").type == pa.timestamp(
        "us",
        tz="UTC",
    )

    assert schema.field("available_at").type == pa.timestamp(
        "us",
        tz="UTC",
    )

    assert schema.field("ingested_at").type == pa.timestamp(
        "us",
        tz="UTC",
    )

    for column in ["open", "high", "low", "close"]:
        assert schema.field(column).type == pa.float64()

    assert schema.field("volume").type == pa.int64()
    assert schema.field("currency").type == pa.string()


def test_required_fields():
    schema = ohlcv_schema(
        frequency=BarFrequency.DAILY,
    )

    required = [
        "security_id",
        "timestamp",
        "available_at",
        "ingested_at",
        "currency",
    ]

    for column in required:
        assert not schema.field(column).nullable


def test_nullable_market_data_fields():
    schema = ohlcv_schema(
        frequency=BarFrequency.DAILY,
    )

    for column in ["open", "high", "low", "close", "volume"]:
        assert schema.field(column).nullable


def test_default_schema_metadata():
    schema = ohlcv_schema(
        frequency=BarFrequency.DAILY,
    )

    assert schema.metadata is not None

    assert schema.metadata[b"schema_name"] == b"ohlcv"
    assert schema.metadata[b"schema_version"] == b"1"
    assert schema.metadata[b"frequency"] == b"1d"
    assert schema.metadata[b"price_adjustment"] == b"raw"
    assert schema.metadata[b"volume_unit"] == b"shares"
    assert schema.metadata[b"timestamp_semantics"] == b"bar_end_utc"


def test_minute_frequency():
    schema = ohlcv_schema(
        frequency=BarFrequency.MINUTE,
    )

    assert schema.metadata is not None
    assert schema.metadata[b"frequency"] == b"1m"


def test_split_adjusted_schema():
    schema = ohlcv_schema(
        frequency=BarFrequency.DAILY,
        adjustment=PriceAdjustment.SPLIT_ADJUSTED,
    )

    assert schema.metadata is not None
    assert schema.metadata[b"price_adjustment"] == b"split_adjusted"


def test_contract_volume_unit():
    schema = ohlcv_schema(
        frequency=BarFrequency.MINUTE,
        volume_unit=VolumeUnit.CONTRACTS,
    )

    assert schema.metadata is not None
    assert schema.metadata[b"volume_unit"] == b"contracts"


def test_schema_can_construct_arrow_table():
    schema = ohlcv_schema(
        frequency=BarFrequency.DAILY,
    )

    table = pa.Table.from_pylist(
        [
            {
                "security_id": "SEC-000001",
                "timestamp": datetime(2025, 1, 2, 21, 0, tzinfo=UTC),
                "available_at": datetime(2025, 1, 2, 21, 0, 5, tzinfo=UTC),
                "ingested_at": datetime(2025, 1, 2, 21, 0, 6, tzinfo=UTC),
                "open": 185.50,
                "high": 188.20,
                "low": 184.90,
                "close": 187.30,
                "volume": 52000000,
                "currency": "USD",
            }
        ],
        schema=schema,
    )

    assert table.num_rows == 1
    assert table.schema.equals(schema, check_metadata=True)

    assert table.column("close")[0].as_py() == pytest.approx(187.30)


def test_arrow_schema_does_not_validate_price_relationships():
    """Document that Arrow types alone cannot validate OHLC prices."""

    schema = ohlcv_schema(
        frequency=BarFrequency.DAILY,
    )

    table = pa.Table.from_pylist(
        [
            {
                "security_id": "SEC-000001",
                "timestamp": datetime(2025, 1, 2, 21, 0, tzinfo=UTC),
                "available_at": datetime(2025, 1, 2, 21, 0, 5, tzinfo=UTC),
                "ingested_at": datetime(2025, 1, 2, 21, 0, 6, tzinfo=UTC),
                "open": 185.50,
                "high": 100.00,
                "low": 200.00,
                "close": 187.30,
                "volume": 52000000,
                "currency": "USD",
            }
        ],
        schema=schema,
    )

    # Arrow accepts these values because the column types are valid.
    # Our next change will introduce numerical data validation.
    assert table.num_rows == 1
