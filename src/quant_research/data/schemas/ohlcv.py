"""Schema definitions for raw OHLCV market data."""

from enum import StrEnum

import pyarrow as pa


class BarFrequency(StrEnum):
    """Supported market data bar frequencies."""

    DAILY = "1d"
    MINUTE = "1m"


class PriceAdjustment(StrEnum):
    """Price adjustment conventions."""

    RAW = "raw"
    SPLIT_ADJUSTED = "split_adjusted"
    TOTAL_RETURN_ADJUSTED = "total_return_adjusted"


class VolumeUnit(StrEnum):
    """Supported volume measurement units."""

    SHARES = "shares"
    CONTRACTS = "contracts"


def ohlcv_schema(
    *,
    frequency: BarFrequency,
    adjustment: PriceAdjustment = PriceAdjustment.RAW,
    volume_unit: VolumeUnit = VolumeUnit.SHARES,
) -> pa.Schema:
    """Return the canonical OHLCV Arrow schema.

    Timestamp conventions:
        timestamp:
            UTC end of the observation interval.

        available_at:
            Earliest timestamp at which the completed bar could
            be used by the research decision process.

        ingested_at:
            Timestamp when this record was ingested.

    Price conventions:
        OHLC prices are denominated in the specified currency.

        Prices use the adjustment method declared in the
        schema metadata.

    Volume conventions:
        Volume is a nonnegative integer count in the specified
        volume unit.

    This function defines structure and metadata only.
    It does not validate numerical relationships.
    """

    fields: list[pa.Field] = [
            pa.field(
                "security_id",
                pa.string(),
                nullable=False,
            ),
            pa.field(
                "timestamp",
                pa.timestamp("us", tz="UTC"),
                nullable=False,
            ),
            pa.field(
                "available_at",
                pa.timestamp("us", tz="UTC"),
                nullable=False,
            ),
            pa.field(
                "ingested_at",
                pa.timestamp("us", tz="UTC"),
                nullable=False,
            ),
            pa.field(
                "open",
                pa.float64(),
                nullable=True,
            ),
            pa.field(
                "high",
                pa.float64(),
                nullable=True,
            ),
            pa.field(
                "low",
                pa.float64(),
                nullable=True,
            ),
            pa.field(
                "close",
                pa.float64(),
                nullable=True,
            ),
            pa.field(
                "volume",
                pa.int64(),
                nullable=True,
            ),
            pa.field(
                "currency",
                pa.string(),
                nullable=False,
            ),
        ]
    return pa.schema(
        fields,
            metadata={
                b"schema_name": b"ohlcv",
                b"schema_version": b"1",
                b"frequency": frequency.value.encode(),
                b"price_adjustment": adjustment.value.encode(),
                b"volume_unit": volume_unit.value.encode(),
                b"timestamp_semantics": b"bar_end_utc",
            },
        )