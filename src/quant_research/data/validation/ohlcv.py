"""Validation of canonical OHLCV market data."""

import math
import re
from dataclasses import dataclass
from datetime import datetime

import pyarrow as pa

from quant_research.core import SecurityId
from quant_research.data.schemas import (
    BarFrequency,
    PriceAdjustment,
    VolumeUnit,
    ohlcv_schema,
)
from quant_research.exceptions import DataValidationError

_CURRENCY_PATTERN = re.compile(r"^[A-Z]{3}$")

_PRICE_COLUMNS = ("open", "high", "low", "close")


@dataclass(frozen=True, slots=True)
class OHLCVValidator:
    """Validate an Arrow table against the canonical OHLCV contract.

    The validator never modifies the supplied table.

    Raises:
        DataValidationError:
            If the table violates the expected schema or data rules.
    """

    frequency: BarFrequency
    adjustment: PriceAdjustment = PriceAdjustment.RAW
    volume_unit: VolumeUnit = VolumeUnit.SHARES

    def validate(self, table: pa.Table) -> None:
        """Validate schema, identifiers, values, and timestamps."""

        if not isinstance(table, pa.Table):
            raise DataValidationError("OHLCV validation requires a pyarrow.Table.")

        self._validate_schema(table)
        self._validate_required_fields(table)
        self._validate_rows(table)

    def _validate_schema(self, table: pa.Table) -> None:
        """Require an exact match with the canonical schema."""

        expected = ohlcv_schema(
            frequency=self.frequency,
            adjustment=self.adjustment,
            volume_unit=self.volume_unit,
        )

        if not table.schema.equals(expected, check_metadata=True):
            raise DataValidationError(
                f"OHLCV schema mismatch.\nExpected:\n{expected}\nReceived:\n{table.schema}"
            )

    def _validate_rows(self, table: pa.Table) -> None:
        """Validate individual observations and unique bar keys."""

        seen_keys: set[tuple[str, datetime]] = set()

        for row_number, row in enumerate(table.to_pylist()):
            self._validate_row(row, row_number)

            key = (row["security_id"], row["timestamp"])

            if key in seen_keys:
                raise DataValidationError(
                    "Duplicate OHLCV observation at "
                    f"row {row_number}: "
                    f"security_id={key[0]}, timestamp={key[1]}"
                )

            seen_keys.add(key)

    def _validate_row(
        self,
        row: dict[str, object],
        row_number: int,
    ) -> None:
        """Validate one OHLCV observation."""

        self._validate_security(row, row_number)
        self._validate_currency(row, row_number)
        self._validate_prices(row, row_number)
        self._validate_volume(row, row_number)
        self._validate_timestamps(row, row_number)

    @staticmethod
    def _validate_security(
        row: dict[str, object],
        row_number: int,
    ) -> None:
        value = row["security_id"]

        try:
            SecurityId(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                f"Invalid security_id at row {row_number}: {value!r}"
            ) from exc

    @staticmethod
    def _validate_currency(
        row: dict[str, object],
        row_number: int,
    ) -> None:
        currency = row["currency"]

        if not isinstance(currency, str):
            raise DataValidationError(f"Invalid currency at row {row_number}: {currency!r}")

        if not _CURRENCY_PATTERN.fullmatch(currency):
            raise DataValidationError(f"Invalid currency at row {row_number}: {currency!r}")

    @staticmethod
    def _validate_prices(
        row: dict[str, object],
        row_number: int,
    ) -> None:
        prices: dict[str, float | None] = {}

        for column in _PRICE_COLUMNS:
            value = row[column]

            if value is None:
                prices[column] = None
                continue

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise DataValidationError(f"Invalid {column} price at row {row_number}: {value!r}")

            prices[column] = float(value)

        open_price = prices["open"]
        high = prices["high"]
        low = prices["low"]
        close = prices["close"]

        if high is not None and low is not None and high < low:
            raise DataValidationError(f"High is below low at row {row_number}.")

        if high is not None:
            for name, price in (
                ("open", open_price),
                ("close", close),
            ):
                if price is not None and price > high:
                    raise DataValidationError(f"{name} exceeds high at row {row_number}.")

        if low is not None:
            for name, price in (
                ("open", open_price),
                ("close", close),
            ):
                if price is not None and price < low:
                    raise DataValidationError(f"{name} is below low at row {row_number}.")

    @staticmethod
    def _validate_volume(
        row: dict[str, object],
        row_number: int,
    ) -> None:
        volume = row["volume"]

        if volume is None:
            return

        if isinstance(volume, bool) or not isinstance(volume, int) or volume < 0:
            raise DataValidationError(f"Invalid volume at row {row_number}: {volume!r}")

    @staticmethod
    def _validate_timestamps(
        row: dict[str, object],
        row_number: int,
    ) -> None:
        timestamp = row["timestamp"]
        available_at = row["available_at"]
        ingested_at = row["ingested_at"]

        if (
            not isinstance(timestamp, datetime)
            or not isinstance(available_at, datetime)
            or not isinstance(ingested_at, datetime)
        ):
            raise DataValidationError(f"Invalid timestamps at row {row_number}.")

        if available_at < timestamp:
            raise DataValidationError(f"Availability precedes bar end at row {row_number}.")

        if ingested_at < available_at:
            raise DataValidationError(f"Ingestion precedes availability at row {row_number}.")

    @staticmethod
    def _validate_required_fields(table: pa.Table) -> None:
        """Reject null values in required columns."""

        for field in table.schema:
            if field.nullable:
                continue

            column = table.column(field.name)

            if column.null_count > 0:
                raise DataValidationError(
                    f"Required column {field.name!r} contains {column.null_count} null values."
                )
