from datetime import UTC, datetime, timedelta

import pyarrow as pa

from quant_research.config import load_config
from quant_research.core import DatasetId
from quant_research.data.schemas import BarFrequency, ohlcv_schema
from quant_research.data.storage import ParquetDatasetStore
from quant_research.data.validation import OHLCVValidator


def main() -> None:
    config = load_config("development")

    store = ParquetDatasetStore(root=config.data.path)

    timestamp = datetime(2025, 1, 2, 21, 0, tzinfo=UTC)

    table = pa.Table.from_pylist(
        [
            {
                "security_id": "SEC-000001",
                "timestamp": timestamp,
                "available_at": timestamp + timedelta(seconds=5),
                "ingested_at": timestamp + timedelta(seconds=10),
                "open": 100.0,
                "high": 105.0,
                "low": 98.0,
                "close": 103.0,
                "volume": 1000000,
                "currency": "USD",
            }
        ],
        schema=ohlcv_schema(frequency=BarFrequency.DAILY),
    )

    validator = OHLCVValidator(frequency=BarFrequency.DAILY)

    validator.validate(table)

    dataset_id = DatasetId("market.ohlcv.daily")

    metadata = store.write(
        dataset_id=dataset_id,
        table=table,
    )

    print(f"Dataset: {metadata.dataset_id}")
    print(f"Version: {metadata.version}")
    print(f"Rows: {metadata.row_count}")

    loaded = store.read(
        dataset_id=dataset_id,
        version=metadata.version,
    )

    assert loaded.equals(table, check_metadata=True)

    validator.validate(loaded)

    print("Storage round trip successful")


if __name__ == "__main__":
    main()
