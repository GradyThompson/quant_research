"""Contract tests for local Parquet dataset storage."""

from uuid import uuid4

import pyarrow as pa
import pytest

from quant_research.core import DatasetId, DatasetVersion
from quant_research.data.storage.parquet import ParquetDatasetStore
from quant_research.exceptions import DatasetIntegrityError, DatasetNotFoundError


@pytest.fixture
def store(tmp_path):
    return ParquetDatasetStore(tmp_path)


@pytest.fixture
def dataset_id():
    return DatasetId("market.ohlcv.daily")


@pytest.fixture
def table():
    return pa.table({"security_id": ["SEC-000001"], "close": [105.0]})


def test_write_read_roundtrip(store, dataset_id, table):
    metadata = store.write(dataset_id, table)
    assert metadata.dataset_id == dataset_id
    assert metadata.row_count == 1
    assert metadata.created_at.utcoffset() is not None
    assert store.exists(dataset_id, metadata.version)
    assert store.get_metadata(dataset_id, metadata.version) == metadata
    assert store.read(dataset_id, metadata.version).equals(table, check_metadata=True)


def test_two_writes_create_distinct_immutable_versions(store, dataset_id, table):
    first = store.write(dataset_id, table)
    second_table = pa.table({"security_id": ["SEC-000001"], "close": [106.0]})
    second = store.write(dataset_id, second_table)
    assert first.version != second.version
    assert store.read(dataset_id, first.version).equals(table)
    assert store.read(dataset_id, second.version).equals(second_table)
    assert [v.version for v in store.list_versions(dataset_id)] == [first.version, second.version]


def test_unknown_dataset_and_version(store, dataset_id):
    unknown = DatasetVersion(uuid4())
    assert not store.exists(dataset_id, unknown)
    assert store.list_versions(dataset_id) == []
    with pytest.raises(DatasetNotFoundError):
        store.read(dataset_id, unknown)


def test_data_corruption_detected(store, dataset_id, table):
    metadata = store.write(dataset_id, table)
    path = store._version_directory(dataset_id, metadata.version) / "data.parquet"
    path.write_bytes(b"corrupt")
    with pytest.raises(DatasetIntegrityError, match="integrity"):
        store.read(dataset_id, metadata.version)


def test_manifest_corruption_detected(store, dataset_id, table):
    metadata = store.write(dataset_id, table)
    path = store._version_directory(dataset_id, metadata.version) / "manifest.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(DatasetIntegrityError, match="manifest"):
        store.get_metadata(dataset_id, metadata.version)


def test_failed_parquet_write_is_not_published(store, dataset_id, table, monkeypatch):
    from quant_research.data.storage import parquet

    def fail(*args, **kwargs):
        raise OSError("simulated disk write failure")

    monkeypatch.setattr(parquet.pq, "write_table", fail)
    with pytest.raises(OSError, match="simulated"):
        store.write(dataset_id, table)
    assert store.list_versions(dataset_id) == []
    directory = store._dataset_directory(dataset_id) / "versions"
    assert list(directory.iterdir()) == []


def test_empty_dataset_roundtrip(store, dataset_id):
    table = pa.table({"close": pa.array([], type=pa.float64())})
    metadata = store.write(dataset_id, table)
    assert metadata.row_count == 0
    assert store.read(dataset_id, metadata.version).equals(table)


def test_lock_file_created(store, dataset_id, table):
    store.write(dataset_id, table)
    assert (store._dataset_directory(dataset_id) / ".write.lock").exists()


def test_lock_contention_prevents_write(store, dataset_id, table, monkeypatch):
    from filelock import FileLock, Timeout

    from quant_research.data.storage import parquet

    original_lock = parquet.FileLock

    def fast_lock(path, timeout):
        return original_lock(path, timeout=0.1)

    monkeypatch.setattr(parquet, "FileLock", fast_lock)
    lock_path = store._dataset_directory(dataset_id) / ".write.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path), timeout=1):
        with pytest.raises(Timeout):
            store.write(dataset_id, table)
    assert store.list_versions(dataset_id) == []


def test_missing_data_file_is_integrity_error(store, dataset_id, table):
    metadata = store.write(dataset_id, table)
    (store._version_directory(dataset_id, metadata.version) / "data.parquet").unlink()
    with pytest.raises(DatasetIntegrityError):
        store.read(dataset_id, metadata.version)


def test_missing_manifest_is_integrity_error(store, dataset_id, table):
    metadata = store.write(dataset_id, table)
    (store._version_directory(dataset_id, metadata.version) / "manifest.json").unlink()
    with pytest.raises(DatasetIntegrityError):
        store.get_metadata(dataset_id, metadata.version)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_id", "market.other.daily"),
        ("version", "550e8400-e29b-41d4-a716-446655440000"),
        ("manifest_version", 999),
        ("row_count", -1),
        ("schema_fingerprint", ""),
        ("content_hash", ""),
    ],
)
def test_invalid_manifest_field(store, dataset_id, table, field, value):
    import json

    metadata = store.write(dataset_id, table)
    path = store._version_directory(dataset_id, metadata.version) / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetIntegrityError):
        store.get_metadata(dataset_id, metadata.version)


def test_staging_detection_and_exclusion(store, dataset_id):
    versions = store._dataset_directory(dataset_id) / "versions"
    versions.mkdir(parents=True)
    staging = versions / ".staging-orphan"
    staging.mkdir()
    assert store.list_staging_directories(dataset_id) == [staging]
    assert store.list_versions(dataset_id) == []
    assert staging.exists()


def test_failed_write_preserves_existing_version(store, dataset_id, table, monkeypatch):
    from quant_research.data.storage import parquet

    original = store.write(dataset_id, table)

    def fail(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(parquet.pq, "write_table", fail)
    with pytest.raises(OSError, match="simulated write failure"):
        store.write(dataset_id, table)
    assert store.read(dataset_id, original.version).equals(table, check_metadata=True)
    assert [item.version for item in store.list_versions(dataset_id)] == [original.version]
    assert store.list_staging_directories(dataset_id) == []
