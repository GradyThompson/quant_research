"""Tests for dataset storage metadata contracts."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from quant_research.core import DatasetId, DatasetVersion
from quant_research.data.storage import DatasetMetadata


def make_metadata(
    *,
    row_count: int = 100,
    schema_fingerprint: str = "schema123",
    content_hash: str = "hash123",
    created_at: datetime | None = None,
) -> DatasetMetadata:
    if created_at is None:
        created_at = datetime(2025, 1, 1, tzinfo=UTC)

    return DatasetMetadata(
        dataset_id=DatasetId("market.ohlcv.daily"),
        version=DatasetVersion(uuid4()),
        created_at=created_at,
        row_count=row_count,
        schema_fingerprint=schema_fingerprint,
        content_hash=content_hash,
    )


def test_valid_metadata():
    metadata = make_metadata()

    assert metadata.row_count == 100
    assert metadata.schema_fingerprint == "schema123"
    assert metadata.content_hash == "hash123"


def test_zero_rows_allowed():
    metadata = make_metadata(row_count=0)

    assert metadata.row_count == 0


def test_negative_row_count_rejected():
    with pytest.raises(ValueError, match="row_count"):
        make_metadata(row_count=-1)


def test_empty_schema_fingerprint_rejected():
    with pytest.raises(ValueError, match="schema_fingerprint"):
        make_metadata(schema_fingerprint="")


def test_empty_content_hash_rejected():
    with pytest.raises(ValueError, match="content_hash"):
        make_metadata(content_hash="")


def test_naive_created_at_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        make_metadata(
            created_at=datetime(2025, 1, 1),
        )


def test_metadata_is_immutable():
    metadata = make_metadata()

    with pytest.raises(FrozenInstanceError):
        metadata.row_count = 200
