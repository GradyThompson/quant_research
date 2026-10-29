"""Metadata contracts for versioned datasets."""

from dataclasses import dataclass
from datetime import datetime

from quant_research.core import DatasetId, DatasetVersion


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """Metadata describing one published dataset version.

    A published dataset version is immutable.

    The storage implementation is responsible for ensuring
    that this metadata corresponds to the stored content.
    """

    dataset_id: DatasetId
    version: DatasetVersion
    created_at: datetime
    row_count: int
    schema_fingerprint: str
    content_hash: str

    def __post_init__(self) -> None:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware.")

        if self.row_count < 0:
            raise ValueError("row_count cannot be negative.")

        if not self.schema_fingerprint:
            raise ValueError("schema_fingerprint cannot be empty.")

        if not self.content_hash:
            raise ValueError("content_hash cannot be empty.")
