"""Abstract interface for immutable dataset storage."""

from abc import ABC, abstractmethod

import pyarrow as pa

from quant_research.core import DatasetId, DatasetVersion
from quant_research.data.storage.models import DatasetMetadata


class DatasetStore(ABC):
    """Abstract interface for versioned dataset storage.

    Implementations must provide immutable dataset versions.

    Once a version is successfully published, subsequent
    reads of that version must return the same dataset.

    Implementations must not silently overwrite existing
    versions.
    """

    @abstractmethod
    def write(
        self,
        dataset_id: DatasetId,
        table: pa.Table,
    ) -> DatasetMetadata:
        """Publish a new immutable dataset version.

        Requirements:
            - Validate the input table.
            - Create a new dataset version.
            - Publish data and metadata atomically.
            - Never overwrite an existing published version.
            - Return metadata for the published version.
        """

    @abstractmethod
    def read(
        self,
        dataset_id: DatasetId,
        version: DatasetVersion,
    ) -> pa.Table:
        """Read an exact published dataset version.

        Requirements:
            - Retrieve only the requested version.
            - Verify stored content integrity.
            - Raise DatasetNotFoundError when absent.
            - Raise DatasetIntegrityError when corrupted.
        """

    @abstractmethod
    def get_metadata(
        self,
        dataset_id: DatasetId,
        version: DatasetVersion,
    ) -> DatasetMetadata:
        """Retrieve metadata for an exact dataset version."""

    @abstractmethod
    def list_versions(
        self,
        dataset_id: DatasetId,
    ) -> list[DatasetMetadata]:
        """List published versions of a logical dataset.

        Results must be ordered by creation time,
        with deterministic tie-breaking.
        """

    @abstractmethod
    def exists(
        self,
        dataset_id: DatasetId,
        version: DatasetVersion,
    ) -> bool:
        """Return whether the exact published version exists."""
