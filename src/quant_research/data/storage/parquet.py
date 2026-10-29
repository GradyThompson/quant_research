"""Local immutable, versioned Parquet dataset storage."""

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from tempfile import mkdtemp
from uuid import UUID, uuid4

import pyarrow as pa
import pyarrow.parquet as pq
from filelock import FileLock

from quant_research.core import DatasetId, DatasetVersion
from quant_research.data.storage.base import DatasetStore
from quant_research.data.storage.models import DatasetMetadata
from quant_research.exceptions import DatasetIntegrityError, DatasetNotFoundError


class ParquetDatasetStore(DatasetStore):
    """Local, per-dataset-locked store with immutable published versions.

    Publication uses a staging directory renamed on the same filesystem.
    This provides atomic visibility, not guaranteed crash durability.
    Cooperating local writers serialize publication per dataset.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def _dataset_directory(self, dataset_id: DatasetId) -> Path:
        return self.root / "datasets" / Path(*dataset_id.value.split("."))

    def _version_directory(self, dataset_id: DatasetId, version: DatasetVersion) -> Path:
        return self._dataset_directory(dataset_id) / "versions" / str(version)

    @staticmethod
    def _digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _schema_digest(schema: pa.Schema) -> str:
        return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()

    @staticmethod
    def _manifest(metadata: DatasetMetadata) -> dict[str, object]:
        return {
            "manifest_version": 1,
            "dataset_id": str(metadata.dataset_id),
            "version": str(metadata.version),
            "created_at": metadata.created_at.isoformat(),
            "row_count": metadata.row_count,
            "schema_fingerprint": metadata.schema_fingerprint,
            "content_hash": metadata.content_hash,
        }

    def _dataset_lock(self, dataset_id: DatasetId) -> FileLock:
        """Serialize cooperating writers for one dataset on a local filesystem."""
        directory = self._dataset_directory(dataset_id)
        directory.mkdir(parents=True, exist_ok=True)
        return FileLock(str(directory / ".write.lock"), timeout=30)

    def list_staging_directories(self, dataset_id: DatasetId) -> list[Path]:
        """Report unpublished staging directories without modifying them."""
        versions = self._dataset_directory(dataset_id) / "versions"
        if not versions.exists():
            return []
        return sorted(
            path
            for path in versions.iterdir()
            if path.is_dir() and path.name.startswith(".staging-")
        )

    def write(self, dataset_id: DatasetId, table: pa.Table) -> DatasetMetadata:
        """Publish a new dataset version under the per-dataset lock."""
        with self._dataset_lock(dataset_id):
            return self._write_unlocked(dataset_id, table)

    def _write_unlocked(self, dataset_id: DatasetId, table: pa.Table) -> DatasetMetadata:
        """Publish a dataset version; caller must hold the write lock."""
        if not isinstance(table, pa.Table):
            raise TypeError("table must be a pyarrow.Table")
        # Domain-specific validators belong to ingestion pipelines; storage
        # requires an Arrow table, and verifies its persistence integrity.
        versions_dir = self._dataset_directory(dataset_id) / "versions"
        versions_dir.mkdir(parents=True, exist_ok=True)
        staging = Path(mkdtemp(prefix=".staging-", dir=versions_dir))
        try:
            version = DatasetVersion(uuid4())
            final = self._version_directory(dataset_id, version)
            data_file = staging / "data.parquet"
            pq.write_table(table, data_file)
            metadata = DatasetMetadata(
                dataset_id=dataset_id,
                version=version,
                created_at=datetime.now(UTC),
                row_count=table.num_rows,
                schema_fingerprint=self._schema_digest(table.schema),
                content_hash=self._digest(data_file),
            )
            (staging / "manifest.json").write_text(
                json.dumps(self._manifest(metadata), sort_keys=True, indent=2),
                encoding="utf-8",
            )
            # Atomic visibility on a local filesystem (same directory).
            # The final path is never opened for writing.
            staging.rename(final)
            return metadata
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def get_metadata(self, dataset_id: DatasetId, version: DatasetVersion) -> DatasetMetadata:
        directory = self._version_directory(dataset_id, version)
        if not directory.is_dir():
            raise DatasetNotFoundError(f"Dataset version not found: {dataset_id}/{version}")
        try:
            raw = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            if raw["manifest_version"] != 1:
                raise ValueError("Unsupported manifest version")
            metadata = DatasetMetadata(
                dataset_id=DatasetId(raw["dataset_id"]),
                version=DatasetVersion(UUID(raw["version"])),
                created_at=datetime.fromisoformat(raw["created_at"]),
                row_count=raw["row_count"],
                schema_fingerprint=raw["schema_fingerprint"],
                content_hash=raw["content_hash"],
            )
            if metadata.dataset_id != dataset_id or metadata.version != version:
                raise ValueError("Manifest identity mismatch")
            return metadata
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            raise DatasetIntegrityError(f"Invalid manifest: {directory}") from exc

    def read(self, dataset_id: DatasetId, version: DatasetVersion) -> pa.Table:
        metadata = self.get_metadata(dataset_id, version)
        data_file = self._version_directory(dataset_id, version) / "data.parquet"
        try:
            if self._digest(data_file) != metadata.content_hash:
                raise ValueError("Parquet file checksum mismatch")
            table = pq.read_table(data_file)
            if table.num_rows != metadata.row_count:
                raise ValueError("Row count mismatch")
            if self._schema_digest(table.schema) != metadata.schema_fingerprint:
                raise ValueError("Schema fingerprint mismatch")
            return table
        except (OSError, ValueError, pa.ArrowException) as exc:
            raise DatasetIntegrityError(
                f"Dataset integrity check failed: {dataset_id}/{version}"
            ) from exc

    def list_versions(self, dataset_id: DatasetId) -> list[DatasetMetadata]:
        directory = self._dataset_directory(dataset_id) / "versions"
        if not directory.exists():
            return []
        result = []
        for path in directory.iterdir():
            if not path.is_dir() or path.name.startswith(".staging-"):
                continue
            try:
                version = DatasetVersion(UUID(path.name))
            except ValueError as exc:
                raise DatasetIntegrityError(f"Unexpected version directory: {path}") from exc
            result.append(self.get_metadata(dataset_id, version))
        return sorted(result, key=lambda item: (item.created_at, str(item.version)))

    def exists(self, dataset_id: DatasetId, version: DatasetVersion) -> bool:
        return self._version_directory(dataset_id, version).is_dir()
