# Local dataset storage contract

The local `ParquetDatasetStore` publishes complete immutable dataset snapshots. Each
version has `data.parquet` and `manifest.json`. Research must pin the dataset ID
and version rather than implicitly using the newest version.

## Reliability guarantees

- Published version directories are never deliberately overwritten through the API.
- SHA-256 checksums, schema fingerprints, row counts, and manifest identity are
  verified on reads. Invalid manifests and missing constituent files are errors.
- Publication stages files within the destination filesystem, then renames the
  completed staging directory to a UUID-named version directory. This provides
  atomic visibility on local filesystems that support atomic directory renames.
- Per-dataset `.write.lock` files serialize *cooperating* writers on a local
  filesystem. Do not delete lock files during operation. Locks do not protect
  against external mutation or guarantee behavior on every network filesystem.
- An ordinary Python write failure attempts to remove the staging directory and
  does not modify previously published versions.
- Interrupted processes may leave `.staging-*` directories. They are excluded
  from version listings; `list_staging_directories()` reports them without
  deleting them. Do not delete these until confirming no writer is active.
- A missing version directory raises `DatasetNotFoundError`; a published version
  missing a manifest or data file raises `DatasetIntegrityError`.

## Not yet guaranteed

- Power-loss durability (`fsync` of files/directories is not implemented).
- Distributed locking, cross-machine coordination, and network filesystem support.
- Protection from out-of-band edits to published Parquet files and manifests.
- Partition-level updates, garbage collection, retention, and data lineage.
- Domain-level market data validation: validate before writing and after reading.

## Migration note

For this change install `filelock` using `uv add filelock` before running tests.
