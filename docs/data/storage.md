# Dataset Storage

## Overview

The storage layer provides versioned, immutable datasets to support reproducible quantitative research. Research components interact with storage through the `DatasetStore` interface, independently of the underlying storage technology.

The initial backend uses local Parquet files.

## Design Principles

1. **Immutability:** Published dataset versions cannot be modified. Corrections create new versions.
2. **Reproducibility:** Research experiments reference exact dataset versions.
3. **Integrity:** SHA-256 checksums and schema fingerprints detect unexpected changes.
4. **Atomicity:** Incomplete writes must never appear as published versions.
5. **Abstraction:** Research code must not depend on physical storage paths or formats.

## Core Contracts

### DatasetMetadata

| Field | Description |
|---|---|
| `dataset_id` | Logical dataset identifier |
| `version` | Immutable UUID-based version |
| `created_at` | Timezone-aware creation timestamp |
| `row_count` | Number of observations |
| `schema_fingerprint` | Hash identifying the schema |
| `content_hash` | SHA-256 checksum of stored content |

### DatasetStore

| Method | Responsibility |
|---|---|
| `write(dataset_id, table)` | Publish a new immutable version |
| `read(dataset_id, version)` | Retrieve and verify an exact version |
| `get_metadata(dataset_id, version)` | Retrieve version metadata |
| `list_versions(dataset_id)` | List published versions deterministically |
| `exists(dataset_id, version)` | Check whether a version exists |

## Storage Layout

```text
data/datasets/
└── market/ohlcv/daily/
    └── versions/
        └── <version_id>/
            ├── data.parquet
            └── manifest.json
```

Each version contains its data and metadata.

## Publication Process

1. Validate the input dataset.
2. Generate a unique version ID.
3. Write Parquet and metadata to a temporary directory.
4. Calculate SHA-256 and schema fingerprints.
5. Verify staged files.
6. Atomically publish the completed version.

Failed writes must not affect existing versions. Atomic visibility does not guarantee crash durability.

## Error Handling

| Exception | Meaning |
|---|---|
| `DataValidationError` | Invalid input data |
| `DatasetNotFoundError` | Requested version does not exist |
| `DatasetIntegrityError` | Stored data is corrupted or inconsistent |

## Invariants

- Published versions are immutable.
- Reads require explicit versions.
- Published data and metadata must be complete.
- Checksums and schema fingerprints must be verifiable.
- Incomplete writes must remain unpublished.
- Storage implementations must satisfy the same interface contract.

## Future Extensions

Partitioned storage, incremental updates, lineage tracking, metadata indexing, concurrent writes, and cloud storage will be introduced only when required.