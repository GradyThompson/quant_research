"""Versioned security-master snapshots backed by the existing DatasetStore.

A manifest dataset acts as the final commit record. Never discover snapshots by
independently selecting the latest version of each component dataset.
"""

import json
from dataclasses import dataclass
from uuid import UUID

import pyarrow as pa

from quant_research.core import DatasetId, DatasetVersion, SecurityId
from quant_research.data.storage import DatasetStore
from quant_research.security_master.models import (
    IdentifierRecord,
    IdentifierType,
    ListingRecord,
    SecurityRecord,
    SecurityType,
    ValidityPeriod,
)
from quant_research.security_master.resolver import SecurityResolver

_SECURITIES = DatasetId("reference.security_master.securities")
_IDENTIFIERS = DatasetId("reference.security_master.identifiers")
_LISTINGS = DatasetId("reference.security_master.listings")
_SNAPSHOTS = DatasetId("reference.security_master.snapshots")
_TIME = pa.timestamp("us", tz="UTC")

_SECURITY_FIELDS: list[pa.Field] = [
    pa.field("security_id", pa.string(), nullable=False),
    pa.field("security_type", pa.string(), nullable=False),
    pa.field("created_at", _TIME, nullable=False),
    pa.field("issuer_id", pa.string()),
]

_IDENTIFIER_FIELDS: list[pa.Field] = [
    pa.field("security_id", pa.string(), nullable=False),
    pa.field("identifier_type", pa.string(), nullable=False),
    pa.field("identifier_value", pa.string(), nullable=False),
    pa.field("valid_from", _TIME, nullable=False),
    pa.field("valid_to", _TIME),
    pa.field("recorded_at", _TIME, nullable=False),
    pa.field("source", pa.string(), nullable=False),
    pa.field("superseded_at", _TIME),
]

_LISTING_FIELDS: list[pa.Field] = [
    pa.field("security_id", pa.string(), nullable=False),
    pa.field("ticker", pa.string(), nullable=False),
    pa.field("mic", pa.string(), nullable=False),
    pa.field("valid_from", _TIME, nullable=False),
    pa.field("valid_to", _TIME),
    pa.field("recorded_at", _TIME, nullable=False),
    pa.field("source", pa.string(), nullable=False),
    pa.field("superseded_at", _TIME),
]

_SECURITIES_SCHEMA = pa.schema(_SECURITY_FIELDS)
_IDENTIFIERS_SCHEMA = pa.schema(_IDENTIFIER_FIELDS)
_LISTINGS_SCHEMA = pa.schema(_LISTING_FIELDS)

_SNAPSHOT_SCHEMA = pa.schema([pa.field("payload", pa.string(), nullable=False)])


@dataclass(frozen=True, slots=True)
class SecurityMasterSnapshot:
    """A committed snapshot reference; store all component versions together."""

    version: DatasetVersion
    securities_version: DatasetVersion
    identifiers_version: DatasetVersion
    listings_version: DatasetVersion


class SecurityMasterRepository:
    """Persist complete security-master snapshots using a versioned DatasetStore.

    A snapshot is published only after all three component versions are written.
    Component versions from an interrupted save may remain orphaned, but are
    never visible through `list_snapshots`. Concurrent save transactions are
    permitted as independent snapshots; there is no mutable 'latest' pointer.
    """

    def __init__(self, store: DatasetStore) -> None:
        self.store = store

    def save(self, resolver: SecurityResolver) -> SecurityMasterSnapshot:
        """Publish a complete immutable snapshot, returning its commit version."""
        securities = pa.Table.from_pylist(
            [
                dict(
                    security_id=str(s.security_id),
                    security_type=s.security_type.value,
                    created_at=s.created_at,
                    issuer_id=s.issuer_id,
                )
                for s in resolver.securities
            ],
            schema=_SECURITIES_SCHEMA,
        )
        identifiers = pa.Table.from_pylist(
            [
                dict(
                    security_id=str(r.security_id),
                    identifier_type=r.identifier_type.value,
                    identifier_value=r.identifier_value,
                    valid_from=r.validity.valid_from,
                    valid_to=r.validity.valid_to,
                    recorded_at=r.recorded_at,
                    source=r.source,
                    superseded_at=r.superseded_at,
                )
                for r in resolver.identifiers
            ],
            schema=_IDENTIFIERS_SCHEMA,
        )
        listings = pa.Table.from_pylist(
            [
                dict(
                    security_id=str(r.security_id),
                    ticker=r.ticker,
                    mic=r.mic,
                    valid_from=r.validity.valid_from,
                    valid_to=r.validity.valid_to,
                    recorded_at=r.recorded_at,
                    source=r.source,
                    superseded_at=r.superseded_at,
                )
                for r in resolver.listings
            ],
            schema=_LISTINGS_SCHEMA,
        )
        a = self.store.write(_SECURITIES, securities)
        b = self.store.write(_IDENTIFIERS, identifiers)
        c = self.store.write(_LISTINGS, listings)
        payload = json.dumps(
            {
                "format": 1,
                "securities": str(a.version),
                "identifiers": str(b.version),
                "listings": str(c.version),
            },
            sort_keys=True,
        )
        committed = self.store.write(
            _SNAPSHOTS,
            pa.Table.from_pylist([{"payload": payload}], schema=_SNAPSHOT_SCHEMA),
        )
        return SecurityMasterSnapshot(committed.version, a.version, b.version, c.version)

    def get_snapshot(self, version: DatasetVersion) -> SecurityMasterSnapshot:
        """Read the final commit record, not independently latest components."""
        table = self.store.read(_SNAPSHOTS, version)
        if not table.schema.equals(_SNAPSHOT_SCHEMA, check_metadata=True) or table.num_rows != 1:
            raise ValueError("Invalid security-master snapshot manifest")
        try:
            raw = json.loads(table.column("payload")[0].as_py())
            if (
                set(raw) != {"format", "securities", "identifiers", "listings"}
                or raw["format"] != 1
            ):
                raise ValueError("Invalid security-master snapshot format")
            return SecurityMasterSnapshot(
                version,
                DatasetVersion(UUID(raw["securities"])),
                DatasetVersion(UUID(raw["identifiers"])),
                DatasetVersion(UUID(raw["listings"])),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid security-master snapshot manifest") from exc

    def list_snapshots(self) -> list[SecurityMasterSnapshot]:
        """Only snapshots with published commit manifests are returned."""
        return [self.get_snapshot(meta.version) for meta in self.store.list_versions(_SNAPSHOTS)]

    def load(self, version: DatasetVersion) -> SecurityResolver:
        """Rebuild and validate the immutable resolver from an exact snapshot."""
        snapshot = self.get_snapshot(version)
        securities = self.store.read(_SECURITIES, snapshot.securities_version)
        identifiers = self.store.read(_IDENTIFIERS, snapshot.identifiers_version)
        listings = self.store.read(_LISTINGS, snapshot.listings_version)
        for table, expected in (
            (securities, _SECURITIES_SCHEMA),
            (identifiers, _IDENTIFIERS_SCHEMA),
            (listings, _LISTINGS_SCHEMA),
        ):
            if not table.schema.equals(expected, check_metadata=True):
                raise ValueError("Security-master component schema mismatch")
        security_records = [
            SecurityRecord(
                SecurityId(r["security_id"]),
                SecurityType(r["security_type"]),
                r["created_at"],
                r["issuer_id"],
            )
            for r in securities.to_pylist()
        ]
        identifier_records = [
            IdentifierRecord(
                SecurityId(r["security_id"]),
                IdentifierType(r["identifier_type"]),
                r["identifier_value"],
                ValidityPeriod(r["valid_from"], r["valid_to"]),
                r["recorded_at"],
                r["source"],
                r["superseded_at"],
            )
            for r in identifiers.to_pylist()
        ]
        listing_records = [
            ListingRecord(
                SecurityId(r["security_id"]),
                r["ticker"],
                r["mic"],
                ValidityPeriod(r["valid_from"], r["valid_to"]),
                r["recorded_at"],
                r["source"],
                r["superseded_at"],
            )
            for r in listings.to_pylist()
        ]
        return SecurityResolver(security_records, identifier_records, listing_records)
