# Security master contracts (Change 10)

- `SecurityRecord` identifies an **instrument**, not a company or ticker. `issuer_id` is optional and opaque for now.
- `IdentifierRecord` attaches a typed identifier (ISIN, CUSIP, FIGI, etc.) to a `SecurityId` for a half-open UTC validity period.
- `ListingRecord` attaches a ticker and four-character market identifier code (MIC) to a security for a half-open UTC validity period.
- `recorded_at` marks when the system recorded the assertion, and may be later than the historical validity period; do **not** impose `recorded_at <= valid_to`.
- Records are immutable. Corrections append new assertions rather than modifying prior assertions.
- The models deliberately **do not** guarantee global uniqueness, external identifier validity, or resolve conflicting assertions. That belongs to Change 11 and its persistence/validation rules.
- `recorded_at` alone is insufficient for full bitemporal history: later implement supersession / recorded-until semantics, stable record IDs and source provenance, then explicit conflict-resolution rules. Do not interpret the latest record as historically authoritative without an as-of query.
- `mic` checks format only, not membership in an official MIC registry; ticker syntax is venue-dependent.
- Changes to the scope of a `SecurityId` (instrument vs listing/share class) should be documented before importing historical vendor data.
