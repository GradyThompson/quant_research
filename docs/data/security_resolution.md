# Historical security resolution — Change 11

## API

Use `SecurityResolver.resolve_ticker(ticker=..., mic=..., valid_at=..., known_at=...)` or
`resolve_identifier(identifier_type=..., identifier_value=..., valid_at=..., known_at=...)`.
All instants must be timezone-aware UTC. Both lookups perform exact matching.

- `valid_at`: market date when the identifier or listing was effective (half-open validity interval).
- `known_at`: historical knowledge cutoff; only assertions recorded by then and not yet superseded are considered.
- Zero matches: `SecurityNotFoundError`.
- Multiple distinct securities: `AmbiguousSecurityError` (no silent priority rules).
- Multiple matching assertions for the **same** security: one unique result.

## Correction semantics

`IdentifierRecord` and `ListingRecord` now have optional `superseded_at` timestamps.
Treat every record as an immutable assertion with knowledge interval
`[recorded_at, superseded_at)`; a historical correction creates a new assertion
and closes the previous assertion's knowledge interval. This is a basic
bitemporal convention, not a fully audited source-of-truth system.

For the prototype, callers must supply complete, internally consistent
assertions. The resolver does not infer supersession automatically, validate
provider identity, prove that an identifier is authentic, or assign permanent
security IDs. In particular, conflict between providers raises ambiguity.

This implementation is in-memory. Persistence, explicit provider policies,
versioned audit events, issuer relationships, and corporate action mappings
will be designed separately. No ticker-only lookup is offered.
