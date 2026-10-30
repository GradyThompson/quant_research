"""Conservative pilot: bootstrap a US-stock security master from Massive.

Endpoint checks do not establish complete historical listing continuity.
The caller must explicitly attest to the requested historical interval.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol

from quant_research.core import SecurityId
from quant_research.security_master.models import (
    IdentifierRecord,
    IdentifierType,
    ListingRecord,
    SecurityRecord,
    SecurityType,
    ValidityPeriod,
)
from quant_research.security_master.resolver import SecurityResolver


class TickerClient(Protocol):
    def list_tickers(self, **kwargs: Any) -> Iterable[Any]: ...


def _get(record: Any, name: str) -> Any:
    return record.get(name) if isinstance(record, dict) else getattr(record, name, None)


def _lookup(client: TickerClient, ticker: str, mic: str, day: date) -> Any:
    results = list(
        client.list_tickers(
            ticker=ticker, market="stocks", date=day.isoformat(), active=True, limit=100
        )
    )
    matches = [
        row
        for row in results
        if _get(row, "ticker") == ticker
        and _get(row, "primary_exchange") == mic
        and _get(row, "locale") == "us"
        and _get(row, "type") == "CS"
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one active US common stock: {ticker}:{mic} on {day}; found {len(matches)}"
        )
    if not _get(matches[0], "share_class_figi"):
        raise ValueError(f"Missing share-class FIGI for {ticker}:{mic} on {day}")
    return matches[0]


def _security_id(figi: str) -> SecurityId:
    # Stable across runs without depending on ticker spelling or query order.
    number = int.from_bytes(hashlib.sha256(figi.encode()).digest()[:8], "big")
    return SecurityId(f"SEC-{number:020d}")


def bootstrap_massive_security_master(
    client: TickerClient,
    targets: Iterable[tuple[str, str]],
    *,
    valid_from: date,
    valid_through: date,
    recorded_at: datetime,
    historical_continuity_confirmed: bool = False,
) -> SecurityResolver:
    """Build a bounded snapshot after verifying identifiers at two endpoints.

    `historical_continuity_confirmed` is an explicit user assertion that
    there were no intervening ticker/listing changes in this interval.
    Endpoint checks alone cannot prove this.
    """
    if not historical_continuity_confirmed:
        raise ValueError("Explicit historical listing continuity confirmation required")
    if valid_through < valid_from:
        raise ValueError("valid_through must not precede valid_from")
    if recorded_at.tzinfo is None or recorded_at.utcoffset() != timedelta(0):
        raise ValueError("recorded_at must be timezone-aware UTC")
    pairs = tuple(targets)
    if not pairs or len(set(pairs)) != len(pairs):
        raise ValueError("targets must be nonempty and unique")
    securities: list[SecurityRecord] = []
    identifiers: list[IdentifierRecord] = []
    listings: list[ListingRecord] = []
    seen_figis: set[str] = set()
    seen_security_ids: set[SecurityId] = set()
    start = datetime.combine(valid_from, datetime.min.time(), UTC)
    end = datetime.combine(valid_through + timedelta(days=1), datetime.min.time(), UTC)
    interval = ValidityPeriod(start, end)
    for ticker, mic in pairs:
        first = _lookup(client, ticker, mic, valid_from)
        last = _lookup(client, ticker, mic, valid_through)
        first_figi = str(_get(first, "share_class_figi"))
        last_figi = str(_get(last, "share_class_figi"))
        if first_figi != last_figi:
            raise ValueError(f"Identity changed over requested interval for {ticker}:{mic}")
        if first_figi in seen_figis:
            raise ValueError(f"Repeated share-class FIGI in target universe: {first_figi}")
        seen_figis.add(first_figi)
        sid = _security_id(first_figi)
        if sid in seen_security_ids:
            raise ValueError("Security ID hash collision")
        seen_security_ids.add(sid)
        securities.append(SecurityRecord(sid, SecurityType.COMMON_STOCK, recorded_at))
        identifiers.append(
            IdentifierRecord(
                sid,
                IdentifierType.SHARE_CLASS_FIGI,
                first_figi,
                interval,
                recorded_at,
                "massive.reference.tickers",
            )
        )
        listings.append(
            ListingRecord(sid, ticker, mic, interval, recorded_at, "massive.reference.tickers")
        )
    return SecurityResolver(securities, identifiers, listings)
