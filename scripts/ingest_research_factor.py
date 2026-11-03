"""Ingest an ETF factor for M3's permitted dates only (no holdout access).

Creates a separate version-pinned factor receipt, without modifying stock
research receipts or the common-stock-only historical security master.
Requires Massive ETF reference and unadjusted minute-aggregate permissions.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd

from quant_research.core import DatasetVersion, TimeRange
from quant_research.data.ingestion.minute import MinuteBarIngestor
from quant_research.data.ingestion.minute_window import session_dataset_id
from quant_research.data.providers.massive_minute import MassiveMinuteProvider
from quant_research.data.providers.models import BarRequest, InstrumentKind, ProviderInstrument
from quant_research.data.research_factor import (
    FactorIdentity,
    FactorPin,
    ingest_research_factor,
    receipt_sha256,
    requested_factor_days,
)
from quant_research.data.schemas import BarFrequency
from quant_research.data.storage import ParquetDatasetStore
from quant_research.model.first_passage_validation import select_receipt_window
from quant_research.paths import resolve_project_path
from quant_research.security_master.massive_bootstrap import _security_id
from quant_research.security_master.models import (
    IdentifierRecord,
    IdentifierType,
    ListingRecord,
    SecurityRecord,
    SecurityType,
    ValidityPeriod,
)
from quant_research.security_master.resolver import SecurityResolver


def _get(row: Any, field: str) -> Any:
    return row.get(field) if isinstance(row, dict) else getattr(row, field, None)


def lookup_etf(client: Any, *, day: date, ticker: str, mic: str) -> FactorIdentity:
    """Require exactly one dated ETF/FIGI assertion, not a guessed ticker identity."""
    results = list(
        client.list_tickers(
            ticker=ticker,
            market="stocks",
            type="ETF",
            date=day.isoformat(),
            active=True,
            limit=100,
        )
    )
    matches = [
        row
        for row in results
        if _get(row, "ticker") == ticker
        and _get(row, "primary_exchange") == mic
        and _get(row, "locale") == "us"
        and _get(row, "type") == "ETF"
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {ticker}:{mic} ETF identity for {day}, got {len(matches)}")
    figi = _get(matches[0], "share_class_figi")
    if not isinstance(figi, str) or not figi.strip():
        raise ValueError(f"Missing stable share-class FIGI for {ticker} on {day}")
    return FactorIdentity(ticker, mic, figi, str(_security_id(figi)))


def _dated_resolver(
    day: date, identity: FactorIdentity, *, recorded_at: datetime
) -> SecurityResolver:
    start = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
    validity = ValidityPeriod(start, start + timedelta(days=1))
    sid = _security_id(identity.figi)
    if str(sid) != identity.security_id:
        raise ValueError("Internal security identity and FIGI disagree")
    source = "massive.reference.etf.retrieved_retrospectively"
    return SecurityResolver(
        [SecurityRecord(sid, SecurityType.ETF, recorded_at)],
        [
            IdentifierRecord(
                sid, IdentifierType.SHARE_CLASS_FIGI, identity.figi, validity, recorded_at, source
            )
        ],
        [ListingRecord(sid, identity.ticker, identity.mic, validity, recorded_at, source)],
    )


def run_factor_ingestion(
    *,
    asset_receipt_path: Path,
    factor_receipt_path: Path,
    store: ParquetDatasetStore,
    ticker: str,
    factor_ticker: str,
    factor_mic: str,
    evaluation_start: date,
    max_sessions: int,
    publication_lag_seconds: int,
    dry_run: bool,
) -> dict[str, object]:
    """Select the identical M3 window before any factor lookup or dataset read."""
    raw: dict[str, object] = json.loads(asset_receipt_path.read_text(encoding="utf-8"))
    if ticker == factor_ticker:
        raise ValueError("Factor ticker must differ from stock ticker")
    asset_lag = raw.get("publication_lag_seconds")
    if asset_lag is not None and asset_lag != publication_lag_seconds:
        raise ValueError("Factor and stock publication-lag assumptions must match")
    retained, _, _ = select_receipt_window(raw, ticker=ticker, evaluation_start=evaluation_start)
    allowed = requested_factor_days(retained)

    # Defer network credentials until after --dry-run or previous receipt checks.
    client: Any = None
    provider: MassiveMinuteProvider | None = None
    calendar = xcals.get_calendar("XNYS")

    def _connect() -> tuple[Any, MassiveMinuteProvider]:
        nonlocal client, provider
        if client is None or provider is None:
            import os

            from dotenv import load_dotenv
            from massive import RESTClient

            from quant_research.paths import PROJECT_ROOT

            load_dotenv(PROJECT_ROOT / ".env", override=False)
            api_key = os.environ.get("MASSIVE_API_KEY")
            if not api_key:
                raise ValueError("MASSIVE_API_KEY must be configured for factor ingestion")
            client = RESTClient(api_key=api_key)
            provider = MassiveMinuteProvider(client=client)
        return client, provider

    def lookup(day: date) -> FactorIdentity:
        api, _ = _connect()
        return lookup_etf(api, day=day, ticker=factor_ticker, mic=factor_mic)

    def verify(entry: dict[str, object]) -> None:
        from uuid import UUID

        from quant_research.core import DatasetId

        day = date.fromisoformat(str(entry["session"]))
        expected_dataset_id = session_dataset_id(factor_ticker, day)
        if str(entry["dataset_id"]) != str(expected_dataset_id):
            raise ValueError(f"Wrong dataset id for {day}")
        data = store.read(
            DatasetId(str(entry["dataset_id"])), DatasetVersion(UUID(str(entry["dataset_version"])))
        )
        if data.num_rows != entry["rows"] or data.num_rows == 0:
            raise ValueError(f"Pinned factor readback row mismatch: {day}")
        if set(data.column("security_id").to_pylist()) != {entry["security_id"]}:
            raise ValueError(f"Pinned factor readback identity mismatch: {day}")

    def ingest(day: date, identity: FactorIdentity) -> FactorPin:
        _, minute_provider = _connect()
        now = datetime.now(UTC)
        resolver = _dated_resolver(day, identity, recorded_at=now)
        ingestor = MinuteBarIngestor(
            provider=minute_provider,
            resolver=resolver,
            store=store,
            publication_lag=timedelta(seconds=publication_lag_seconds),
        )
        session = pd.Timestamp(day)
        opening = calendar.session_open(session).to_pydatetime()
        closing = calendar.session_close(session).to_pydatetime()
        request = BarRequest(
            instrument=ProviderInstrument(factor_ticker, InstrumentKind.TICKER, factor_mic),
            frequency=BarFrequency.MINUTE,
            period=TimeRange(opening, closing + timedelta(microseconds=1)),
        )
        dataset_id = session_dataset_id(factor_ticker, day)
        result = ingestor.ingest(request, known_at=now, dataset_id=dataset_id)
        if str(result.security_id) != identity.security_id:
            raise ValueError(f"Factor identity mismatch during ingestion on {day}")
        table = store.read(dataset_id, result.metadata.version)
        return FactorPin(
            dataset_id=str(dataset_id),
            dataset_version=str(result.metadata.version),
            rows=table.num_rows,
            first_bar_end=table.column("timestamp")[0].as_py().isoformat(),
            last_bar_end=table.column("timestamp")[-1].as_py().isoformat(),
        )

    return ingest_research_factor(
        receipt_path=factor_receipt_path,
        asset_receipt_sha256=receipt_sha256(asset_receipt_path),
        permitted_days=allowed,
        ticker=factor_ticker,
        mic=factor_mic,
        publication_lag_seconds=publication_lag_seconds,
        lookup_identity=lookup,
        ingest_session=ingest,
        verify_existing=verify,
        max_sessions=max_sessions,
        dry_run=dry_run,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ticker", required=True, help="Stock receipt used to select research days"
    )
    parser.add_argument("--factor-ticker", default="QQQ")
    parser.add_argument("--factor-mic", default="XNAS", choices=("XNYS", "XNAS"))
    parser.add_argument(
        "--receipt", type=Path, default=Path("data/dev/minute_training_receipt_20250113.json")
    )
    parser.add_argument(
        "--factor-receipt", type=Path, default=Path("data/dev/qqq_m3_factor_receipt_20250113.json")
    )
    parser.add_argument("--store-root", type=Path, default=Path("data/dev"))
    parser.add_argument("--evaluation-start", type=date.fromisoformat, default=date(2025, 1, 13))
    parser.add_argument("--publication-lag-seconds", type=int, default=60)
    parser.add_argument("--max-sessions", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = run_factor_ingestion(
        asset_receipt_path=resolve_project_path(args.receipt),
        factor_receipt_path=resolve_project_path(args.factor_receipt),
        store=ParquetDatasetStore(resolve_project_path(args.store_root)),
        ticker=args.ticker.upper(),
        factor_ticker=args.factor_ticker.upper(),
        factor_mic=args.factor_mic,
        evaluation_start=args.evaluation_start,
        max_sessions=args.max_sessions,
        publication_lag_seconds=args.publication_lag_seconds,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.dry_run:
        print(
            "NOTICE: historical ETF identities retrieved retrospectively; publication timing assumed"
        )


if __name__ == "__main__":
    main()
