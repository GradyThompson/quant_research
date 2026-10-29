"""Deterministic orchestration of existing single-security daily ingestion.

Each security is ingested through an injected callable. The caller supplies
point-in-time security resolution, provider access, dataset naming and storage
through the existing DailyBarIngestor. Never silently replace an immutable
version or pretend that failed securities succeeded.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True, order=True)
class IngestionTarget:
    """Unique ticker + listing MIC in the selected *fixed* research universe."""

    ticker: str
    mic: str

    def __post_init__(self) -> None:
        if (
            not self.ticker
            or self.ticker != self.ticker.strip()
            or self.ticker != self.ticker.upper()
        ):
            raise ValueError("ticker must be a nonempty, uppercase, trimmed string")
        if len(self.mic) != 4 or self.mic != self.mic.upper() or not self.mic.isalpha():
            raise ValueError("mic must be a four-letter uppercase MIC")


@dataclass(frozen=True, slots=True)
class IngestionFailure:
    target: IngestionTarget
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class BatchIngestionResult(Generic[T]):
    successful: tuple[tuple[IngestionTarget, T], ...]
    failed: tuple[IngestionFailure, ...]

    @property
    def succeeded(self) -> bool:
        return not self.failed


@dataclass(frozen=True, slots=True)
class BatchDailyIngestor(Generic[T]):
    """Run a single-security ingestion callable for each unique security.

    The callable must actually persist the existing validated OHLCV dataset
    and return its metadata/version. No retries are attempted because retrying
    writes can create extra immutable versions or mask ambiguous failures.
    """

    ingest_one: Callable[[IngestionTarget], T]
    continue_on_error: bool = True

    def ingest(self, targets: Sequence[IngestionTarget]) -> BatchIngestionResult[T]:
        if not targets:
            raise ValueError("universe must contain at least one security")
        if len(set(targets)) != len(targets):
            raise ValueError("duplicate ticker/MIC pairs in universe")
        successful: list[tuple[IngestionTarget, T]] = []
        failed: list[IngestionFailure] = []
        for target in targets:
            try:
                metadata = self.ingest_one(target)
            except Exception as exc:
                if not self.continue_on_error:
                    raise
                failed.append(IngestionFailure(target, type(exc).__name__, str(exc)))
            else:
                successful.append((target, metadata))
        return BatchIngestionResult(tuple(successful), tuple(failed))
