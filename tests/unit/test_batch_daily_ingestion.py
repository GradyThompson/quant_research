"""Pure orchestration tests: no Massive credentials or network required."""

import pytest

from quant_research.data.ingestion.batch import BatchDailyIngestor, IngestionTarget

AAPL = IngestionTarget("AAPL", "XNAS")
MSFT = IngestionTarget("MSFT", "XNAS")


def test_ingests_each_target_and_preserves_result_order():
    requested: list[IngestionTarget] = []

    def ingest_one(target: IngestionTarget) -> str:
        requested.append(target)
        return f"saved:{target.ticker}"

    report = BatchDailyIngestor(ingest_one).ingest([AAPL, MSFT])
    assert requested == [AAPL, MSFT]
    assert report.successful == ((AAPL, "saved:AAPL"), (MSFT, "saved:MSFT"))
    assert report.failed == ()
    assert report.succeeded


def test_reports_individual_failures_and_continues():
    def ingest_one(target: IngestionTarget) -> str:
        if target == AAPL:
            raise RuntimeError("vendor request failed")
        return "ok"

    report = BatchDailyIngestor(ingest_one).ingest([AAPL, MSFT])
    assert report.successful == ((MSFT, "ok"),)
    assert [(failure.target, failure.error_type, failure.message) for failure in report.failed] == [
        (AAPL, "RuntimeError", "vendor request failed")
    ]
    assert not report.succeeded


def test_fail_fast_mode():
    def fail(_: IngestionTarget) -> str:
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        BatchDailyIngestor(fail, continue_on_error=False).ingest([AAPL, MSFT])


def test_duplicate_target_rejected_before_any_work():
    requested: list[IngestionTarget] = []

    def save(target: IngestionTarget) -> str:
        requested.append(target)
        return "ok"

    with pytest.raises(ValueError, match="duplicate"):
        BatchDailyIngestor(save).ingest([AAPL, AAPL])
    assert requested == []


def test_empty_universe_is_error():
    with pytest.raises(ValueError, match="at least one"):
        BatchDailyIngestor(lambda target: target).ingest([])


@pytest.mark.parametrize(
    "ticker,mic", [("aapl", "XNAS"), (" AAPL", "XNAS"), ("", "XNAS"), ("AAPL", "nasdaq")]
)
def test_invalid_target_rejected(ticker: str, mic: str):
    with pytest.raises(ValueError):
        IngestionTarget(ticker, mic)


def test_same_ticker_different_mics_not_deduplicated():
    second = IngestionTarget("AAPL", "XNYS")
    result = BatchDailyIngestor(lambda t: t.mic).ingest([AAPL, second])
    assert len(result.successful) == 2
