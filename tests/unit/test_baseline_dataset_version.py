"""Regression: JSON UUID strings must become UUID objects in DatasetPin."""

from uuid import UUID

import pytest

from quant_research.backtesting import baseline


def test_baseline_decodes_dataset_version_uuid(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = UUID("e91ff25a-202a-4a9a-a98f-eb7c1d0b4342")
    seen = []

    class StopAfterPins(Exception):
        pass

    def capture(**kwargs: object) -> None:
        seen.extend(kwargs["pins"])  # type: ignore[arg-type]
        raise StopAfterPins

    monkeypatch.setattr(baseline, "run_pinned_backtest", capture)
    config = {
        "store_root": "data/dev",
        "datasets": [{"dataset_id": "market.ohlcv.daily.aapl", "version": str(expected)}],
        "instruments": [{"security_id": "SEC-14290302951544531124"}],
        "buy_security_id": "SEC-14290302951544531124",
        "shares": 1,
        "initial_cash_usd": "10000",
    }
    with pytest.raises(StopAfterPins):
        baseline.run(config)
    assert len(seen) == 1
    assert seen[0].version.value == expected
    assert isinstance(seen[0].version.value, UUID)
