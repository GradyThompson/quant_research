"""Source integrity, historical identity, split control and builder smoke test."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from types import ModuleType

import exchange_calendars as xcals
import pandas as pd
import pytest

from quant_research.model.ml_barrier import row_from_json


def _builder(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # A container without PyArrow can still validate pure builder functions.
    # In an installed repository the regular ingestion package is used.
    import quant_research

    root = Path(quant_research.__file__).resolve().parent
    name = "quant_research.data.ingestion"
    try:
        __import__(name)
    except ModuleNotFoundError as exc:
        if exc.name != "pyarrow":
            raise
        fake = ModuleType(name)
        fake.__path__ = [str(root / "data" / "ingestion")]
        monkeypatch.setitem(sys.modules, name, fake)
    path = Path(__file__).resolve().parents[2] / "scripts" / "build_p6_barrier_dataset.py"
    spec = importlib.util.spec_from_file_location("p6_barrier_builder_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _summary() -> object:
    from quant_research.data.ingestion.minute_market_summary import DailySummary

    return DailySummary(
        opening_volume=150_000,
        opening_high=101,
        opening_low=99,
        opening_open=100,
        opening_close=100.5,
        last_close=100.5,
        dollar_volume=110_000_000,
        close_eligible=True,
    )


def test_historical_identity_and_split_are_preselection_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _builder(monkeypatch)
    past = [{"FOO": _summary(), "BAR": _summary()} for _ in range(20)]
    current = {"FOO": _summary(), "BAR": _summary()}
    selected = module.choose_liquid(
        current,
        past,
        {"FOO", "BAR"},
        {"BAR"},
        universe=10,
        min_adv=50_000_000,
        min_price=20,
    )
    assert [r[0] for r in selected] == ["FOO"]
    assert (
        module.choose_liquid(
            current, past[:19], {"FOO"}, set(), universe=10, min_adv=50_000_000, min_price=20
        )
        == []
    )


def test_manifest_hash_validation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _builder(monkeypatch)
    day = date(2024, 1, 3)
    file = tmp_path / "raw.gz"
    file.write_bytes(b"sample")
    hash_ = hashlib.sha256(file.read_bytes()).hexdigest()
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    record = {
        "format": 1,
        "bucket": "flatfiles",
        "key": f"us_stocks_sip/minute_aggs_v1/{day:%Y/%m/%Y-%m-%d}.csv.gz",
        "revisions": [{"path": str(file), "sha256": hash_, "bytes": 6}],
    }
    (manifests / f"{day}.json").write_text(json.dumps(record), encoding="utf-8")
    assert module.verified_source(manifests, day) == (file, hash_)
    file.write_bytes(b"changed")
    with pytest.raises(ValueError, match="revision"):
        module.verified_source(manifests, day)


def test_builder_does_not_read_protected_sessions_and_reuses_day_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _builder(monkeypatch)
    import gzip

    cal = xcals.get_calendar("XNYS")
    all_days = [
        d.date()
        for d in cal.sessions_in_range(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-04-30"))
    ]
    protected = all_days[25]
    monkeypatch.setattr(module, "protected_research_days", lambda receipt: {protected})
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    examined: list[date] = []

    def verified(root: Path, day: date) -> tuple[Path, str]:
        assert day != protected
        examined.append(day)
        return tmp_path / "dummy.gz", f"{day}-hash"

    monkeypatch.setattr(module, "verified_source", verified)
    monkeypatch.setattr(
        module, "summarize_market_job", lambda job: (job[0], {"FOO": _summary()}, True)
    )
    calls: list[date] = []

    def valid_stocks(self: object, day: date) -> set[str]:
        assert day != protected
        calls.append(day)
        return {"FOO"}

    monkeypatch.setattr(module.MassiveEvidence, "common_stocks", valid_stocks)
    monkeypatch.setattr(module.MassiveEvidence, "split_days", lambda self, a, b: set())

    def bars(
        source: Path, *, session: date, opening: datetime, tickers: set[str], **kwargs: object
    ) -> tuple[dict[str, list[dict[str, object]]], bool]:
        records: list[dict[str, object]] = []
        for j in range(390):
            end = opening + timedelta(minutes=j + 1)
            price = 100 + j * 0.0001
            records.append(
                {
                    "timestamp": end,
                    "available_at": end + timedelta(minutes=1),
                    "open": price,
                    "high": price + 0.001,
                    "low": price - 0.001,
                    "close": price,
                    "volume": 5000,
                }
            )
        return {ticker: records for ticker in tickers}, False

    monkeypatch.setattr(module, "load_or_parse_selected_minutes", bars)
    config = dict(
        start=all_days[0],
        end=all_days[-1],
        validation_start=all_days[40],
        receipt=receipt,
        manifest_root=tmp_path,
        evidence_root=tmp_path / "evidence",
        summary_root=tmp_path / "summaries",
        extract_root=tmp_path / "extract",
        day_cache_root=tmp_path / "day_cache",
        api_key="",
        universe=5,
        progress_every=100,
    )
    output1 = tmp_path / "one"
    first = module.build_dataset(**config, output_dir=output1)
    assert protected not in examined
    assert first["counts"]["rows"] > 1
    assert first["protected_holdout_opened"] is False
    assert all(item["day"] != str(protected) for item in first["source_revisions"])
    with gzip.open(output1 / "rows.jsonl.gz", "rt", encoding="utf-8") as handle:
        produced = [row_from_json(json.loads(line)) for line in handle]
    assert len(produced) == first["counts"]["rows"]
    assert all(r.day != protected for r in produced)
    before = len(calls)
    second = module.build_dataset(**config, output_dir=tmp_path / "two")
    assert len(calls) == before  # source-keyed day extraction cache
    assert second["counts"]["rows"] == first["counts"]["rows"]
