"""P5-A.3: reference-type and split reconciliation, NOT a corrected backtest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from quant_research.model.p5_reference_reconciliation import reconcile_p5_report
from quant_research.paths import PROJECT_ROOT

BASE = "https://api.massive.com"


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid object JSON: {path}")
    return data


def _get_json(url: str, key: str) -> dict[str, Any]:
    """Use Authorization header; never embed API keys in URLs or logs."""
    if urlparse(url).scheme != "https" or urlparse(url).netloc != "api.massive.com":
        raise ValueError("unexpected Massive API host")
    for attempt in range(5):
        request = Request(
            url, headers={"Authorization": f"Bearer {key}", "Accept": "application/json"}
        )
        try:
            with urlopen(request, timeout=30) as response:
                payload: Any = json.load(response)
            if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                raise ValueError("Massive response lacks results list")
            return payload
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 4:
                raise RuntimeError(f"Massive API HTTP {exc.code} for {urlparse(url).path}") from exc
            time.sleep(min(2**attempt, 16))
        except URLError as exc:
            if attempt == 4:
                raise RuntimeError(f"Massive API request failed for {urlparse(url).path}") from exc
            time.sleep(min(2**attempt, 16))
    raise RuntimeError("unreachable Massive retry exhaustion")


def _fetch_paged(url: str, key: str) -> dict[str, Any]:
    all_results: list[dict[str, Any]] = []
    seen: set[str] = set()
    while url:
        if url in seen:
            raise ValueError("pagination cycle detected")
        seen.add(url)
        payload = _get_json(url, key)
        for item in payload["results"]:
            if not isinstance(item, dict):
                raise ValueError("invalid provider result row")
            all_results.append(item)
        next_url = payload.get("next_url")
        if next_url is not None and not isinstance(next_url, str):
            raise ValueError("invalid next_url")
        url = next_url or ""
    return {"results": all_results, "recorded_at": datetime.now(UTC).isoformat()}


def _cached_get(
    *,
    cache_root: Path,
    family: str,
    parameters: dict[str, str],
    key: str,
) -> dict[str, Any]:
    canonical = family + "?" + urlencode(sorted(parameters.items()))
    sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    path = cache_root / family / f"{sha}.json"
    if path.is_file():
        record = _read_json(path)
        if record.get("query") != canonical or not isinstance(record.get("payload"), dict):
            raise ValueError(f"cached evidence mismatch: {path}")
        return record["payload"]
    url = f"{BASE}/{family}?{urlencode(parameters)}"
    payload = _fetch_paged(url, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Query-specific files are immutable. A rerun reuses previous completed calls.
    with path.open("x", encoding="utf-8") as handle:
        json.dump({"query": canonical, "payload": payload}, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def collect_evidence(
    *,
    report: dict[str, Any],
    api_key: str,
    cache_root: Path,
    horizon: int,
    cost_bps: float,
) -> dict[str, Any]:
    """Fetch only dated reference and split data implicated by saved trades."""
    if report.get("holdout_accessed") is not False:
        raise ValueError("research holdout is not certified untouched")
    buckets = report["results"][f"{horizon}d_positive_short_all"]
    cost_key = next((k for k in buckets if abs(float(k) - cost_bps) < 1e-9), None)
    if cost_key is None:
        raise ValueError("cost scenario not found in saved report")
    trades = buckets[cost_key]["trades_detail"]
    if not trades:
        raise ValueError("report has no saved trades")
    # Reference point is the formation date: the selection was made there.
    refs = sorted({(str(t["formation_day"]), str(t["ticker"])) for t in trades})
    split_names = sorted({str(t["ticker"]) for t in trades} | {str(t["factor"]) for t in trades})
    start = min(str(t["entry_day"]) for t in trades)
    end = max(str(t["exit_day"]) for t in trades)
    references: dict[str, Any] = {}
    splits: dict[str, Any] = {}
    for index, (day, ticker) in enumerate(refs, 1):
        references[f"{day}|{ticker}"] = _cached_get(
            cache_root=cache_root,
            family="v3/reference/tickers",
            parameters={
                "ticker": ticker,
                "date": day,
                "active": "true",
                "market": "stocks",
                "limit": "1000",
            },
            key=api_key,
        )
        if index % 25 == 0 or index == len(refs):
            print(f"Dated reference evidence: {index}/{len(refs)}", flush=True)
    for index, ticker in enumerate(split_names, 1):
        splits[ticker] = _cached_get(
            cache_root=cache_root,
            family="stocks/v1/splits",
            parameters={
                "ticker": ticker,
                "execution_date.gte": start,
                "execution_date.lte": end,
                "limit": "5000",
            },
            key=api_key,
        )
        if index % 25 == 0 or index == len(split_names):
            print(f"Split evidence: {index}/{len(split_names)}", flush=True)
    return {
        "schema_version": 1,
        "source": "Massive historical ticker references and stocks/v1/splits",
        "collected_at": datetime.now(UTC).isoformat(),
        "split_start": start,
        "split_end": end,
        "references": references,
        "splits": splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--collect-evidence", action="store_true")
    parser.add_argument("--cache-root", type=Path, default=Path("data/dev/massive_rest_evidence"))
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--cost-bps", type=float, default=4.0)
    args = parser.parse_args()
    report_path = args.report / "validation.json" if args.report.is_dir() else args.report
    if not report_path.is_file():
        raise FileNotFoundError(f"research report not found: {report_path}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.output}")
    report = _read_json(report_path)
    if args.collect_evidence:
        if args.evidence.exists():
            raise FileExistsError(f"refusing to overwrite evidence: {args.evidence}")
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        api_key = os.environ.get("MASSIVE_API_KEY", "")
        if not api_key:
            raise ValueError("MASSIVE_API_KEY missing from project .env or environment")
        evidence = collect_evidence(
            report=report,
            api_key=api_key,
            cache_root=args.cache_root,
            horizon=args.horizon,
            cost_bps=args.cost_bps,
        )
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        evidence = _read_json(args.evidence)
    result = reconcile_p5_report(report, evidence, horizon=args.horizon, cost_bps=args.cost_bps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Original trades: {result['original_trades']}")
    print(
        f"Verified common-stock trade subset: {result['verified_common_stock_trades_after_original_selection']}"
    )
    print(f"Excluded / unverified: {result['excluded_or_unverified_counts']}")
    print(
        f"Post-hoc arithmetic split-reconciled subset (NOT A REBACKTEST): "
        f"{result['verified_subset_arithmetic_bps_per_original_entry_session']:+.3f}bps/entry-session"
    )
    print(f"Audit: {args.output}")


if __name__ == "__main__":
    main()
