# P4.4.3 — Faster historical opening-range scans

This patch changes the *historical research adapter*, not the opening-range trading strategy, trade timing, or score definitions. It does not change Massive credential handling, M2.5, or execution infrastructure.

## Why the old run can be slow

The market-wide compressed Massive minute files contain all US equities. The old parser calls `csv.DictReader` on every row even though the research script needs only five symbols, and it repeats full decompression on every experiment. The old script also reports no progress until the entire run completes. File SHA-256 verification reads each compressed file once before decompression.

## Improvements

1. `minute_research_extract.py` contains a selective parser that detects the CSV header and, for the normal ticker-first Massive layout, rejects unrelated ticker rows before constructing Python CSV objects. The fallback uses `csv.reader` for arbitrary header order. Validation of selected price, time, and volume records remains in place.
2. An optional per-day gzip JSON extraction cache stores selected ticker bars only. Cache identity includes the source file's *verified SHA-256*, all requested symbols, publication lag, and cache schema. An atomic replacement prevents partially written cache entries. A corrupt or mismatched entry fails rather than silently being used. Manifest SHA-256 verification is **still performed for every source file on every run** before the cache is read.
3. The long-history script prints progress during source verification, session extraction, and ticker scoring. It keeps output reports immutable and still excludes protected original holdout dates before accessing source files.
4. If interrupted during first-time extraction, previously completed day caches survive and can be reused on the next run.

## Install

```bash
cd ~/Documents/investing/quant_research
unzip -o ~/Downloads/p4_4_3_orb_scan_performance.zip -d .
```

## Verification

```bash
uv run ruff format
uv run ruff check . --fix
uv run ruff format --check .
uv run mypy src/quant_research
```

```bash
uv run pytest
```

## Run (new output directory)

```bash
uv run python scripts/research_opening_range_flat_history.py \
    --ticker AAPL \
    --ticker NVDA \
    --ticker MSFT \
    --ticker PLTR \
    --factor-ticker QQQ \
    --start 2023-01-03 \
    --end 2024-12-31 \
    --validation-start 2024-01-02 \
    --horizon 30 \
    --horizon 60 \
    --roundtrip-cost-bps 4 \
    --roundtrip-cost-bps 8 \
    --roundtrip-cost-bps 12 \
    --progress-every 10 \
    --output-dir data/dev/models/p4_4_orb_2023_2024_fast
```

The default extraction cache is `data/dev/massive_flat_files/research_extracts`. The cache contains **selected bars only**, not the full US equity universe. For a no-cache run, add `--no-extract-cache`. To change its location, supply `--extract-cache-root PATH`.

## Resource considerations

- The **first optimized run** must still SHA-256 verify each market file, decompress every gzip, and scan every row. Selective parsing reduces Python object allocation, but does not eliminate I/O or decompression.
- Subsequent runs with identical ticker sets and source revisions can reuse the extracted-bars cache, avoiding gzip decompression and full-market CSV parsing, **while still hashing all compressed source files for integrity**.
- A different ticker set, publication lag, or source revision creates a different cache key. This is intentional; it prevents an apparently successful but incomplete backtest.
- All source sessions and final reports stay subject to the original protected-date exclusion, immutable output rule, and price-proxy caveats.
- Use the full historical history before selecting a variant. The earlier 15-session ORB outcomes remain exploratory and are not an independent holdout.

Local validation used 17 isolated synthetic tests and a synthetic 300,000-row ticker-first CSV filtering benchmark. The ticker-filter stage was ~5.7x faster in that *synthetic* benchmark; it is not a forecast of end-to-end speed on your machine.
