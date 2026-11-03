# P6.1.2 — Bounded parallel dataset build

This patch changes **only** `scripts/build_p6_barrier_dataset.py` and adds regression tests. It applies on top of P6 and P6.1.1; it does not replace model or audit files.

## What is faster

- `--workers N` overlaps SHA-256 verification and daily market summaries with source-pinned selected-symbol minute extraction and barrier labeling, using an ordered, bounded process pool.
- At most `2 * N` prefetch summary tasks and `2 * N` pending daily label results are retained. The 20-session historical liquidity screen remains serial and chronological.
- No source hashes, identity verification, split exclusions or required publication delays are skipped.
- Original day-cache identity `P6-day-v3` and selected-extract cache identities are unchanged. Existing completed cache files remain valid and are reused on rerun with the **same** start/end, universe size, source revisions, split evidence and 20-session history.
- The final output remains immutable; the incomplete dataset is written to a temporary file and committed only on successful completion.
- The output manifest includes `build_performance` counters: elapsed time, time blocked waiting for summary/source work, REST historical-reference requests, time blocked waiting for labels, completed day cache hits/misses.

## Recommended invocation

If a previous run is still active, do **not** overwrite the script while Python is executing it. Either allow the run to complete or interrupt it, install the patch, then rerun. The existing completed daily caches are preserved by a normal interruption.

```bash
cd ~/Documents/investing/quant_research
unzip -o ~/Downloads/P6_1_2_Faster_Dataset_Builder.zip -d .

uv run python scripts/build_p6_barrier_dataset.py \
    --start 2023-01-03 \
    --end 2024-12-31 \
    --validation-start 2024-01-02 \
    --max-universe 500 \
    --workers 2 \
    --progress-every 10 \
    --output-dir data/dev/models/p6_dataset_2023_2024
```

Try 2 workers first, then 3–4 if the dataset is on a fast SSD and memory is ample. More workers can be slower when storage is saturated. `--workers 1` retains the original serial scheduling behavior.

**Cache caveat:** The 300-stock Nov 2023–Apr 2024 smoke-test day caches will NOT match a 500-stock 2023–2024 run, because both universe size and full-range split fingerprint are in the original cache key. This patch correctly preserves those cache keys for safety; it cannot reuse mismatched results. Completed days from the **current** 500-stock/full-range run are reusable.

## Diagnostics

At the end, read `data/dev/models/p6_dataset_2023_2024/manifest.json`, especially:

```json
"build_performance": {
  "workers": 2,
  "elapsed_seconds": 0.0,
  "summary_wait_seconds": 0.0,
  "reference_seconds": 0.0,
  "label_wait_seconds": 0.0,
  "day_cache_hits": 0,
  "day_cache_misses": 0
}
```

These **example numbers are placeholders**, not performance measurements. Wait times are wall-clock time blocked at stage boundaries and can overlap background worker execution. If `reference_seconds` dominates, dated Massive API pagination—not gzip parsing—is the principal limitation and increasing process workers may provide little benefit. If `summary_wait_seconds` or `label_wait_seconds` dominates, the process pool should help subject to disk throughput.

## Verification

```bash
uv run ruff format
uv run ruff check . --fix
uv run ruff format --check .
uv run mypy src/quant_research
uv run pytest
```

Only the builder script and one additional test file are installed. The patch does not change output label/feature definitions, original holdout exclusions, or modeling methods. The new multiprocessing test compares serial and parallel JSON observations and source hashes on real synthetic minute gzip files and checks that protected dates are never submitted for source work.
