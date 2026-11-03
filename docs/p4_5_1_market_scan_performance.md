# P4.5.1 — Faster market-wide research (no new data)

## What changed

The original P4.5 experiment repeatedly scans market-wide minute files,
including every session from the start date. The market summary is a separate
calculation for each trading day, so this patch offers a bounded process pool
with `--summary-workers N`. Results feed the sequential point-in-time selector
in **original chronological order**, regardless of worker completion time.

Existing source SHA-256 verification, cache identities, protected-session
exclusion, and downstream event/portfolio calculations are unchanged.
Only authorized days are queued, and at most `2*N` summaries are in flight.
The default is one worker, preserving original single-process behavior.
Use two workers initially; try four on fast local SSDs with adequate memory.
Multiple concurrent file scans may be slower on HDDs or bandwidth-constrained
storage. The optimization cannot remove gzip decompression or source hashing.

## Bigger immediate gain: shorter warmup

P4.5 uses exactly 20 prior **consecutive** sessions for volume and liquidity
baselines. For Jan–Feb 2024 validation, a November 2023 start provides ample
warmup; there is no need to scan January–October 2023. This is equivalent for
validation dates once the original 20-day window is established, provided
there are no protected/missing session gaps that invalidate the lookback.

### Fast Jan–Feb 2024 smoke test

```bash
uv run python scripts/research_cross_sectional_orb.py \
    --start 2023-11-01 \
    --end 2024-02-29 \
    --validation-start 2024-01-02 \
    --summary-workers 2 \
    --progress-every 5 \
    --output-dir data/dev/models/p4_5_1_rvol_smoke
```

### Full 2024 development study

```bash
uv run python scripts/research_cross_sectional_orb.py \
    --start 2023-11-01 \
    --end 2024-12-31 \
    --validation-start 2024-01-02 \
    --summary-workers 2 \
    --progress-every 10 \
    --output-dir data/dev/models/p4_5_1_rvol_2024
```

Use fresh output directories; original reports are never overwritten. Existing
summary caches remain valid; workers will use them without re-parsing.

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

## Performance expectations

- Warmup reduction saves approximately 70% of source-day reads for the
  Jan–Feb 2024 smoke run; this is a work reduction, NOT a guaranteed wall-clock
  speedup.
- Process-level parallelism can accelerate first-time Python/gzip summaries
  if CPUs and storage sustain concurrent file scans. Actual speedup depends
  on I/O, CPU, cache hits, and selected-stock extraction. No end-to-end Massive
  dataset benchmark was performed.
- SHA-256 validation is still a full source read. A cached summary avoids
  decompressing and parsing the source again, but not that verification.

This remains exploratory research. The reserved final holdout must remain
untouched.
