# P4.6: ETF early-to-late intraday momentum (exploratory)

## Motivation

The 2023/2024 fixed-stock ORB and full-year 2024 cross-sectional high-RVOL
strategies did **not** establish profitability after 12 bps round-trip costs.
Market-wide abnormal RVOL improved upon ordinary-volume controls, but did not
pay its presumed costs. P4.6 tests a DIFFERENT hypothesis, not a retuned ORB.

Gao, Han, Li, Zhou (2018), *Market intraday momentum*, JFE, investigated
whether first-half-hour US market ETF returns predicted last-half-hour ETF
returns. Their sample was historically earlier; their result is not evidence
that an implementable 2024 edge exists. We use 2024 only as development.

## Predeclared candidates

Two momentum signals and one nonpredictive benchmark are evaluated per ETF:

* `prev_close_momentum` (primary): sign of log(10:00 ET close / prior
  completed regular-session close).
* `open_momentum` (alternative): sign of log(10:00 ET close / 09:30 ET open).
* `always_long` (market-exposure benchmark): +1 every eligible session.

No returns are used to calibrate coefficients or choose thresholds. A zero
signal means cash. Fixed one trade per signal, ETF, and normal trading day.
No assumption that SPY and QQQ trades can be independently capitalized and
aggregated into portfolio equity returns.

* First 30 bars must exist at 09:30–10:00 ET; signal bar must be published.
* Decision happens by 15:30 ET, after the morning bars have been published.
* Entry proxy: **open of the 15:31–15:32 ET minute** (one-minute delay).
* Exit proxy: **open of the 15:59–16:00 ET minute** (before the close auction).
* The position is held approximately 28 minutes. This differs deliberately
  from a 15:30–16:00 research close-to-close return.
* Complete eligible first-30 and last-29-minute segments are required.
* The prior reference must come from a complete 390-bar, *prior* trading day;
  crossing any protected day resets that history. Half days are not traded.
* Rounded fixed cost scenarios: 4, 8, 12, and 20 bps round trip, **same trades
  across costs**. This is a price-only proxy; spreads/impact/borrow unknown.
* Bootstrap interval resamples entire trading sessions. No automatic model
  promotion; output is development evidence.

## Install

Unzip the repository-relative patch at the root of quant_research.

## Smoke test (previously inspected development history)

```bash
uv run python scripts/research_late_session_momentum.py \
    --start 2023-12-28 \
    --end 2024-02-29 \
    --validation-start 2024-01-02 \
    --progress-every 10 \
    --output-dir data/dev/models/p4_6_late_etf_smoke
```

## Full development run

```bash
uv run python scripts/research_late_session_momentum.py \
    --start 2023-12-28 \
    --end 2024-12-31 \
    --validation-start 2024-01-02 \
    --progress-every 10 \
    --output-dir data/dev/models/p4_6_late_etf_2024
```

The script uses `data/dev/massive_flat_files/minute_manifests`, the original
`data/dev/minute_training_receipt_20250113.json`, and the source-revision
keyed extraction cache. Full-market gzip extraction for SPY/QQQ may be slow on
the first run; subsequent runs with exactly these symbols use the cache.

## Required analysis

Evaluate *primary* SPY and QQQ signals separately. The candidate is only
worth continuing if its after-cost advantage over both cash and always-long
is robust across many sessions and months, with a confidence interval and
outlier concentration that do not depend on a handful of dates. Assess long
and short positions separately, as short borrowing is excluded. Do not tune
the entry time or feature thresholds to rescue negative 2024 results.

2023 and 2024 have been repeatedly explored. Do not present results on these
dates as untouched validation. Reserve genuinely unseen later dates for
locked-final evaluation; the original protected days are never opened.

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
