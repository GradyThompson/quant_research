# M2.1 — Volatility diagnostics and scenario-specific coverage

The M2 mathematical first-passage equations and historical barrier labeling are unchanged.
M2.1 adds two **research-only** diagnostics to explain disagreements between predicted
and observed barrier-touch rates. Neither method estimates directional drift, fits
an ML classifier, or generates trades.

## What changed

- The historical scorer can compute **two point-in-time volatility estimates** on
  exactly the same observation dates:
  - `close` (the original M2 estimate): RMS one-minute log close-to-close returns;
  - `parkinson`: square root of the mean `log(high/low)^2 / (4 log 2)` over completed
    adjacent-minute bars.
- `--per-scenario-stride` samples each barrier configuration at intervals of
  `horizon_minutes + 1`, instead of discarding most short-horizon observations by
  applying the largest scenario's stride to every scenario.
- The report adds volatility method, mean/min/max estimated volatility, observed
  and predicted total touch rates, number of scored sessions, and number of
  sessions with any touch. Exclusions are split by volatility method.
- The original defaults are preserved: `close` volatility and global
  `--decision-stride` sampling.

## Run

```bash
uv run python scripts/validate_first_passage.py \
  --ticker AAPL \
  --decision-stride 122 \
  --per-scenario-stride \
  --volatility-method close \
  --volatility-method parkinson \
  --output-dir data/dev/models/aapl_m21_volatility
```

The `--decision-stride 122` argument is ignored when
`--per-scenario-stride` is specified, although the report preserves the
explicit value for provenance. This mode avoids overlapping **outcome windows
within each scenario**, but outcomes across different scenarios still overlap.
The additional observations are not independent across sessions or scenarios.

No holdout data is loaded and the existing 38/1/10/1/10 split is preserved.
All evaluation dates remain **development data**, not independent test data.

## Interpretation

First inspect the `+0.5% / -0.5%, 15-minute` rows. In the previous run,
9/40 forecasts touched a barrier, versus about 1.6 touches expected from the
predicted mean probability. Compare the two estimators using identical scored
observations; do not select an estimator based on one favorable run.

Parkinson range volatility is **not necessarily superior**: it relies on a
continuous-path, approximately constant-volatility assumption, may respond to
microstructure noise, and can be distorted by price spikes and erroneous highs
or lows. Close-to-close volatility has its own sampling limitations. Minute
OHLC cannot resolve first-touch order when both barriers are reached in the
same bar. Model evaluation references a *known close*, not a future executable
trade entry.

Before model selection or trading, evaluate across more **distinct validation
sessions and securities** with cluster-aware uncertainty estimates. Preserve
an untouched holdout and audit actual bar publication timestamps.
