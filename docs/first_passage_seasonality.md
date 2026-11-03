# M2.4 — Intraday volatility seasonality (development only)

## Why

M2.3 found a consistent 0.3–0.44x forecast/forward variance ratio in the
first 60 bars, but around 1.65–2.2x in the middle 240 bars for four equities.
The overall variance ratio conceals this intraday heterogeneity.

## Method

The original first-passage solver remains untouched. New volatility-method
choices are `seasonal_close` and `seasonal_parkinson`. They fit three fixed
intraday *relative variance* factors using **only sessions before the first
validation session**:

- First 60 minute-bar positions (opening period)
- Next 240 positions (middle period)
- Last 90 positions (closing period)

For each method, variance in period `b` is estimated from adjacent-minute
historical squared returns (or Parkinson range variances). Each period mean
is shrunk toward the historical overall mean using 390 prior observations.
The resulting factors scale variance, **not standard deviation**.

At each decision, a rolling raw variance is divided by the mean fitted factor
of the *same past returns* in its lookback window, then multiplied by the
average fitted factor over the *future holding window*. The latter is a
function of clock position and horizon only, not future prices. With a
variance-driven zero-price-drift diffusion, this deterministic variance
schedule is an exact Brownian variance-time change; the existing solver can
use the equivalent horizon-average variance.

The fit excludes validation and holdout sessions. The calculation is
point-in-time, conditional on the historical dataset's bar-availability
convention, and uses no future return observations to form predictions.

## Run

Keep the original baseline and compare both new variants on identical
forecasts (use a new output directory):

```bash
for ticker in AAPL NVDA MSFT PLTR; do
    uv run python scripts/validate_first_passage.py \
        --ticker "$ticker" \
        --per-scenario-stride \
        --volatility-method close \
        --volatility-method parkinson \
        --volatility-method seasonal_close \
        --volatility-method seasonal_parkinson \
        --scenario 0.002,0.002,5 \
        --scenario 0.003,0.002,10 \
        --scenario 0.005,0.003,15 \
        --scenario 0.005,0.005,30 \
        --scenario 0.01,0.005,60 \
        --output-dir "data/dev/models/${ticker}_m24"
done
```

Check printed `Paired intraday seasonality Brier differences` for consistent
improvements, also log loss, probability calibration and *time-of-day*
forward variance ratios. Profile factors are saved in `validation.json`.

## Limits

- Three buckets are deliberately coarse; session-position indexing is an
  approximation, especially for missing or early-close bars.
- The training-only profile uses prior sessions and is held fixed throughout
  validation; it does not update using future validation observations.
- Previous validation dates are development data; do not open the reserved
  holdout for exploratory model selection.
- Variance adjustment does not resolve jumps, OHLC double-touch ordering,
  spread/slippage, or a directional trading edge.
- No probabilities from the existing `first_passage_model.py` adapter are
  changed by this patch; seasonal variants are experimental validation modes.
