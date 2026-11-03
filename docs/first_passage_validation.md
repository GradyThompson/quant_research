# M2 — First-passage probability validation

This is a **mathematical calibration study**. It does not fit a classifier,
search trade thresholds, or generate any simulated trades.

## Models

Both models use a geometric diffusion with a volatility estimated from
already-published adjacent-minute log returns:

- `log_zero` (the original M1): `d log(S) = sigma dW`.
- `price_zero`: `d log(S) = -(sigma^2/2) dt + sigma dW`, equivalent to
  `dS / S = sigma dW`. It is a zero-expected-**price**-drift benchmark.

Hitting probabilities are computed using the absorbing Brownian transition
kernel. The price-zero solution exponentially tilts the driftless first-hit
flux, using image sums at short horizons and the integrated eigenseries at
long horizons. These are closed-form series, not Monte Carlo estimates.

At an unlimited horizon, the price-zero model gives
`P(upper first) = lower_pct / (upper_pct + lower_pct)` exactly, e.g.
`1/3` for +2% / -1%. A martingale price process has no positive expected
profit before transaction costs merely because variance is large.

## Historical labels

Minute bars use **END timestamps** and `available_at` publication times.
The decision price is the close of the last published bar. Both barriers
are measured relative to that *known reference close*, not the open of a
future trade. The first scoring minute must start exactly at `available_at`.
The next `horizon_minutes` bars must exist and be consecutive in that
session. No overnight continuation is assumed.

For each future minute bar:

1. A gap open at/through a barrier determines which one was crossed first.
2. Otherwise if both high and low cross their barriers within that same
   minute, the result is **ambiguous** and is excluded.
3. Otherwise, the first touch determines upper or lower; if neither is
   reached through the complete horizon, label neither.

Missing windows and unaligned publication times are excluded. This can
introduce selection bias and is reported explicitly. Using completed bars
with `available_at` does **not** prove that historical market data were
actually published at those times; check receipt provenance.

## Chronological split and privacy of holdout

Defaults reuse the existing **38 training / 1 embargo / 10 validation /
1 embargo / 10 holdout** partition from the 60 sessions before 2025-01-13.
The new mathematical model does **not** train on these dates, but it uses
all pre-validation sessions as point-in-time volatility warm-up.

The validator **never even loads** the post-validation embargo, ten holdout
sessions, or the January 13 evaluation day. Only the earlier validation
sessions are scored. These validation days have already been used in
prior model development; they are not a fresh out-of-sample test.

## Run

```bash
uv run python scripts/validate_first_passage.py \
    --ticker AAPL \
    --output-dir data/dev/models/aapl_first_passage_m2
```

Customize the grid by repeating `--scenario` (fractions, not percent units):

```bash
uv run python scripts/validate_first_passage.py \
    --ticker NVDA \
    --scenario 0.005,0.005,15 \
    --scenario 0.02,0.01,60 \
    --decision-stride 122 \
    --output-dir data/dev/models/nvda_first_passage_m2
```

The report gives three-class Brier, log loss, observed-versus-predicted
frequencies, upper-probability calibration bins, descriptive volatility-regime
strata, exclusion reasons, exact
pinned source dataset versions, and a SHA-256 of the receipt. The default
62-minute stride makes decisions approximately nonoverlapping for horizons
up to 60 minutes, but 120-minute outcomes overlap; use 122 or larger to
avoid horizon overlap across decisions on one security. Different
configurations and securities are still dependent.

## Caveats before trading

- Constant volatility and continuous diffusion are approximations.
- Historical OHLC may miss within-minute ordering, gaps, and microstructure.
- The forecast reference close can differ from an executable future fill.
- The price-neutral model is not a forecast of actual market risk premia.
- Proper scores alone do not establish a profitable trading edge.
- Re-running validation to tune volatility estimates consumes validation
  data; freeze choices before a genuinely new independent evaluation.
