# First-passage baseline (M1)

## Scope

M1 is a **mathematical probability model**, not a trading strategy and not
an established profitable signal. It implements three mutually exclusive
outcomes over a finite horizon: upper touched first, lower touched first,
and neither touched. Parameters are volatility, upper/lower price-barrier
fractions, and horizon in trading minutes.

Files:

- `src/quant_research/model/first_passage.py` — pure-standard-library probability calculation.
- `src/quant_research/model/volatility.py` — historical as-of volatility.
- `src/quant_research/model/first_passage_model.py` — `BarrierModel` adapter plus grid evaluation.
- `scripts/verify_first_passage.py` — deterministic calculation examples.

## Mathematical model

Assume `X_t = log(P_t/P_0) = sigma * W_t`, where sigma is the standard
 deviation of a one-minute log return. The absorbing boundaries are
`a=log(1-lower_pct)<0` and `b=log(1+upper_pct)>0`. The solver integrates
the diffusion's first exit through either boundary by the method of images
(short horizons) and the absorbing heat-kernel eigenseries (long horizons).
It **does not simulate discrete minute closes**. The long-time upper-first
probability is `-a/(b-a)`; for +2%/-1% this is about 33.67%, not precisely
33.33%, due to logarithmic barriers.

**Zero log-price drift is not zero expected simple-price return.** A
price-martingale geometric Brownian motion would require a log drift
`-sigma**2/2`; M1 intentionally omits that drift as the simplest baseline.
This approximation needs empirical verification before strategy design.

## Volatility

M1 estimates `sigma = sqrt(mean(r_i**2))` from the latest 390 valid
adjacent one-minute log close-to-close returns (minimum 60 by default).
It does not cross missing-bar or overnight gaps. All source bars must
be complete, ordered, and known at `decision_at`. There is no calibration
or fitting against future barrier outcomes.

The estimator is deliberately simple; squared returns can respond sharply
to jumps, outliers, and volatility regimes. M1 holds volatility constant
through the forecast horizon.

## Usage

```bash
uv run python scripts/verify_first_passage.py
```

For integration, supply the existing `BarrierModelInput` with a historical
PyArrow minute-bar table containing `timestamp`, `available_at`, `close`.
Then call `run_barrier_model(FirstPassageModel(), input)` or evaluate many
`BarrierScenario`s in one `FirstPassageModel.predict_many(input, scenarios)`
call. The model produces `BarrierForecast` and **does not provide a calibrated
confidence value**.

## Next step — probability validation, before trading

1. Prepare historical decisions with bars strictly available at each time.
2. Use the **same entry reference**, clock, and horizon definition for
   forecasts and actual outcome labels. Avoid forecasts that cross an
   unmodeled market close. The current model assumes continuous trading
   over the specified interval.
3. Score upper-first, lower-first, neither with multiclass Brier and log
   loss and compare with empirical and constant-probability baselines.
4. Audit minute bars where both barriers touch during the same bar;
   OHLC data cannot recover first-touch order and such labels are ambiguous.
5. Evaluate across separate dates, securities, and barrier/horizon grids;
   do not tune using the untouched final holdout.
6. Only after probability validation, introduce trading costs, conservative
   fill assumptions, and a separately tested profit-based policy.

**M1 has not yet been empirically calibrated or profit-tested.**
