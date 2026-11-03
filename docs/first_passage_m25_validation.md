# M2.5 — Development-only validation audit

## Scope

M2.5 makes no changes to the Brownian first-passage solver, return-bar volatility
estimators, or M2.4 seasonality equations. It adds safeguards and tests for:

- An explicit training/embargo split: only the designated first 38 training
  sessions fit the intraday variance profile. The preceding embargo session is
  loaded solely as point-in-time volatility warm-up. The validation sessions
  remain for evaluation only; the reserved 11 withheld sessions are not loaded.
- A historical-frequency comparison: a three-class empirical distribution per
  barrier configuration, conditioned on the fixed opening/middle/closing
  session periods. It uses first-touch labels from *training only* and shrinks
  each period's estimate toward pooled training frequencies with 30 pseudo-
  observations. Pooled frequencies themselves have a Dirichlet(1,1,1) prior.
- Paired comparison against this historical-frequency benchmark using the same
  validation decisions, multiclass /3 Brier and log loss, plus session-block
  bootstrap intervals for frequency Brier minus model Brier. Positive values
  favor the first-passage model.
- Upper-first, lower-first and neither calibration gaps, each with a session-
  block interval, rather than relying solely on aggregated any-touch rates.
- Explicit tests of opening gap order, minute-bar double hits, embargo
  isolation, and training/validation separation.

The pre-existing always-neither baseline remains available. This historical-
frequency baseline is more informative but is not a claim of market edge.

## Testing and interpretation

Keep the M2.4 barrier grid unchanged, compare all four volatility estimators,
and write reports to new directories such as `data/dev/models/AAPL_m25`.

Inspect `historical_frequency_benchmark.scores`, especially
`brier_skill_vs_frequency` and
`frequency_minus_model_brier_95pct_session_bootstrap`. A point estimate
slightly above zero is not convincing if the interval spans zero. Likewise,
class-specific calibration intervals can remain wide with ten days.

**Note:** validation data has already informed the choice of intraday
seasonality, so the current ten-session window is *development* data. Further
exploration must not be presented as independent confirmation. Freeze a
candidate and protocol before ever opening the reserved holdout.

Ambiguous both-barrier minute bars remain excluded from three-class scoring
because their first-touch ordering is not observable from OHLC. Their number
is reported by the existing exclusions summary. This exclusion can bias
class-level scoring; an independent tick-data audit would be necessary to
resolve order unambiguously.
