# Public competing-risks: performance and precision audit (first increment)

## Scope

This increment targets the computational core and validation decision policy. It
is **not** a claim that every module in the repository has been optimized.
The fitted categorical hazard likelihood, probability definitions, barrier
labels, model parameter file format, and holdout isolation remain unchanged.

## Verified performance opportunities

- Dense training previously rebuilt all historical return/volume arrays for
  *each at-risk minute*. The first nine features are constant within an
  episode; only elapsed time changes. The code now computes the history-based
  features once per decision.
- Predictions previously recomputed history-based features once for each
  forecast horizon minute. Prediction now constructs the feature matrix from
  one history-based feature vector.
- Binary AUC previously compared every positive with every negative
  (`O(n_positive * n_negative)`). Rank grouping computes exact AUC, including
  ties, in `O(n log n)`.
- Training and validation used separate implementations for checking pinned
  dataset row counts and security identities, and the validation runner
  loaded training tables twice. A shared pinned-session audit is now used;
  training data are loaded once for fitting and validation tables once for
  scoring.

A local synthetic benchmark against the uploaded original model gave a 21.6x
speedup in episode feature construction and a 32.8x speedup in repeated
forecasts. These figures are environment-specific microbenchmarks. Use
`scripts/benchmark_competing_risks_features.py` to reproduce the feature
calculation comparison locally.

## Precision objective

The forecasting model continues to estimate calibrated upper-first,
lower-first, and neither probabilities by maximizing regularized likelihood.
The **decision policy** emphasizes positive-trade precision:

- True positive: selected trade whose *outcome-based net return proxy* > 0.
- False positive: selected trade whose outcome-based net return proxy <= 0.
- Precision: TP / (TP + FP).
- Conservative rank: one-sided Wilson lower confidence bound of precision.
- Eligibility: minimum trade count, minimum opportunity fraction, minimum
  Wilson lower bound, and positive average outcome-based net-return proxy.
- Optimize: among eligible thresholds, rank Wilson lower bound first, observed
  precision second, net-return proxy third. Otherwise abstain.

Defaults are at least 10 selected validation observations, at least 5% of
scored decisions, and a one-sided 95% Wilson lower bound of 50% precision.
These are **research defaults**, not optimized hyperparameters or guarantees.
The threshold grid is 5%-point increments from 0.05 to 0.95, chosen on
validation only. Applying the selected threshold to holdout without
retuning is essential.

The outcome-based economic proxy assumes fixed round-trip costs and *zero*
gross timeout return. Its TP/FP labels are NOT actual executed trade outcomes.
The simulator's entry gap, slippage, spread, commission and timeout fills
must be incorporated in a later validation increment before treating these
figures as a trading advantage.

## Modeling limitation to address next

The fit objective uses a **sum** of hazard-row log likelihoods and a fixed
ridge penalty. Decreasing training stride increases the number of overlapping
hazard rows, thereby changing effective penalty strength. Dense episodes also
have correlated errors. Do not interpret an improvement in validation metrics
as evidence that denser sampling alone caused a causal improvement. Next,
benchmark sample-normalized or session-weighted objectives, tune ridge only
inside training/validation folds, and check stability across securities/dates.

Also, the January 13, 2025 pilot has already influenced engineering and model
choices. It is **not** an untouched test session.

## Suggested next profiling priorities

1. Profile `fit_competing_risks()` on the completed historical dataset, paying
   attention to optimization time, memory and dense overlapping episodes.
2. Measure Parquet read/write and flat-file parsing separately from model
   fitting; do not add caching that breaks dataset-version pinning.
3. Consolidate duplicated backtester-specific accounting and OHLCV checks only
   after frozen-forecast replay regression tests establish exact equivalence.
4. Replace outcome proxies with executable, cost-aware validation trades.
5. Use nested chronological or walk-forward selection, then independently
   score the reserved holdout once. Report TP, FP, precision, opportunity rate,
   net expectancy and uncertainty across sessions and securities.

The research model and artifacts stay in the public code and `data/dev/models/`
respectively; no `private_strategies` changes are needed.
