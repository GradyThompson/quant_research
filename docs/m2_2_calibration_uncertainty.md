# M2.2 — Uncertainty and baseline diagnostics

This increment **does not change** the mathematical first-passage probabilities, volatility estimators, historical labeler, selected sessions, or forecast grids. It adds diagnostics to the existing `validate_first_passage.py` report.

## New fields

The top-level report field `diagnostics` contains:

- `calibration`: observed and model-expected first touches; observed-minus-expected touch **rate**; a 95% percentile *session-block bootstrap* interval; the always-neither multiclass Brier score; and Brier skill (`1 - model_Brier / always_neither_Brier`), which is null if no touches occurred. Also records touch counts by session.
- `paired_volatility`: matched predictions under `close` and `parkinson` for the same decision timestamp, scenario, drift and observed outcome. Includes the mean **close minus Parkinson Brier** (positive favors Parkinson) with a session-block interval. Mismatched decision keys or labels raise an error rather than silently changing the comparison population.
- `uncertainty`: resampling parameters and explicit limitations.

A session is the resampling unit because forecasts from the same market day are correlated. With only ten validation days, even session-block intervals may be unstable and should **not** be interpreted as definitive statistical tests. Results across barrier grids and securities are correlated too. No tuning should use the reserved holdout.

Brier uses the established per-class mean convention: `(sum over three outcome classes of squared probability error)/3`. A classifier that always predicts neither has Brier `2/3 × observed_touch_rate`; this is a useful but deliberately simplistic reference. Its hard-zero event probabilities are unsuitable as a finite log-loss benchmark.

## Run

```bash
uv run python scripts/validate_first_passage.py \
    --ticker AAPL \
    --per-scenario-stride \
    --volatility-method close \
    --volatility-method parkinson \
    --output-dir data/dev/models/aapl_m22_diagnostics
```

## How to interpret

- An interval for `observed_minus_expected_touch_rate` that spans zero indicates the data do not sharply distinguish observed and predicted touch rates, given the resampling approximation.
- `brier_skill_vs_neither > 0` means the mathematical model improves multiclass Brier over an always-neither benchmark on the observed sample; negative means worse.
- A paired `close_minus_parkinson_brier` interval spanning zero means there is no clear evidence to prefer one volatility estimator under the chosen sample and scoring rule.
- This is a forecast of barriers relative to the last *published close*, not a prediction of the outcome of a future executed trade.

Before changing the volatility dynamics, evaluate more distinct sessions and multiple securities, keep both estimators if the evidence is mixed, and audit timestamps and bar coverage. The historical holdout remains unopened.
