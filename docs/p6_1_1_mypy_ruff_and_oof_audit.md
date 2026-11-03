# P6.1.1 — Static typing fixes and zero-retrain OOF audit

**Scope:** three existing files are changed and one report-only audit script is added. The predictive models, ensemble weights, barriers, cost assumptions, chronology, original results, and dataset schema are unchanged.

## What was fixed

- `src/quant_research/model/ml_barrier.py`: validate published opening bars using `itertools.chain` rather than `Sequence + Sequence`; deserialize the baseline as an explicit three-float tuple. `row_from_json` now gives a controlled error for an incorrectly sized baseline.
- `src/quant_research/model/ml_barrier_ensemble.py`: convert NumPy indices to native Python `int` before indexing typed `Sequence[BarrierRow]` and extending the selected-indices list.
- `scripts/research_ml_barrier.py`: iterate through dictionary `.values()` instead of binding unused keys (Ruff B007).

No dataset rebuild or model retraining is required to run the audit. The corrected runner is for future model experiments. Existing saved results are preserved.

## Install

```bash
cd ~/Documents/investing/quant_research
unzip -o ~/Downloads/P6_1_1_Static_Fixes_and_OOF_Audit.zip -d .
```

## Verify

```bash
uv run ruff format
uv run ruff check . --fix
uv run ruff format --check .
uv run mypy src/quant_research
```

```bash
uv run pytest
```

## Audit saved research, no retraining

```bash
uv run python scripts/audit_p6_ml_barrier_report.py \
    --report data/dev/models/p6_ml_barrier_smoke/validation.json \
    --cost-bps 4 \
    --output data/dev/models/p6_ml_barrier_smoke_audit.json
```

`--cost-bps` refers to the **additional** spread/slippage scenario over estimated IBKR commissions; it must match an existing reported scenario.

The audit reconstructs the saved per-session returns from trade-level net P&L and fails closed on mismatches, then reports:

- Trade count, number of scored days, utilization of the three available daily slots.
- Individual Brier score and log loss, mean/median net return per selected trade.
- Original per-session net mean and a **five-session circular-block bootstrap** interval.
- Sensitivity after removing five best trades (concentration diagnostic only).
- Paired `direct - math`, `direct - ensemble`, `ensemble - math` return differences and block-bootstrap intervals over exactly the same sessions.
- Overlap in selected trades, plus the Jaccard fraction of common selections.

The bootstrap interval is an exploratory serial-dependence diagnostic, **not** a multiple-testing-adjusted or independent-holdout significance claim. No trade re-selection or recalibration occurs.

## Interpretation of the initial user results

At 4 bps **additional** round-trip costs, the observed smoke-test means were:

| Model | Net bps per session | Original iid session-bootstrap CI |
|---|---:|---|
| Math | -7.444 | [-16.805, +1.421] |
| Logistic | +2.630 | [-15.470, +18.032] |
| Direct LightGBM | +11.140 | [-3.273, +24.145] |
| Decomposed touch/direction | +0.459 | [-18.068, +16.143] |
| Probability correction | -0.169 | [-11.888, +10.348] |
| Fixed-weight ensemble | -1.402 | [-16.274, +11.567] |

All individual intervals include zero; six models selected exactly 126 trades. The best **point estimate** is direct LightGBM, not the ensemble. Direct minus mathematical mean is +18.584 bps per session, but only a *paired* daily interval can establish whether that difference is distinguishable from zero on this particular sample.

Identical trade counts across models suggest the selector may be filling all available daily slots. The audit exposes the utilization explicitly. Whether a model chooses the right symbols matters more here than the gross number of signals, and an uncalibrated positive EV estimate is not proof that any trade is positive EV.

**Next research step only after audit:** extend the development-period out-of-fold comparison on a larger permitted chronological sample. Do not adjust ensemble weights based on this smoke test; only consider training/calibrating a stacking model on an inner chronological fold if there is enough development history. Preserve the final protected holdout.

## Local verification limitations

23 focused synthetic tests passed (original suite plus new report-audit/type regression tests), and a synthetic full-model report was successfully audited end-to-end. Python compilation and ZIP integrity were checked. The environment lacked a downloadable Ruff/mypy installation and the complete locked project environment, so these tools and the full repository suite must be run locally.
