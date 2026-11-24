# Selected research results

## Analytical demonstration (reproducible without data)

Run `uv run --locked python examples/first_passage_demo.py` after installing the locked environment. The table below is computed by the repository's analytical solver. It is not a Monte Carlo estimate or an empirical market result.

Assumptions: +0.5% and -0.5% price barriers, volatility 0.001 per square-root minute, price-martingale drift.

| Horizon (min) | Upper first | Lower first | Neither |
| ---: | ---: | ---: | ---: |
| 5 | 0.025650 | 0.025045 | 0.949305 |
| 15 | 0.197225 | 0.195966 | 0.606809 |
| 30 | 0.355498 | 0.354787 | 0.289715 |
| 60 | 0.467121 | 0.466956 | 0.065922 |

## Documented development findings

| Experiment | Reported observation | Source |
| --- | --- | --- |
| 2023/2024 fixed-stock opening-range breakouts and 2024 cross-sectional high-RVOL | Did not establish profitability after assumed 12 bps round-trip costs | [Research note](../docs/p4_6_late_session_momentum.md) |
| AAPL validation, 50 fully evaluated forecasts | 39 timed out; binary upper-touch labels do not capture timeout returns | [Model-selection note](../docs/maintenance/next_model_decision.md) |
| Historical residual-short audit | Investigated security classification, share changes and arithmetic versus log P&L | [Audit methodology](../docs/p5_a_3_reference_action_reconciliation.md) |

These are observations recorded in existing development notes, not newly reproduced findings or independent holdout results. The original generated reports are not included in the repository snapshot.

## Adding empirical results

After checking vendor redistribution terms, add aggregate results from existing saved reports. For probability comparisons, extract Brier score, log loss, number of evaluated decisions and sessions, evaluation dates and target definition. Compare only models scored on identical observations. For return comparisons, record cost assumptions, execution proxy, sample period and uncertainty. Clearly mark development results; do not open the protected holdout solely for a portfolio release.

Do not publish credentials, raw licensed market data, proprietary strategy files or report fields containing restricted data. Do not infer missing values or use illustrative numbers as empirical results.
