# P5-A.1 — Mypy corrections and diagnostic short-side audit

Research only; no changes to trade selection, labels, hedge cost calculations, or source access.

## Changes

- `model/late_session_momentum.py`: retains the already issued P4.6.1 fix; distinct `first_bar`/`last_bar` variables are narrowed after `dict.get()`.
- `model/conditional_residual.py`: accumulates typed float net results while scoring; avoids converting `dict[str, object]` into float.
- `scripts/audit_p5_residual_report.py`: **read-only** audit of an existing `validation.json`. Rebuilds saved per-entry-session net returns, checks hedge costs and timestamps, summarizes ticker/factor concentration and monthly trades, and measures overlap at entry timestamps and fragility to top trades. Reads no data files and does not access holdout.
- `test_audit_p5_residual_report.py`: checks accounting, overlap, corruption detection, and missing holdout assurance.

## Run

```bash
uv run ruff format
uv run ruff check . --fix
uv run ruff format --check .
uv run mypy src/quant_research
```

```bash
uv run pytest
```

Then audit the **existing** completed report (no experiment rerun needed):

```bash
uv run python scripts/audit_p5_residual_report.py \
    --report data/dev/models/p5_a_etf_residual_2024/validation.json \
    --horizon 10 \
    --side positive_short \
    --regime all \
    --cost-bps 4 \
    --output data/dev/models/p5_a_1_positive_short_10d_audit.json
```

To audit the negative-long case, change `--side negative_long` and select a **new** `--output` filename. For 12 bps use `--cost-bps 12` if present in the original report.

## Research interpretation

`net_bps_per_entry_session` is not deployable daily return. For example, if there are 3 new positions every day with 10-session exits, a book could have around 30 concurrent positions, *before hedges*. Results must be recalculated using available capital, ongoing holdings, turnover costs, margin, and borrow costs.

The report cannot prove real trading profitability: historical common-stock identities, split/dividend adjustments, exit/delisting handling, short-locate feasibility, spreads, and ETF hedge adequacy are incomplete. The audit does not address those upstream issues; it only identifies whether observed positive gross proxies are concentrated and whether position overlap would be material.

The `range` and `trend` subgroups share the **full eligible-entry-session denominator** and their reported return contributions sum to `all`. Do not compare their raw `net/session` figures as if they were conditional return averages.

A 10-day positive-short result is a high-priority **verification lead**, not a trading strategy. Fix the data and portfolio realism before testing additional features or strategies on the same research dates.
