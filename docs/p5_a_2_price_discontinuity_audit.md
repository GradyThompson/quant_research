# P5-A.2 — Raw-price discontinuity audit

This **read-only** diagnostic investigates P5-A's positive-residual short results. It does not access protected datasets, refit or retune signals, verify corporate actions, or generate portfolio P&L.

P5-A computes positive-short stock profit as the negative of an **unadjusted** opening-price log return. A 4-for-3 split makes the raw stock price approximately 25% lower and creates a false short gain of about +2,877 bps if shares outstanding and split adjustment are ignored. The original daily 30% and whole-hold 40% log discontinuity filters do not catch every possible split ratio.

The audit reconstructs each underlying stock's raw open-to-open return from the stored unhedged-short diagnostic and the reported per-leg transaction cost. It identifies (1) declines of at least 15% and (2) price ratios within 2% of several split-like ratios, with no claim that any are confirmed splits. It calculates the contribution from these flagged trades and retains the **original entry-session denominator** when excluding them, treating removed trades as abstentions. It reports dates and tickers for the top trades, so they can be reconciled against a point-in-time split history.

## Run

```bash
uv run python scripts/audit_p5_split_risk.py \
    --report data/dev/models/p5_a_etf_residual_2024/validation.json \
    --horizon 10 \
    --cost-bps 4 \
    --top 20 \
    --output data/dev/models/p5_a_2_split_risk_audit.json
```

Then run the standard repository verification:

```bash
uv run ruff format
uv run ruff check . --fix
uv run ruff format --check .
uv run mypy src/quant_research
```

```bash
uv run pytest
```

## Interpretation and next gate

A split-like price ratio is **not** proof of a stock split: genuine price declines can resemble it. The experiment should not be promoted even if it survives the exclusion test. The next research milestone must use historical split/dividend-adjusted prices, point-in-time security identities, delisting returns, and borrow availability, followed by capital-constrained book accounting. Do not replace verified corporate actions with permanent 15% censoring: that would selectively eliminate difficult short positions and bias results.
