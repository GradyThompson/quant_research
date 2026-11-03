# P5-A.3 — Historical instrument and corporate-action reconciliation

**Purpose:** Determine whether the large historical residual-short winners come from real stock movements, incorrect instrument selection, split effects, or hedged-log accounting approximations. This is **not** a corrected strategy backtest.

P5-A ranked the entire ticker namespace without verified historical common-stock selection. This means leveraged/inverse ETFs and ADRs may enter a strategy intended for US common stock. A price ratio near 0.75 does not verify a 4-for-3 split. The initial score used signed log return differences, whereas actual stock and ETF holdings have arithmetic marked-to-market P&L.

This patch fetches:

- Date-specific ticker rows from `/v3/reference/tickers` using the **formation date** (ticker, market, active, date). Only an unambiguous `type=CS`, `locale=us` record with a share-class FIGI passes this diagnostic.
- Official split and stock-dividend share-change records from `/stocks/v1/splits` for all involved stock and hedge tickers over the holding intervals.
- Each source response is cached on disk under a hash of its query. Successful responses are never overwritten; a rerun resumes completed fetches. Protected price sessions are not requested.

Only report trades with both confirmed common-stock classification and complete split evidence are reconciled. Unknown records, missing FIGI, non-common instruments and missing responses are reported separately rather than presumed valid. Split adjustments use the share-count multiplier **after the entry open through the exit open**.

The arithmetic approximation (bps per original short-stock notional) is:

`-10_000 * (exp(stock_log_return) * stock_share_multiplier - 1) + beta * 10_000 * (exp(ETF_log_return) * ETF_share_multiplier - 1) - cost_per_leg_bps * (1 + beta)`.

P5-A's saved log-hedged gross, beta and stock-only result suffice to reconstruct both log returns; no historical bar reread is required. This still ignores borrow charges, interest, dividends, financing, varying hedge weights, hedge margin, market impact, missing delisted exits, and portfolio cash/equity accounting. It cannot fix a signal that was itself selected using unadjusted prices. **Post-hoc exclusions should NEVER be reported as newly validated alpha.**

## Install

Unzip the repo-relative patch into your `quant_research` repository.

## Standard verification

```bash
uv run ruff format
uv run ruff check . --fix
uv run ruff format --check .
uv run mypy src/quant_research
```

```bash
uv run pytest
```

## Run — evidence collection and reconciliation

From the repository root, with the existing `MASSIVE_API_KEY` in `.env`:

```bash
uv run python scripts/audit_p5_reference_actions.py \
  --report data/dev/models/p5_a_etf_residual_2024/validation.json \
  --collect-evidence \
  --evidence data/dev/models/p5_a_3_reference_evidence.json \
  --cost-bps 4 \
  --horizon 10 \
  --output data/dev/models/p5_a_3_reference_reconciliation.json
```

Initial evidence collection may make hundreds of REST calls. Existing per-query evidence caches are reused on restart. REST reference and split endpoints are listed by Massive as available under its current Stocks plans; actual plan entitlement/rate limiting should be verified in your account. No new subscription should be necessary.

To re-run the reconciliation **without any remote calls**, omit `--collect-evidence` and choose a new output filename:

```bash
uv run python scripts/audit_p5_reference_actions.py \
  --report data/dev/models/p5_a_etf_residual_2024/validation.json \
  --evidence data/dev/models/p5_a_3_reference_evidence.json \
  --cost-bps 4 \
  --horizon 10 \
  --output data/dev/models/p5_a_3_reference_reconciliation_repeat.json
```

## Decision after the audit

1. Inspect how many of the 642 trades were **non-common-stock** or ambiguous/unverified; inspect official split events and compare corrected arithmetic with original log score.
2. Re-run P5-A **from before signal selection**, using the existing dated security master to retain only verified historical common stocks. Apply dated split/stock-dividend transformations to the *entire* price and feature history, and use reference identity to prevent ticker-reuse errors. Handle dividends and delistings separately.
3. Replace overlapping per-entry-session performance with daily portfolio accounting (including actual hedge positions and short borrow), with chronological development folds and protected holdout untouched.

**No model promotion on the basis of this audit.**
