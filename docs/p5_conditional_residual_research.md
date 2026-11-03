# P5-A — Conditional residual shocks: research pilot

**Status:** exploratory signal-discovery tool. **Not** a historical security-master-correct, corporate-action-adjusted or executable strategy backtest.

## Question

Do unusually negative (positive) five-session stock moves relative to a *preselected lagged ETF factor model* predict subsequent three-, five- or ten-session factor-relative recoveries (continuation)? Is any effect conditional on how range-bound the S&P 500 has recently been?

The ETF factor is selected per security by the highest positive squared correlation fitted on the *60 daily returns strictly before the five-session signal formation window*. The factor exposures, intercept and residual standard deviation are frozen before measuring the shock. Candidate stocks are the 300 most historically dollar-liquid symbols, using the preceding 20 sessions only, with preceding average daily dollar volume at least $50 million and current price at least $20. Signals require absolute formation residual z-score at least 2. The three most negative and three most positive qualifying shocks are retained per decision session. Comparisons use standardized opposite-direction trades, separated by side.

Daily trend state is the ratio `abs(sum(SPY daily returns)) / sum(abs(SPY daily returns))` over the last 60 sessions. `range` is defined as <= 0.25, `trend` as > 0.25, *not* optimized on the current data.

## Timing and returns

- Decision after formation-day regular-session close.
- Long negative residual / short positive residual in the next **regular-session opening-minute bar open**, priced as a proxy.
- Exit at the opening-minute bar open after 3, 5 or 10 held exchange sessions.
- The putative factor-hedged profit (in basis points per $1 stock notional) is `direction × (stock open-to-open return − beta × ETF open-to-open return)`.
- Assumed round-trip stock cost `C` and ETF hedge cost `abs(beta) × C` are both deducted. There is no explicit borrow, funding, borrow availability, margin, spread or fill model.
- Baseline results report mean net bps per stock trade and *entry session* at up to three fixed slots per side. **Since the horizon exceeds one day, overlapping positions are not reconciled into a capital-limited portfolio. Do not treat session bps as daily portfolio returns.**
- Confidence intervals use 10-day moving blocks sampled from a vector of entry-session returns. These are descriptive, unadjusted for the related tests or outcome-dependent strategy choice.
- Price discontinuities of 30% or more in daily closes and 40% or more across entry/exit opens are suppressed. This heuristic does **not** replace split or dividend adjustments.
- Future window with a missing/protected session is censored, never bridged or assigned an invented close. Missing exits are counted in diagnostics.

## Research integrity

- Protected original receipt dates are excluded **before** any source verification, cache load or scan. Protected dates are preserved as empty holes in the exchange calendar so formation and exit windows cannot silently bridge them.
- Each authorized minute flat file is SHA256-checked against its existing immutable manifest before its market-wide daily cache can be used. No brokerage credentials are used.
- Historical stock identity is **not** yet verified. Ticker-like symbols can include ETFs, ADRs, ticker changes or delistings. Successful open-to-open exit observations may be biased by missing/delisted securities.
- Correlation-based ETF factor selection is **not verified industry classification**. The initial P5-A cannot test the published *industry-adjusted* effect directly. For that, the next milestone must use historical stock identity, point-in-time industry mappings and corporate-action-adjusted daily prices.
- 2023–2024 are previously examined development dates, not an untouched validation set. Positive P5-A results would be leads for further research, not proof of profitability.
- The experiment is explicitly set up to reject simplistic always-revert hypotheses: negative-long vs positive-short shocks and low-trend vs high-trend regimes are reported separately.

## Install

Unzip into the repo root. The patch adds new files only and does not overwrite M2.5 or P4 experiments.

## Local verification

```bash
uv run ruff format
uv run ruff check . --fix
uv run ruff format --check .
uv run mypy src/quant_research
```

```bash
uv run pytest
```

## Run on already downloaded 2023–2024 Massive minute data

```bash
uv run python scripts/research_conditional_residual.py \
    --start 2023-01-03 \
    --end 2024-12-31 \
    --validation-start 2024-01-02 \
    --max-universe 300 \
    --progress-every 10 \
    --output-dir data/dev/models/p5_a_etf_residual_2024
```

Full-year input is required for the 60-session beta fit and 20-session liquidity lookback. This run reuses `market_summaries` source-pinned P4.5 cache when available. `validation.json` is immutable (no overwrite).

For a faster synthetic smoke check using existing 2023 files, choose a shorter **development** evaluation window but retain at least 65 prior exchange sessions for estimating the factor and formation window. For example:

```bash
uv run python scripts/research_conditional_residual.py \
    --start 2023-01-03 \
    --end 2023-07-31 \
    --validation-start 2023-05-01 \
    --max-universe 100 \
    --output-dir data/dev/models/p5_a_etf_residual_smoke
```

This evaluates only the earliest part of the downloaded development period. It does not create independent out-of-sample data.

## How to interpret

Inspect `candidate_diagnostics` and `total_scored_events` first. If there are many `missing_or_discontinuous_exit` observations, the results **cannot** be promoted even if returns appear positive. For each 3/5/10-day horizon and negative/positive shock, compare `all`, `range`, and `trend`. A convincing signal requires robust incremental factor-relative returns **after both-leg costs**, good sample sizes, multiple yearly windows, stable performance excluding the biggest winners, and meaningful results under separately held-back dates. Do not select the best of 18+ related tests as an independent discovery.

## Next implementation gate

If a signal is statistically promising, do not immediately deploy it. Next integrate: a versioned historical security-master mapping, corporate actions, verified industry classification, delisting settlement values, a turnover-aware overlapping-position portfolio, IBKR-based commissions plus execution/borrow stress assumptions, and a frozen forward evaluation. Only then consider multi-day variance/barrier modeling.
