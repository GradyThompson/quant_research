# P4.3: Opening-range continuation hypothesis (development only)

This is a **fixed, falsifiable strategy family**, not a validated alpha or execution simulator.
It uses **existing minute OHLCV for AAPL/NVDA/MSFT/PLTR and QQQ**: no new vendor purchases.
P4.2's frequent, small-horizon ridge forecasts were generally too weak to cover
hypothetical transaction costs. We pivot to selective intraday events with 30- or
60-minute theoretical holding periods, while retaining the original holdout.

## Registered pilot: three nested variants

The opening range is the first **15 complete regular-session minutes** after
09:30 New York time (DST-aware). A breakout triggers when a **published minute
close** crosses the high/low of those 15 bars by more than **5 bps**. Check at
most the first **120 session minutes**. Each variant may enter only **once** per
session, and uses the first qualifying breakout. No stop orders are assumed.

1. `breakout`: either-side price breakout.
2. `market_confirmed`: breakout **and** QQQ published session-to-date return
   has the same sign. QQQ cannot be stale by more than 90 seconds.
3. `market_volume_confirmed`: 2, plus first-15m share volume >= **1.5x**
   the mean of the **previous ten sessions'** first-15m share volume.
   Missing volume or missing history means no entry; never use the remainder of
   the day's volume to decide a trade.

The first future minute **open strictly after** published signal availability is
used as the entry price proxy; the last of 30 or 60 uninterrupted future minute
bars' close is the exit price proxy. Both endpoints must exist within the
session. The strategy exits after fixed time, without stop or profit target.
Signals never use future candles to enter. Scoring deducts assumed flat 4, 8,
and 12 bps round-trip costs, and assigns zero return to no-trade sessions.
Each ticker/variant/horizon is separately evaluated; these are **not portfolio
returns**. Missing quotes, fees, queue positioning, funding, taxes, borrow,
short availability, fill probability, and market impact may materially change
real profit and loss.

## Leakage and selection protection

Read only pinned, permitted sessions. The original M2.5/P3 protected embargo
and final holdout remains excluded even if the requested anchor changes.
Score only the existing staged, nonoverlapping development-validation days.
No hyperparameters are optimized from validation results. Multiple variants,
horizons, and tickers are exploratory, not a familywise corrected discovery.
The first 10 permitted sessions are for trailing-volume reference (no future
volume); validation is scored on identical dates across candidates. Fold and
pooled reports include individual trade times and the exact dataset versions.

**Minimum evidence to progress:** more independent market sessions, a meaningful
number of trades spread over dates, positive net return at 8–12 bps,
positive session-bootstrap lower confidence interval (exploratory only),
consistency across time blocks, and comparison to trivial cash and buy-and-hold
benchmarks. These are research gates, not guarantees. Do not unlock the
reserved final holdout until selection rules are frozen and costs are reviewed.

## Run

```bash
for ticker in AAPL NVDA MSFT PLTR; do
    uv run python scripts/research_opening_range.py \
        --ticker "$ticker" \
        --factor-ticker QQQ \
        --factor-receipt data/dev/qqq_m3_factor_receipt_20250113.json \
        --horizon 30 \
        --horizon 60 \
        --roundtrip-cost-bps 4 \
        --roundtrip-cost-bps 8 \
        --roundtrip-cost-bps 12 \
        --output-dir "data/dev/models/${ticker}_p4_3_orb"
done
```

Existing reports are never overwritten. No trading/execution code is added.
