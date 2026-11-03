# P4.5 — Market-wide abnormal-volume continuation (research only)

## Rationale and research hypothesis

P4.4 failed the 2023 and 2024 cost-adjusted fixed-stock breakout screen. The
sole positive 2024 PLTR relative-volume result did not repeat in 2023. Do not
promote any P4.4 strategy. P4.5 is a separate hypothesis: a 15-minute opening
range breakout occurring in an unusually active, historically liquid symbol may
have better forward returns than a similarly liquid, ordinary-activity symbol.

P4.5 compares *the same fixed first breakout event* under an unconfirmed and
QQQ-confirmed filter. Confirmed portfolios cannot wait for a later event after
rejecting the first. This prevents a variant-dependent opportunity definition.

## Point-in-time definition

- Mandatory previous 20 consecutive exchange-session summaries. A missing
  summary or protected session breaks the history; future data cannot fill gaps.
- Previous completed session close >= $10, same-day first open >= $10.
- Previous 20-session average dollar volume >= $20 million/day. This is a
  liquidity proxy, not observed spreads or executable depth.
- Current first 15 complete minute bars are published at 09:46 ET under a
  one-minute delay. Current first-15 volume / prior first-15 volume >= 1.5.
- Select at most 20 highest-RVOL symbols. For each, pick a unique 0.8–1.2 RVOL
  control stock with nearest prior average dollar volume. Both are ranked using
  information known by 09:46 ET.
- Exclude opening gaps exceeding 30% in magnitude as a crude corporate-action
  guard; this also excludes genuine large news events and is not a substitute
  for adjusted point-in-time data.
- Detect first 5 bp breakout beyond the initial opening range between minute
  16 and 120. Use completed, published bars. Execution proxy is the first
  one-minute open *strictly after* the decision instant. Fixed holding periods
  of 30 and 60 minutes, closed at the future bar close.
- If a qualified stock stops trading after the opening range, it remains a
  valid candidate at selection time; its future trade is rejected only if a
  complete execution window cannot be constructed. Never exclude a candidate
  using future end-of-day completeness.
- Two candidate cohorts: high RVOL and matched ordinary RVOL. Two policies
  per cohort: first breakout and first breakout filtered by concurrent QQQ
  direction. QQQ confirmation uses only published factor data.
- The first three actionable events per day, sorted by decision time (with
  relative volume as a tie-breaker), consume one third of notional each.
  Remaining slots are held in cash; each slot is used at most once per day.
  Assumed costs are 8, 12, and 20 bp round-trip, with *identical trades*.
- Session bootstrap CI resamples days, not independent trades.

## Limitations—do not call this validated trading performance

**Flat-file symbols do not establish common-stock identity.** The alpha-only
symbol screen excludes some legitimate issues and can include ETFs, ADRs, and
historically changing tickers. The existing security master can enforce common
stock membership only when point-in-time listing/identity coverage is verified
for every scoring session. Thus this P4.5 output is a discovery *pilot*, not a
final survivorship-free US common-stock strategy study.

Minute-bar opens and closes are not bid/ask quotes. Missing spread, impact,
short borrow, halted-sale liquidity, and execution timing can overstate returns.
An event-selected cohort may trade only on unusual activity; controls are
matched on *prior dollar liquidity* but not on sector or price volatility.
Corporate actions are not adjusted. A 30% gap exclusion is incomplete.

Year 2023 and 2024 data have already influenced research; neither year is an
untouched holdout. All reserved original holdout dates are excluded BEFORE
source verification or cache opening. No brokerage access is introduced.

## Install

Run from repository root:

```bash
unzip -o ~/Downloads/P4_5_Cross_Sectional_RVOL_Research.zip -d .
```

## Verification

```bash
uv run ruff format
uv run ruff check . --fix
uv run ruff format --check .
uv run mypy src/quant_research
```

```bash
uv run pytest
```

## Small first run (uses already downloaded 2023–2024 market-wide files)

```bash
uv run python scripts/research_cross_sectional_orb.py \
    --start 2023-01-03 --end 2024-02-29 \
    --validation-start 2024-01-02 \
    --progress-every 10 \
    --output-dir data/dev/models/p4_5_rvol_smoke
```

The smoke period is exploratory and not valid for claiming statistical
significance. For the full 2024 development evaluation:

```bash
uv run python scripts/research_cross_sectional_orb.py \
    --start 2023-01-03 --end 2024-12-31 \
    --validation-start 2024-01-02 \
    --progress-every 10 \
    --output-dir data/dev/models/p4_5_rvol_2024
```

Output: `validation.json`, with selected trade timestamps, cohorts, costs,
fixed-slot portfolio daily return and confidence intervals. The first run
parses each entire market-wide source once and caches summaries by verified
source hash; it then selectively extracts the day's ranked symbols for any
validation sessions. Subsequent runs reuse both caches but still hash source
files. Expect meaningful first-pass I/O and parsing time for two years.

## Decision gates

Reject when after-cost results are negative, trade counts are too small, gains
are concentrated in a few events, or the high-RVOL cohort does not outperform
an ordinary-volume control. If promising, first integrate verified historical
security identities and independently test predeclared rules on genuinely new
later dates; then address real bid/ask fills and portfolio execution risk.
No new data subscriptions are required for this pilot.
