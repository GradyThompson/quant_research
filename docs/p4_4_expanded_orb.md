# P4.4: Expanded opening-range research with Massive minute flat files

## Research decision

The 15-session P4.3 pilot suggested possible positive price-proxy returns for
PLTR/30m and MSFT/60m, but security/horizon selection was made **after**
examining those sessions. The pilot is exploratory only. P4.4 keeps all P4.3
signal rules fixed and tests them on a much larger chronological sample.

## Budget and scope

Existing Massive US-equities minute flat files are available to the user for
up to five years. No new market-data subscription or broker integration.
The first research pass covers AAPL, MSFT, NVDA, PLTR, and QQQ; all US-equities
coverage supports later point-in-time universe screening, but **this patch does
not claim to implement a full historical all-stocks universe**.

The new downloader is resumable and uses the repository's existing Massive
S3 downloader, content-addressed cache, and checksum manifests. It does not
read or request the original protected M2.5/P3 embargo and holdout dates.

## Methodology and boundaries

- Market-wide minute bars are still **unadjusted minute OHLCV**, not NBBO,
  borrow inventory, or executable limit/market-order fills.
- A 60-second publication lag is **assumed**, not historically verified.
- Only a published 15-minute opening range and published QQQ market direction
  may inform a trade. First eligible trade entry is strictly after decision.
- One entry per asset/day/variant. Hold positions for 30 or 60 future minutes.
- The trading rule remains **identical** across cost levels and years.
- 4, 8, 12 bps per round trip represent stress scenarios, not estimated fills.
- Validation scores use equal-weighted sessions including cash/no-trade days.
- Results are shown for the whole validation period and by calendar year.
- Multiple strategy variants, horizons, securities and years create
  multiple-testing risk; a positive best case is **not** independent evidence.
- PLTR and MSFT were chosen partly because of P4.3 results; the old 15 days
  and any data previously inspected must not be advertised as untouched testing.
- The script preserves the original protected calendar days by reading the
  original receipt **metadata only** before downloading or opening raw files.
  It also excludes the original evaluation date (2025-01-13).
- The file-streaming prototype uses ticker strings, NOT historical security-id
  snapshots. This can create identity/survivorship errors over multi-year
  histories; do not use the output as a market-wide or portfolio backtest until
  the point-in-time security-master and universe pipeline is integrated.
- Missing factor data fail closed. Missing asset dates are reported, never
  interpreted as observed zero-return opportunities.
- The script does not produce executable portfolio-level P&L. Concurrent stock
  signals can exceed any realistic capital constraint, and overnight handling,
  borrow, and corporate actions need separate validation.

## Phase A: first obtain a fully separate historical research sample

Recommended development window (without consuming later data): 2023-01-03 to
2024-12-31, with validation starting 2024-01-02. This contains a warmup year
for volume baselines and a full validation year, apart from protected dates.
Treat any previously inspected December 2024 sessions as development/reused
rather than independent confirmation. Do not use 2025–26 as automatic
holdout until the universe and model rules are frozen.

Download a manageable batch (repeat until no pending dates):

```bash
uv run python scripts/download_research_minute_flat_files.py \
    --start 2023-01-03 \
    --end 2024-12-31 \
    --max-sessions 25
```

The script downloads whole US-equity market flat files, but processes only
requested symbols during the subsequent research scan. Files can be large;
check local storage and vendor plan access before downloading many years.

Run the fixed-rule strategy across permitted dates:

```bash
uv run python scripts/research_opening_range_flat_history.py \
    --ticker AAPL \
    --ticker NVDA \
    --ticker MSFT \
    --ticker PLTR \
    --factor-ticker QQQ \
    --start 2023-01-03 \
    --end 2024-12-31 \
    --validation-start 2024-01-02 \
    --horizon 30 \
    --horizon 60 \
    --roundtrip-cost-bps 4 \
    --roundtrip-cost-bps 8 \
    --roundtrip-cost-bps 12 \
    --output-dir data/dev/models/p4_4_orb_2023_2024
```

The research script refuses to run until all unprotected trading dates in the
range have immutable, verified Massive source manifests. This is deliberate:
missing a subset of market days can bias a breakout backtest.

Quality checks:

```bash
uv run ruff format
uv run ruff check . --fix
uv run ruff format --check .
uv run mypy src/quant_research
```

```bash
uv run pytest
```

## Evaluation gate before any expanded ML

Prioritize the originally simple `breakout` variant at 8 and 12 bps assumed
round-trip costs. Require sufficiently many **independent sessions with trades**,
a positive session-bootstrap lower bound, and broad support rather than
concentration in a few sessions. Inspect annual and monthly stability and
compare against always-long, always-short and market-factor controls before
claiming any strategy is profitable. The reported session bootstrap is
exploratory; a later untouched chronological split is still required.

If the basic ORB deteriorates, do not optimize on the same validation period.
Next in priority: time-of-day conditional market momentum; volatility
compression/expansion; relative-volume breakout using a point-in-time selected
liquid universe and a larger sample. These use current OHLCV data at $0
incremental subscription cost.
