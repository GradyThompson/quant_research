# P4 pilot: independent directional-return research

## Goal and experimental status

This patch is a **research scaffold**, not an established directional edge, a
first-passage drift implementation, or a trading signal. M2.5 and M3 are not
modified. The repository context ZIP does not include the pinned OHLCV files or
validation outputs, so this patch has **not been run on the real market data**.

The existing M3 corrector fits only first-touch outcomes, keeping total touch
probability fixed. P4 instead fits continuous signed log returns, including
prices that never crossed a barrier. This is a diagnostic to determine whether
there is predictive mean information at all before building a drifted
first-passage model.

## Prespecified comparison (no feature fishing)

For each 5-, 10- and 15-minute horizon, compare four simple ridge models on
identical chronological validation decisions:

| Name | Regressors |
| --- | --- |
| `intercept` | Historical training mean return |
| `factor` | Last published 5-minute QQQ log return |
| `residual` | Stock 5-minute beta-adjusted residual z-score |
| `combined` | Both factor and residual features |

Residual direction is **learned**, not hard-coded to reversal; a negative
coefficient is reversal and a positive coefficient is continuation. Do not
count reversing the sign as a separate hypothesis. Features are standardized
using **training only**; ridge penalty applies to slopes but not the intercept.

The current code uses a non-overlapping h-minute grid within each session and
builds PIT features from synchronized, strictly published asset and factor
closes. **Publication-lag correction:** the decision occurs at the last
published bar's `available_at`, which can be 60 seconds after its bar-end
`timestamp`. The P4.2 target begins at the OPEN of the earliest complete-minute
bar starting strictly after that decision (matching the repository entry
simulator). The initial bar must begin no more than one minute later
and the horizon must be contiguous. Never assume
the next *indexed* bar has not yet started. This correction prevents an entire
study from silently returning zero rows when lag is one minute. Historical beta/scale samples exclude the signal window. A training
example is accepted only when its return label has matured by the cutoff.

**Research label (P4.2):** `log(close_of_horizon_final_bar / open_of_first_bar_strictly_after_decision)`.
This includes the post-decision price movement *after* the next bar's open.
If any minute is absent, that decision is excluded. The label is a **price
proxy, not a broker fill or quoted executable return**. Opening fills, bid/ask,
market impact, shorting, and exit fills are not modeled. Importantly, this
reference is **different** from M2.5's last-published-close barriers: do not
inject a fitted P4 return into first-passage drift without explicit
reference/horizon alignment and new tests.

The primary statistical metric is pooled out-of-fold MSE in squared basis
points. The comparison is **intercept MSE minus candidate MSE**; positive
favors the candidate. Session-block bootstrap CIs are diagnostics, not
multiple-testing-corrected discovery claims. Inspect per-fold coefficients,
per-ticker performance, and the number of scored sessions. Out-of-fold
forecast/return correlation is a secondary diagnostic. This does not measure
net profit or independent confirmation.

## Pinned history / immutable holdout

The CLI uses the original receipt loader and staged session folds. It
additionally checks the *original* 2025-01-13 M2.5/P3 protected embargo +
final holdout before loading any bars, and rejects a requested window that
would expose those days. **Never reassign the original protected dates to
training**, even if later receipts or a new evaluation anchor are introduced.

Expanding pre-anchor development history is possible if additional earlier
sessions are pinned, with proportionate `--training-sessions`,
`--validation-sessions` and `--alpha-fit-sessions` sizes. Current defaults
only yield a small number of development folds. A future platform-grade
experiment manifest should permanently track immutable protected sessions,
all tested features, source versions, decision timestamps, costs, experiment
families, and model-selection rules. More recent research should use a new
chronological sample without consuming the original protected sessions.

## Usage

```bash
for ticker in AAPL NVDA MSFT PLTR; do
    uv run python scripts/research_directional_returns_walkforward.py \
        --ticker "$ticker" \
        --factor-ticker QQQ \
        --factor-receipt data/dev/qqq_m3_factor_receipt_20250113.json \
        --horizon 5 --horizon 10 --horizon 15 \
        --output-dir "data/dev/models/${ticker}_p4_return_pilot"
done
```

Use a **new** output directory on each attempt. The program refuses to
replace existing reports. It never reads withheld source sessions, accesses
a broker, selects a winning model, or simulates fills. The test ZIP contains
source and tests only, not historical datasets.

## Next model decision

First increase permitted development coverage and evaluate the prespecified
candidates. A favorable MSE alone is insufficient for deployment. Require
coefficient stability across non-overlapping session blocks, out-of-time
replication, and **cost-aware gross-to-net analysis** on the *next-open*
reference. Account for multiple candidates, correlated securities, and
concurrent horizons.

Only if a directional return model survives these filters should we fit
`mu_log` / `mu_price` consistently with the volatility horizon and construct
an actual **drifted three-outcome first-passage solver**. All three outcomes,
including timeout, must then be re-evaluated and benchmarked against M2.5.
The original final holdout must stay unopened until the candidate and
validation procedure are frozen.

## P4.1 timing fix and troubleshooting

The original P4 pilot incorrectly excluded all candidates under the existing
60-second assumed publication delay. It required `future[0].timestamp >
current.available_at` while choosing the *next indexed* bar, which already
ended when `available_at` occurred. This patch locates the first future bar
by its **start** time and retains the original point-in-time feature guards.

Each experiment now reports `row_diagnostics` counts, and the CLI prints
per-fold training/validation row counts when no folds score. Use a fresh output
directory; the script refuses to overwrite research results. Do not interpret
previous zero-row reports as evidence against directionality.

`tests/unit/test_directional_return_walkforward_cli.py` loads the script by
its absolute file path: the src-layout installation does not necessarily put
`scripts/` on the import path when running pytest. The synthetic fold test
now uses 60-second availability lag and requires a scored fold. Additional
regression tests cover zero, 30-, 60-, and 90-second publication lags; a 120-second
lag is rejected by the pre-existing 90-second maximum staleness guard.

## P4.2 cost-aware directional decision pilot (added 2026-11-01)

**Previous P4.1 MSE findings are not proof of profitability:** on the 15
scored development sessions per security, the combined signal improved
intercept-only MSE in only 2 of 12 security/horizon combinations. Since the
same short development history has guided research, these are exploratory.

**Timing correction (P4.2):** the first future bar's `open` is now required to
occur **strictly after** `decision_at` (not equal to it), exactly matching the
existing `simulate_next_minute_entry` convention. With an assumed one-minute
publication delay, the new eligible entry is typically one minute later than
in P4.1. This invalidates direct comparisons of historical P4.1 validation
scores and requires new output directories and reruns. The target still ends
at the horizon's final complete minute close. No earlier price is used in a
point-in-time feature.

Each fold now computes, without fitting any additional models or selecting any
validation-dependent thresholds, a **fixed sign-of-forecast position rule**:

- `LONG` when `forecast_bps > assumed_roundtrip_cost_bps + extra_edge_bps`;
- `SHORT` when `forecast_bps < -assumed_roundtrip_cost_bps - extra_edge_bps`;
- otherwise abstain (`CASH`).

Defaults are **4, 8, and 12 bps** *total assumed round-trip cost* and a fixed
**2 bps** extra edge requirement. These are sensitivity assumptions, **not
measured spreads or broker charges**. The repository's minute entry simulator
uses 2 bps entry slippage and $0.005/share commission as configurable defaults;
actual cost depends on price and liquidity, includes the exit leg, and may be
higher. Market impact, crossing the spread, order queue, short locates, borrow
fees, partial fills, and the inability to fill at a specific historical
minute open remain unmodeled. Numbers in the report must be described as
**cost-adjusted historical price proxies**, never executable net P&L.

For each intercept/factor/residual/combined model and cost scenario, P4.2
reports trades, abstentions/participation, long/short counts, gross and net
bps per trade, **net bps per opportunity** (including cash observations at
zero), fraction of winning trades, and session-block bootstrap confidence
intervals. Zero trades means **no trading evidence**, not success. The cash
benchmark is 0 bps per opportunity. Out-of-fold predictions come solely
from chronologically trained models; the original holdout is excluded.

Trade decisions do not overlap *within a security and horizon* due to the
existing h-minute grid. The report does **not** aggregate multiple securities
or horizons as one portfolio or assume capital can be simultaneously allocated
to all trade windows. Positive proxy returns are only a gate for a deeper
quote-aware execution and out-of-time validation study. Do not select the
winning cost scenario, horizon, ticker, or feature using these same validation
periods and call that selected result independently confirmed.

Usage for a fresh report:

```bash
for ticker in AAPL NVDA MSFT PLTR; do
    uv run python scripts/research_directional_returns_walkforward.py \
        --ticker "$ticker" \
        --factor-ticker QQQ \
        --factor-receipt data/dev/qqq_m3_factor_receipt_20250113.json \
        --horizon 5 --horizon 10 --horizon 15 \
        --roundtrip-cost-bps 4 --roundtrip-cost-bps 8 --roundtrip-cost-bps 12 \
        --excess-edge-bps 2 \
        --output-dir "data/dev/models/${ticker}_p4_2_cost_proxy"
done
```

**Next directional milestone:** use longer, genuinely new chronological
research periods, then construct quote-aware post-decision entry/exit labels.
Before changing model families, check whether any horizon produces a positive
*per-opportunity* net cost proxy across independent sessions and securities.
Only then evaluate new predictors (event-time order-flow imbalance, lagged
signed trades, relative momentum across peers, or intraday regime effects)
on a declared time period; avoid backfitting this 15-session sample.
