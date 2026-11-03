# Profit-aware barrier model: experiment and profit check

## Status and why this model

The competing-risks hazard model predicts whether an upper or lower barrier
will be hit, or whether neither is reached. These are not equivalent to
**positive net trade P&L**: most AAPL validation observations were timeouts,
and their close prices matter. No algorithm can be declared the "best possible"
without a comparable, untouched out-of-sample sample and defined costs.

This experiment adds a small **profitability classifier** rather than another
large architecture. Its supervised target is `net P&L per share > 0`, with
future entry, stop-first same-bar resolution, gap-aware stop, horizon timeout,
both sides' slippage and per-share commissions. Its only inputs are causal
OHLCV-based features from `competing_risks._feature_base`.

The model uses one regularized binary logistic regression with seven varying causal features,
**equal total weight per training session** and unpenalized intercept. It learns the
probability that a trade would have made money. It also stores the
training-only average positive and nonpositive returns to estimate expected
net trade return. No retraining on validation or holdout outcomes occurs.

This is a strong, interpretable baseline—not a proven profit-optimal model.
The results should be compared against the public competing-risks model on
identical chronological windows, execution assumptions and opportunity limits.

## Files

- `src/quant_research/model/profit_execution.py`: reusable, fast historical
  net trade labeling. Regression tests compare it against the existing
  `backtesting.minute_trade` simulator.
- `src/quant_research/model/profit_model.py`: training, prediction and safe
  numerical model serialization.
- `src/quant_research/model/profit_research.py`: chronological split, precision
  threshold selection, and cash-sized single-security profit replay.
- `scripts/run_profit_check.py`: command-line research and optional holdout.
- `src/quant_research/model/profit_portfolio.py`: shared-cash historical
  portfolio replay with concurrent position caps and a daily loss halt.
- `tests/unit/test_profit_*.py`: alignment, learning, policy, integration tests.

The public hazard model and previous scripts are untouched. No private
strategy directories or models are involved.

## Run on local pinned AAPL data

First run formatting, lint, typing, and tests:

```bash
uv run ruff format
uv run ruff check . --fix
uv run ruff format --check .
uv run mypy src/quant_research
```

Then:

```bash
uv run pytest
```

Run a **validation-only** profit check. Existing receipt and dataset store are
reused; no S3/download step and no private model is needed.

```bash
uv run python scripts/run_profit_check.py \
    --ticker AAPL \
    --training-stride 5 \
    --output-dir data/dev/models/aapl_profit_trial_01
```

Output contains:

- `profit_model.npz`: trained coefficients, train-only payoff means, costs.
- `profit_check.json`: source receipt hash, frozen chronological split,
  training composition, validation AUC and log-loss, all threshold results,
  selected/abstained decision and, if approved, a sequential cash replay.

The validator requires at least 10 selected opportunities, a 5% selection
rate, positive observed validation mean net return and a 50% **one-sided 95%
Wilson precision lower bound**. All selected opportunities must also have
positive *predicted* net EV. If no threshold qualifies, the model **does not
trade**. Do not relax these rules simply to obtain a positive backtest.

### Sealed holdout: one-time evaluation only

Evaluate the reserved 10 holdout sessions **only after** the model,
threshold search protocol and costs are frozen. This reads the holdout only
if validation selected a policy:

```bash
uv run python scripts/run_profit_check.py \
    --ticker AAPL \
    --training-stride 5 \
    --output-dir data/dev/models/aapl_profit_sealed_01 \
    --evaluate-holdout
```

Each output directory must be new; existing outputs are never overwritten.
Using a new directory does **not** make a previously inspected holdout fresh.
Repeated iteration on the same ten AAPL validation sessions already weakens
claims of unbiased model selection. Prefer a new period for future trials.

### Execution conventions

- Minute `timestamp` is the **bar end**; entry is at the first strictly
  subsequent eligible bar **open** after `available_at` of the decision.
- The complete `horizon` must be observable and contiguous. The trade cannot
  cross the session end.
- Entry: open plus buy-side slippage; exit: target, stop, or last bar close
  minus sell-side slippage. Both touches in the same bar resolve to **stop**.
- Stop gaps use the lower opening price. Favourable target gaps are capped
  at the target. Commission is applied once on entry and once on exit.
- Precision means **selected positive-net trades / selected trades**, not
  upper-first classification precision.
- Replay sizes each trade at up to 10% of available cash and enforces one
  open position per security. A single-security cash account is used.

### Limitations before trading

- OHLC bars cannot establish exact executable fills, spreads, queue priority,
  market impact, borrow restrictions or minute-level publication latency.
- Simulation cash drawdown is **realized-only**, not mark-to-market intratrade.
- Dense overlapping training episodes remain dependent; 38 sessions, not the
  large episode count, determines the broad independent information content.
- This is **single-security**, not a diversified portfolio optimizer.
- Selected thresholds are tuned to validation outcomes. The validation
  replay is not an independent out-of-sample profit result.
- Pinned historical security identity and bar availability verification flags
  are copied into the output; they may remain unverified.
- Real-world expected profits are unknown. The correct output can be
  **no approved trade policy**.

## Later improvements (only after a comparable backtest)

1. Extend chronological training and evaluate across many securities and
   independent dates with equal transaction-cost and risk settings.
2. Consider monotonic calibration (training-only cross-fitting) or a small
   nonlinear model only if baseline discrimination is stable; do not optimize
   hyperparameters against holdout.
3. Run a portfolio-level replay with simultaneous opportunities, capital
   reservations, maximum daily loss, fill sensitivity and stress costs.
4. Report session-bootstrap uncertainty and opportunity-adjusted precision,
   not simply a threshold's observed success rate.

## Check multiple securities with shared capital

This runs the same model architecture and predeclared policy requirements on
each symbol. Each model and threshold is trained/selected independently,
then approved trades are combined into one shared-cash replay with a maximum
of three concurrent positions, 10% initial per-position equity allocation
and a 2% daily realized loss halt for new entries. This is not a full
real-time order-book simulator.

```bash
uv run python scripts/run_profit_batch.py \
    --tickers AAPL,NVDA,MSFT,PLTR \
    --output-dir data/dev/models/profit_batch_trial_01
```

Each symbol has its own model and report; `batch_summary.json` records which
symbols abstained, validation profit AUC and whether the holdout was opened.
The batch also writes `portfolio_validation.json`, calculated from policies
selected on the SAME validation dates (so it is not independent evidence).

Only with a finalized protocol and a genuinely reserved holdout:

```bash
uv run python scripts/run_profit_batch.py \
    --tickers AAPL,NVDA,MSFT,PLTR \
    --output-dir data/dev/models/profit_batch_sealed_01 \
    --evaluate-holdout
```

When a holdout is actually opened, the batch writes `portfolio_holdout.json`
with realized equity, precision and per-trade ledger. The portfolio simulator
reserves cash, processes exits in timestamp order, limits open positions and
prevents new entries after the daily realized loss limit. It uses selected
per-security replay orders and does **not** model order-book fills, partial
fills, bid/ask spread dynamics, mark-to-market drawdown or intraday buying
power rules. Do not add per-security P&Ls together to obtain this figure.
