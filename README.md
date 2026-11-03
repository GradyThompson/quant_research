# quant_research

Research repository for historical, point-in-time U.S. equity market data, mathematical intraday barrier forecasts, and development-only trading simulations.

## Current probability model (M2.5)

The current research model is **finite-horizon Brownian first passage**, not a learned directional classifier. Given an upper price barrier, a lower price barrier and a holding horizon, it reports the probabilities of reaching the upper barrier first, the lower barrier first, or neither barrier. `price_zero` is the price-martingale drift convention; `log_zero` is retained for sensitivity checks.

Volatility methods: `close`, `parkinson`, `seasonal_close`, and `seasonal_parkinson`. The seasonal variants fit opening/middle/closing intraday variance factors **only on designated training sessions**. Earlier embargo bars may warm up rolling volatility but are not used for parameter fitting. The 10-session development validation is separated from an untouched reserved holdout.

Calibration and benchmark entry point:

```bash
uv run python scripts/validate_first_passage.py \
    --ticker AAPL --per-scenario-stride \
    --volatility-method close --volatility-method seasonal_close \
    --scenario 0.002,0.002,5 \
    --output-dir data/dev/models/AAPL_validation_example
```

Read `docs/first_passage_seasonality.md` and `docs/first_passage_m25_validation.md` for mathematical and validation details. Calibration and Brier improvements are **not evidence of positive expected trading returns**.

## Preliminary net-profit replay

The new `scripts/replay_first_passage_profit.py` is a **development-only diagnostic**. It evaluates precomputed `price_zero` forecasts, enters at the next minute's reported open, applies target/stop/time exits, and charges configured spread, slippage and per-share commissions. It assumes one security at a time and one non-overlapping position. The defaults are illustrative, **not an IBKR rate quote**.

```bash
uv run python scripts/replay_first_passage_profit.py \
    --ticker AAPL --scenario 0.002,0.002,5 \
    --volatility-method seasonal_close --policy model_proxy \
    --output-dir data/dev/models/AAPL_profit_proxy_v1
```

A fixed comparator uses `--policy always_long` or `--policy always_short` and a **different** output directory. Do not optimize across validation outcomes or open the withheld holdout to tune an entry rule.

**Limitations:** Minute OHLC lacks bid/ask quotes and intrabar ordering. The model's barriers are referenced to the known close, whereas executable entries and exits use the *subsequent* open; some calibration forecasts with ambiguous first touches are absent from this replay. Stop-before-target is assumed when both prices appear in the same minute. The result is not a broker-fill simulation, cannot establish live profitability, and the `model_proxy` heuristic omits the conditional return on timeouts. A theoretically fair price martingale offers no expected gross trading profit from first-passage probabilities alone.

## Data and reproducibility

Research uses version-pinned Massive minute datasets under `data/dev/` and a training receipt (normally `data/dev/minute_training_receipt_20250113.json`). Replays refuse to overwrite an existing output directory. Preserve original receipts and generated reports. Do not treat January 13, 2025 engineering diagnostics as an untouched holdout.

## Checks

```bash
uv run ruff format
uv run ruff check . --fix
uv run ruff format --check .
uv run mypy src/quant_research
```

```bash
uv run pytest
```

## Repository layout

- `src/quant_research/data/`: market data ingestion and storage.
- `src/quant_research/security_master/`: historical identities.
- `src/quant_research/model/`: first-passage mathematics, volatility, calibration, benchmarks, and execution approximations.
- `scripts/`: reproducible research commands.
- `tests/unit/`: offline regression checks.
- `data/dev/`: development datasets, receipts and generated reports.

To bundle context:

```bash
uv run python scripts/bundle_project.py --output ~/Downloads/project_context.zip
```
