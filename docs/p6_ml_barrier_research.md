# P6 — ML cross-sectional three-outcome first-passage research

**Research only. No broker integration, no claims of profitability, and no final-holdout access.**

This patch applies on top of your P5-A/P4.5 repository. It adds three source/model components, two CLI scripts, and 18 tests. The models require LightGBM and scikit-learn; both are open-source libraries (no data subscription required).

## Before installing

The script **requires** these prior installed repository utilities:

- `quant_research.data.ingestion.minute_market_summary` (P4.5.1 / P5)
- `quant_research.data.ingestion.minute_research_extract` (P4.4.3)
- `quant_research.model.opening_range_history.protected_research_days`
- `quant_research.model.first_passage.first_passage_probabilities`

They are deliberately reused, not forked.

## Installation

```bash
cd ~/Documents/investing/quant_research
unzip -o ~/Downloads/P6_Integrated_ML_Barrier_Research.zip -d .
uv add 'lightgbm>=4.6.0,<5' 'scikit-learn>=1.7,<2'
```

`uv add` updates both `pyproject.toml` and `uv.lock`, avoiding overwriting the current version of your project file. Existing source files are not replaced.

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

## Build a reusable dataset (one-time heavy operation)

Initial development dataset using already downloaded Nov 2023–Apr 2024 market-wide minute flat files:

```bash
uv run python scripts/build_p6_barrier_dataset.py \
    --start 2023-11-01 \
    --end 2024-04-30 \
    --validation-start 2024-03-01 \
    --max-universe 300 \
    --progress-every 5 \
    --output-dir data/dev/models/p6_dataset_smoke
```

**This dataset uses dated Massive `/v3/reference/tickers` `type=CS` and a bulk `/stocks/v1/splits` download (both cached).** You need your existing `MASSIVE_API_KEY` in `.env`. Per-day reference data is fetched with pagination and cached under `data/dev/massive_rest_evidence/p6`. We reject missing common-stock identity and FIGI, and ignore any stock on a split ex-date. Protected original sessions are skipped *before* accessing daily manifests or minute prices. Separate daily candidate/label cache is stored at `data/dev/massive_flat_files/p6_barrier_day_cache`. This is a retrospective effective-date vendor snapshot, not an authenticated copy of what was published on that exact historical morning.

**Time:** The first run verifies the market-wide source files, reads cached daily summaries and scans selected symbols, and may perform many REST requests. It can be significantly slower than a four-symbol study; subsequent dataset rebuilds reuse the per-day source-keyed cache. Output directories are immutable. If interrupted, rerun with the *same* command: the final output directory will not have been created and completed daily caches remain available.

## Walk-forward comparison (fast relative to data ingest)

```bash
uv run python scripts/research_ml_barrier.py \
    --dataset data/dev/models/p6_dataset_smoke \
    --min-train-sessions 60 \
    --validation-sessions 20 \
    --positions 3 \
    --notional-dollars 10000 \
    --additional-cost-bps 4 \
    --additional-cost-bps 8 \
    --additional-cost-bps 12 \
    --output-dir data/dev/models/p6_ml_barrier_smoke
```

This train/validation workflow uses expanding chronological folds with a **one-session embargo** and zero overlap in scored sessions. The fixed ensemble is compared on identical out-of-fold decisions against the analytical zero-price-drift mathematical baseline, regularized logistic regression, direct three-class LightGBM, decomposed touch×direction LightGBM, and a three-outcome ratio correction to the analytical model. No model selection, ensemble weight fitting, or probability post-calibration is conducted on validation outcomes.

Read:

- `data/dev/models/p6_ml_barrier_smoke/validation.json`
- `data/dev/models/p6_ml_barrier_smoke/oof_predictions.jsonl.gz`

The report contains fold scores, pooled Brier/log-loss, observed versus predicted class frequencies, **paired daily net-return deltas against the mathematical baseline** and session-block bootstrap confidence intervals, portfolio-capped selected trades, and fixed selections across cost scenarios.

## Specific target and features

At 10:01 New York time, after the 10:00 bar has been published with a 60-second delay, features are frozen. The entry uses the first eligible minute-open **strictly after** decision time (typically 10:02). First-touch is observed against entry price at upper +0.5% and lower −2.0% during the next 60 minutes. Timeout exits at the minute-open 60 minutes after entry. If the high and low hit both barriers within the same minute, the *long stop wins* pessimistically. Gap-through stop uses the opening price, not the stop quote. Missing future bars result in omitted/censored examples, counted in the dataset manifest; they are **not** priced at zero loss.

Features consist of point-in-time returns over 5/15/30 minutes, intraday range, close-in-range, 1m realized volatility, opening relative volume (historical 20-session reference), opening dollar volume, lagged average dollar volume, market returns, and market-relative return. The last three feature slots are the zero-price-drift analytical probabilities. This is **not the seasonal, trained M2.5 forecast**; integrating the actual pinned M2.5 calibration is a subsequent milestone after this pilot.

## Distinct ensemble targets

1. **Math:** analytical barrier geometry and morning realized variance.
2. **Touch head:** LightGBM binary model for any touch vs neither.
3. **Direction head:** LightGBM binary model for upper vs lower given a touch.
4. **Timeout head:** regularized regression on *only* neither outcomes to estimate the timeout payoff.
5. **Direct head:** LightGBM three-class first-passage forecast.
6. **Correction:** analytical baseline multiplied by the square root of the learned three-class likelihood ratio relative to the training class prior.
7. **Ensemble:** predeclared weights 0.15 math + 0.25 decomposed + 0.25 correction + 0.35 direct; logits/weights are not tuned on development validation.

If an outcome is too rare in a fold, the corresponding classifier uses a Laplace-smoothed training prior (no fabricated positives). Absolute predicted probabilities need explicit calibration before live use; the report's reliability summaries are diagnostics.

## Economic metric

Rank only positive predicted expected net payoff (not raw win probability), computed from three-outcome probabilities, a training-estimated conditional lower-hit loss (minimum 200 bps), a separate timeout prediction, and estimated costs. Limit to three **long-only** $10,000 price-proxy positions per day, otherwise hold cash. Every position closes within the same trading session; no overlapping-day position issue. IBKR Pro Tiered commission approximation per side is `max($0.35, shares×$0.0035)`, where shares = $10,000 / entry price; round-trip fees and slippage are **additional** 4/8/12 bps scenarios. This is not a validated fill model or broker quote.

## Critical limitations / go-no-go

- **Historical common-stock identity:** Massive retrospective dated snapshots are effective-dated, not contemporaneous publication snapshots. Historical identity changes, FIGI gaps and delistings need stronger audit before production claims.
- **Corporate actions:** P6 excludes affected split dates and trains only intraday returns. Dividend ex-date and intraday opening data anomalies are not fully reconciled.
- **Censoring:** missing minute/exits are omitted and counted; results can be biased when halts are correlated with losses. Resolve this before any profitability claim.
- **Commission:** IBKR approximation excludes venue/regulatory fees, fill microstructure, and opportunity/impact costs. Four-basis-point additional friction is an assumed scenario, not measured.
- **Calibrated probability:** no calibrated ML probability edge has yet been established. Compare all heads to the mathematical baseline, including the *paired session-block incremental net P&L* and probability losses. Treat the smoke test as engineering and exploratory development.
- **Multiple testing:** the original 2023/2024 data have influenced many hypotheses. Do not treat positive results on those dates as new independent evidence. Preserve later protected holdout until the final frozen model evaluation.

## Longer research run (after smoke is clean)

```bash
uv run python scripts/build_p6_barrier_dataset.py \
    --start 2023-01-03 \
    --end 2024-12-31 \
    --validation-start 2024-01-02 \
    --max-universe 500 \
    --progress-every 10 \
    --output-dir data/dev/models/p6_dataset_2023_2024
```

```bash
uv run python scripts/research_ml_barrier.py \
    --dataset data/dev/models/p6_dataset_2023_2024 \
    --min-train-sessions 60 \
    --validation-sessions 20 \
    --positions 3 \
    --notional-dollars 10000 \
    --additional-cost-bps 4 \
    --additional-cost-bps 8 \
    --additional-cost-bps 12 \
    --output-dir data/dev/models/p6_ml_barrier_2024
```

Only select model/parameters after inspecting independent chronological evidence. No trading deployment is included.
