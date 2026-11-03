# Repository cleanup: public research project

## Keep the working research path small

Recommended daily workflow:

1. `scripts/ingest_minute_flat_history.py` — obtain historical minute data.
2. `scripts/validate_security_models.py` — fit, score, and select research policies.
3. `scripts/replay_minute_pilot_portfolio.py` — examine execution and portfolio risk.
4. `uv run pytest` — regression checks.

All development artifacts should remain under `data/dev/`. The existing pinned
receipts, source versions, trained parameters, and validation reports are
reproducibility records, not disposable caches.

## Safe disposable files (generated; not source)

These may be deleted while tools are stopped and will be regenerated:

- `**/__pycache__/`, `**/*.pyc`
- `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`
- `.coverage`, `htmlcov/`, `.hypothesis/`
- Build products: `build/`, `dist/`, `*.egg-info/`
- `.ipynb_checkpoints/`, `.DS_Store`
- `.venv/` **only if you intend to recreate it** with `uv sync`.

Do not delete `.git/`, `uv.lock`, `.python-version`, `pyproject.toml`, or
`data/dev` dataset stores. Do not delete the pinned security-master snapshots,
source receipts, or model provenance; the research pipeline references them.

## Optional manual scripts to archive (not automatically removed)

- `scripts/test_massive.py`
- `scripts/test_massive_minute.py`
- `scripts/test_storage.py`
- `scripts/test_minute_model_window.py`

These are standalone manual smoke-test commands, not the `tests/` automated
suite. Search references before removal and archive them if still useful.

- `scripts/run_private_barrier_model.py`
- `scripts/evaluate_private_barrier_signal.py`

These support an older private-model workflow that is not needed for the
current public competing-risks development model. Do **not** delete the
corresponding public model interfaces or tests unless their dependents have
been migrated. `evaluate_private_barrier_signal.py` imports the private model
loader from the other script.

### Longer-term simplification, after regression coverage

The repository contains several daily- and minute-specific ingestion and
backtesting pathways. Migrate one path at a time to the active flat-file,
receipt-pinned workflow, then remove genuinely unreferenced wrappers.
Do not delete modules solely because they have a similar name: many implement
different stages (provider, conversion, execution, or audit).

## Model optimization discipline

The public competing-risks model should remain distinguishable from the
trading policy. Validation threshold selection uses *proxies*, not actual
execution. Treat dense overlapping episodes as dependent samples. Compare
changes on unchanged validation sessions. Keep holdout data untouched until
the model and policy are frozen; January 13, 2025 is already a diagnostic date.
