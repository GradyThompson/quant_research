# P6.1.3 — Python 3.13 multiprocessing warning fix

## Symptom

During `uv run pytest` with Python 3.13, `multiprocessing/popen_fork.py` emits:

> DeprecationWarning: This process is multi-threaded, use of fork() may lead to deadlocks in the child.

This warning is not a failing assertion or proof that the dataset is corrupt. It comes from the default `fork` start method used by `ProcessPoolExecutor` on Linux, in particular when pytest or a dependency has already started threads.

## Change

`build_p6_barrier_dataset.py` now uses `forkserver` when the operating system supports it, otherwise `spawn`, when `--workers > 1`. A one-worker run retains the serial behavior. Both methods create workers without directly forking a multithreaded Python process.

Because `forkserver` and `spawn` must import worker functions by their real module name, the multiprocessing regression test now imports the builder as the normal `build_p6_barrier_dataset` module instead of loading it under a private dynamic name. No barrier, screening, source-receipt, cache-key, or protected-date logic changed.

## Running build

The build you already started can continue. Changing the file on disk will **not** change the start method in an already-running interpreter. Install the patch when convenient for your *next* invocation and to remove test warnings.

## Installation

Unzip into the project root, preserving repository-relative paths:

```bash
unzip -o ~/Downloads/P6_1_3_Python313_Process_Start_Fix.zip -d .
```

Standard checks:

```bash
uv run ruff format
uv run ruff check . --fix
uv run ruff format --check .
uv run mypy src/quant_research
uv run pytest
```

Optional strict warning regression:

```bash
uv run pytest tests/unit/test_ml_barrier_dataset_performance.py -W error::DeprecationWarning
```

## Verification scope

In a local reconstructed project snapshot, 20 P6 focused tests passed, including serial/parallel result equivalence, day-cache reuse, and protected-date exclusion, with `DeprecationWarning` configured as an error. A test-only stub replaced an unrelated optional `pyarrow` ingestion-package import in that local snapshot; the production patch does not change ingestion imports. Full repository lint/type checks were not run in the author's environment.
