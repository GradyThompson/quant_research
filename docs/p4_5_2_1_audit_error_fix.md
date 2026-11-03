# P4.5.2.1 — Audit CLI error fixes

This is a minimal correction to the read-only P4.5.2 audit. It does not modify strategy research, source market data, or the protected holdout.

Changes:
- Move `import sys` with other imports at the top of `tests/unit/test_audit_cross_sectional_orb_report.py` to satisfy Ruff E402.
- Accept either a completed research output directory or the exact path to its `validation.json` as `--report`.
- Print an actionable file-location hint rather than an uncaught `FileNotFoundError` if the report path is wrong.

Find the real report from the repository root:

```bash
find data/dev/models -type f -name validation.json | sort
```

Example (replace directory with the actual output directory):

```bash
uv run python scripts/audit_cross_sectional_orb_report.py \
    --report data/dev/models/p4_5_1_rvol_smoke \
    --cost-bps 12 \
    --output data/dev/models/p4_5_2_rvol_smoke_audit.json
```

Standard repository checks:

```bash
uv run ruff format
uv run ruff check . --fix
uv run ruff format --check .
uv run mypy src/quant_research
```

```bash
uv run pytest
```
