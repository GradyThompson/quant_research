# P4.5.2 — Cost-aware result audit (read-only)

This patch reads *only* an existing `validation.json`; it never opens market data, manifests, brokerage connections, or holdout files. It is not a new strategy or independent validation.

### Run

```bash
uv run python scripts/audit_cross_sectional_orb_report.py \
    --report data/dev/models/p4_5_1_rvol_smoke/validation.json \
    --cost-bps 12 \
    --output data/dev/models/p4_5_2_rvol_smoke_audit.json
```

The tool verifies that the daily P&L computed from archived trades agrees with the published result at the selected cost. It reports:

- 95% session-resampled confidence intervals (diagnostic, unadjusted across strategies).
- A *paired* high-RVOL minus normal-control comparison on the same dates.
- Shared event identities and distinct selected events for market-confirmed versus unfiltered portfolios.
- Exposure by long/short side, calendar month, and top-trade concentration.
- Net performance with the best 1, 3, or 5 trades removed, without replacing those trades.

**Interpretation**: 123 trades across approximately 41 sessions means the 3-slot cap is binding. Confirmation-filter and unfiltered counts can be almost identical despite different tickers being traded. The two strategies are not necessarily nested portfolios. Source data lacks historical quote-based execution prices, short-borrow information, and verified common-stock identity. Cost scenarios are stress tests, not fill estimates.

Keep P4.5.2 separate from signal design. If 2024 smoke performance survives the audit, run the *already fixed* P4.5 rule on the rest of permitted 2024 dates, and freeze it before any genuinely uninspected later window. Do not open the original protected holdout.

Verification:

```bash
uv run ruff format
uv run ruff check . --fix
uv run ruff format --check .
uv run mypy src/quant_research
```

```bash
uv run pytest
```
