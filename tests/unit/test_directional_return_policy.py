"""Fixed, cost-aware directional policy: no validation-based tuning."""

from datetime import UTC, date, datetime, timedelta
from math import isclose

import pytest

from quant_research.model.directional_return_policy import score_return_policy
from quant_research.model.directional_return_scoring import ReturnPrediction


def _row(day: date, minute: int, realized_bps: float, model_bps: float) -> ReturnPrediction:
    t = datetime(day.year, day.month, day.day, 15, 0, tzinfo=UTC) + timedelta(minutes=minute)
    return ReturnPrediction(
        day=day,
        decision_at=t,
        realized=realized_bps / 10_000,
        predictions={"intercept": 0.0, "signal": model_bps / 10_000},
    )


def test_fixed_cost_threshold_long_short_and_cash() -> None:
    first = date(2025, 1, 2)
    second = date(2025, 1, 3)
    rows = [
        _row(first, 1, 15, 11),  # 15 - 8 = +7bps
        _row(first, 6, -15, -20),  # +15 - 8 = +7bps
        _row(first, 11, 40, 10),  # abstain at equality (8+2)
        _row(second, 1, 5, 1),  # abstain
        _row(second, 6, -2, 11),  # -2 - 8 = -10bps
    ]
    result = score_return_policy(
        rows, roundtrip_cost_bps=8, excess_edge_bps=2, bootstrap_replicates=25
    )
    m = result["models"]["signal"]
    assert m["trades"] == 3
    assert m["long_trades"] == 2
    assert m["short_trades"] == 1
    assert m["traded_sessions"] == 2
    assert m["win_fraction_net"] == 2 / 3
    assert isclose(m["total_net_bps_unit_notional"], 4)
    assert isclose(m["net_bps_per_trade"], 4 / 3)
    assert isclose(m["net_bps_per_opportunity"], 4 / 5)
    assert m["net_bps_per_opportunity_95pct_session_bootstrap"] is not None
    assert m["net_bps_per_trade_95pct_session_bootstrap"] is not None
    assert result["models"]["intercept"]["trades"] == 0
    assert result["models"]["intercept"]["net_bps_per_trade"] is None
    assert result["models"]["intercept"]["net_bps_per_opportunity"] == 0
    assert result["proxy_not_executable"] is True


def test_no_trades_does_not_become_false_positive_profit() -> None:
    rows = [_row(date(2025, 1, 2), 1, 20, 0.1), _row(date(2025, 1, 3), 1, -20, -0.1)]
    result = score_return_policy(rows, roundtrip_cost_bps=12, bootstrap_replicates=10)
    assert result["models"]["signal"]["trades"] == 0
    assert result["models"]["signal"]["net_bps_per_opportunity"] == 0
    assert result["models"]["signal"]["net_bps_per_trade"] is None


@pytest.mark.parametrize("cost,margin", [(-1, 2), (4, -1), (float("nan"), 2)])
def test_cost_inputs_are_validated(cost: float, margin: float) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        score_return_policy(
            [_row(date(2025, 1, 2), 1, 2, 4)],
            roundtrip_cost_bps=cost,
            excess_edge_bps=margin,
        )


def test_duplicate_decisions_are_rejected() -> None:
    row = _row(date(2025, 1, 2), 1, 2, 4)
    with pytest.raises(ValueError, match="duplicate"):
        score_return_policy([row, row], roundtrip_cost_bps=4)


def test_proxy_result_is_observation_order_independent() -> None:
    rows = [_row(date(2025, 1, 2), 1, 10, 20), _row(date(2025, 1, 3), 1, -10, -20)]
    a = score_return_policy(rows, roundtrip_cost_bps=4, bootstrap_replicates=10)
    b = score_return_policy(rows[::-1], roundtrip_cost_bps=4, bootstrap_replicates=10)
    assert a == b
