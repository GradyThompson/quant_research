import pytest

from quant_research.model.conditional_first_passage import ThreeWayForecast
from quant_research.strategy.expected_payoff import (
    BarrierPayoff,
    TradeDirection,
    evaluate_expected_payoff,
)


def test_favorable_long_and_short_decisions_are_symmetric():
    payoff = BarrierPayoff(0.003, 0.003, 0.0, 3)
    forecast = ThreeWayForecast(0.46, 0.30, 0.24)
    evaluation = evaluate_expected_payoff(forecast, payoff, min_edge_bps=1)
    assert evaluation.direction is TradeDirection.LONG
    assert evaluation.long_net_return == pytest.approx(0.00018)
    assert evaluation.short_net_return == pytest.approx(-0.00078)
    opposite = evaluate_expected_payoff(
        ThreeWayForecast(0.30, 0.46, 0.24), payoff, allow_short=True
    )
    assert opposite.direction is TradeDirection.SHORT
    assert opposite.selected_net_return == pytest.approx(0.00018)


def test_costs_timeout_and_minimum_edge_can_force_no_trade():
    forecast = ThreeWayForecast(0.46, 0.30, 0.24)
    no_trade = evaluate_expected_payoff(forecast, BarrierPayoff(0.003, 0.003, -0.002, 3))
    assert no_trade.direction is TradeDirection.NO_TRADE
    assert no_trade.selected_net_return == 0
    threshold = evaluate_expected_payoff(
        forecast, BarrierPayoff(0.003, 0.003, 0, 3), min_edge_bps=2
    )
    assert threshold.direction is TradeDirection.NO_TRADE


def test_cost_and_payoff_validation():
    with pytest.raises(ValueError, match="explicitly supplied"):
        BarrierPayoff(0.003, 0.003, float("nan"), 3)
    with pytest.raises(ValueError, match="nonnegative"):
        BarrierPayoff(0.003, 0.003, 0, -1)
    with pytest.raises(ValueError, match="upper barrier"):
        BarrierPayoff(-0.003, 0.003, 0, 3)
    with pytest.raises(ValueError, match="minimum edge"):
        evaluate_expected_payoff(
            ThreeWayForecast(0.5, 0.5, 0), BarrierPayoff(0.003, 0.003, 0, 3), min_edge_bps=-1
        )
