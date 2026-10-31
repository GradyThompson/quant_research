"""Verify same-sample benchmark arithmetic and oracle labeling."""

from math import log

import pytest

from quant_research.backtesting.calibration_benchmarks import (
    score_constant_baseline,
    score_retrospective_frequency_baseline,
    score_uniform_baseline,
)


def test_uniform_baseline() -> None:
    result = score_uniform_baseline([0, 1, 2, 2])
    assert result.count == 4
    assert result.brier_score == pytest.approx(2 / 9)
    assert result.log_loss == pytest.approx(log(3))


def test_retrospective_frequency_is_in_sample() -> None:
    result = score_retrospective_frequency_baseline([0, 0, 1, 2])
    assert result.probabilities == (0.5, 0.25, 0.25)
    assert result.brier_score is not None
    assert result.log_loss is not None
    assert result.brier_score < 2 / 9


def test_empty_and_invalid_inputs() -> None:
    result = score_uniform_baseline([])
    assert result.count == 0
    assert result.brier_score is None
    with pytest.raises(ValueError, match="sum to 1"):
        score_constant_baseline([0], probabilities=(0.2, 0.2, 0.2))
    with pytest.raises(ValueError, match="Actual class"):
        score_uniform_baseline([3])
