"""Small reproducible end-to-end signal research experiment.

Exploratory correlations ONLY: raw OHLCV can contain split-related jumps,
fixed contemporary universes imply survivorship bias, and close-to-close
labels are not executable strategy returns.
"""

from dataclasses import dataclass

import pyarrow as pa

from quant_research.evaluation import SignalEvaluation, SignalEvaluator
from quant_research.labels import ForwardReturnLabel
from quant_research.signals import MeanReversionSignal


@dataclass(frozen=True, slots=True)
class MeanReversionExperimentResult:
    signals: pa.Table
    labels: pa.Table
    evaluation: SignalEvaluation


@dataclass(frozen=True, slots=True)
class MeanReversionExperiment:
    """Evaluate an N-observation reversal hypothesis on canonical OHLCV bars.

    The same input snapshot is used for scoring and outcomes. The scoring
    process itself sees only past observations; labels use future outcomes.
    Do not treat the resulting summary as an out-of-sample strategy backtest.
    """

    lookback: int = 5
    horizon: int = 5
    quantiles: int = 5
    min_cross_section: int = 3

    def run(self, bars: pa.Table) -> MeanReversionExperimentResult:
        signals = MeanReversionSignal(lookback=self.lookback).compute(bars)
        labels = ForwardReturnLabel(horizon=self.horizon).compute(bars)
        evaluation = SignalEvaluator(
            quantiles=self.quantiles,
            min_cross_section=self.min_cross_section,
        ).evaluate(signals, labels)
        return MeanReversionExperimentResult(signals, labels, evaluation)
