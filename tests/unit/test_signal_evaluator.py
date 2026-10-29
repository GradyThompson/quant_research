from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pytest

from quant_research.evaluation import SignalEvaluator
from quant_research.exceptions import DataValidationError

T = datetime(2025, 1, 2, 21, tzinfo=UTC)


def tables(scores=(0.1, 0.2, 0.3, 0.4, 0.5), returns=(-0.05, -0.01, 0.0, 0.04, 0.08)):
    signal_rows = []
    label_rows = []
    for index, (score, future) in enumerate(zip(scores, returns, strict=True)):
        sid = f"SEC-{index + 1:06d}"
        signal_rows.append(
            {
                "security_id": sid,
                "timestamp": T,
                "available_at": T + timedelta(minutes=15),
                "signal": score,
            }
        )
        label_rows.append(
            {
                "security_id": sid,
                "timestamp": T,
                "available_at": T + timedelta(minutes=15),
                "label_end_time": T + timedelta(days=7) if future is not None else None,
                "label_available_at": T + timedelta(days=7, minutes=15)
                if future is not None
                else None,
                "forward_return": future,
            }
        )
    return pa.Table.from_pylist(signal_rows), pa.Table.from_pylist(label_rows)


def test_positive_predictive_correlation():
    signals, labels = tables()
    result = SignalEvaluator().evaluate(signals, labels)
    assert result.summary["valid_pairs"] == 5
    assert result.summary["mean_ic"] > 0.9
    assert result.summary["mean_rank_ic"] == pytest.approx(1.0)
    assert (
        result.quantile_returns.iloc[-1]["mean_forward_return"]
        > result.quantile_returns.iloc[0]["mean_forward_return"]
    )


def test_missing_outcomes_excluded_from_coverage():
    signals, labels = tables(returns=(-0.05, None, 0.0, 0.04, 0.08))
    result = SignalEvaluator().evaluate(signals, labels)
    assert result.summary["coverage"] == pytest.approx(0.8)
    assert result.daily_metrics.iloc[0]["n"] == 4


def test_constant_signals_produce_undefined_ic():
    signals, labels = tables(scores=(1.0,) * 5)
    result = SignalEvaluator().evaluate(signals, labels)
    assert result.summary["mean_ic"] is None
    assert result.summary["mean_rank_ic"] is None


def test_duplicate_keys_rejected():
    signals, labels = tables()
    signals = pa.concat_tables([signals, signals.slice(0, 1)])
    with pytest.raises(DataValidationError, match="Duplicate"):
        SignalEvaluator().evaluate(signals, labels)


def test_missing_label_rows_rejected():
    signals, labels = tables()
    with pytest.raises(DataValidationError, match="Missing label"):
        SignalEvaluator().evaluate(signals, labels.slice(0, 4))


def test_outcome_cannot_precede_signal_availability():
    signals, labels = tables()
    records = labels.to_pylist()
    records[0]["label_available_at"] = T
    with pytest.raises(DataValidationError, match="temporal alignment"):
        SignalEvaluator().evaluate(signals, pa.Table.from_pylist(records))


def test_invalid_parameters():
    with pytest.raises(ValueError):
        SignalEvaluator(quantiles=1)
    with pytest.raises(ValueError):
        SignalEvaluator(min_cross_section=2)
