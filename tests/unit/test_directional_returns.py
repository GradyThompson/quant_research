"""Synthetic P4 tests: publication, maturity, model fitting and paired scoring."""

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from math import cos, exp, isclose, log, sin

import pytest

from quant_research.model.directional_return_scoring import (
    ReturnPrediction,
    score_return_predictions,
)
from quant_research.model.directional_returns import (
    ReturnResearchRow,
    build_return_rows,
    fit_directional_return_model,
)
from quant_research.model.first_passage_validation import MinuteBar, Session


def _minute_sessions() -> tuple[list[Session], list[Session]]:
    assets, factors = [], []
    for d, day in enumerate((date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6))):
        a_bars, f_bars = [], []
        for i in range(125):
            stamp = datetime(day.year, day.month, day.day, 14, 31, tzinfo=UTC) + timedelta(
                minutes=i
            )
            f_close = 100.0 * exp(0.0002 * i + 0.0002 * sin(i * 0.7 + d))
            a_close = 200.0 * exp(0.00025 * i + 0.001 * sin(i * 0.8 + d))
            a_bars.append(
                MinuteBar(stamp, stamp, a_close * 1.0001, a_close * 1.001, a_close * 0.999, a_close)
            )
            f_bars.append(
                MinuteBar(stamp, stamp, f_close * 1.0001, f_close * 1.001, f_close * 0.999, f_close)
            )
        assets.append(Session(day, tuple(a_bars)))
        factors.append(Session(day, tuple(f_bars)))
    return assets, factors


def _training_rows() -> tuple[list[ReturnResearchRow], datetime]:
    start = datetime(2025, 1, 2, tzinfo=UTC)
    rows = []
    for i in range(120):
        decision = start + timedelta(minutes=i)
        factor = sin(i * 0.31) * 0.001
        residual = cos(i * 0.13)
        rows.append(
            ReturnResearchRow(
                day=decision.date(),
                decision_at=decision,
                label_available_at=decision + timedelta(minutes=5),
                horizon_minutes=5,
                realized_log_return=0.00012 + 0.2 * factor + 0.00045 * residual,
                factor_return=factor,
                residual_z=residual,
            )
        )
    return rows, start + timedelta(minutes=125)


def test_build_return_rows_uses_mature_future_open_to_close_label() -> None:
    asset, factor = _minute_sessions()
    rows = build_return_rows(
        asset_sessions=asset,
        factor_sessions=factor,
        horizon_minutes=5,
        beta_lookback=70,
        min_beta_observations=60,
    )
    assert rows
    assert len({(r.day, r.decision_at) for r in rows}) == len(rows)
    first = rows[0]
    day = next(s for s in asset if s.day == first.day)
    idx = next(i for i, b in enumerate(day.bars) if b.available_at == first.decision_at)
    assert first.label_available_at == day.bars[idx + 6].available_at
    assert isclose(
        first.realized_log_return,
        log(day.bars[idx + 6].close / day.bars[idx + 2].open),
        abs_tol=1e-15,
    )
    assert first.label_available_at > first.decision_at


@pytest.mark.parametrize("lag_seconds", [0, 30, 60, 90])
def test_return_labels_respect_publication_lag(lag_seconds: int) -> None:
    assets, factors = _minute_sessions()
    lag = timedelta(seconds=lag_seconds)
    assets = [
        Session(
            session.day,
            tuple(replace(bar, available_at=bar.timestamp + lag) for bar in session.bars),
        )
        for session in assets
    ]
    factors = [
        Session(
            session.day,
            tuple(replace(bar, available_at=bar.timestamp + lag) for bar in session.bars),
        )
        for session in factors
    ]
    diagnostics: dict[str, int] = {}
    rows = build_return_rows(
        asset_sessions=assets,
        factor_sessions=factors,
        horizon_minutes=5,
        beta_lookback=70,
        min_beta_observations=60,
        diagnostics=diagnostics,
    )
    assert rows, f"lag={lag_seconds}, diagnostics={diagnostics}"
    assert diagnostics["eligible_rows"] == len(rows)
    for row in rows:
        day = next(session for session in assets if session.day == row.day)
        first = next(
            (
                i
                for i, bar in enumerate(day.bars)
                if row.decision_at
                < bar.timestamp - timedelta(minutes=1)
                <= row.decision_at + timedelta(minutes=1)
            ),
            None,
        )
        assert first is not None
        assert first + 4 < len(day.bars)
        assert isclose(
            row.realized_log_return,
            log(day.bars[first + 4].close / day.bars[first].open),
            abs_tol=1e-15,
        )
        assert row.label_available_at == day.bars[first + 4].available_at
        assert row.label_available_at > row.decision_at
    for previous, current in zip(rows, rows[1:], strict=False):
        if previous.day == current.day:
            assert current.decision_at > previous.decision_at


def test_excessive_publication_lag_is_explicitly_diagnosed() -> None:
    assets, factors = _minute_sessions()
    lag = timedelta(seconds=120)
    asset_lagged = [
        Session(s.day, tuple(replace(b, available_at=b.timestamp + lag) for b in s.bars))
        for s in assets
    ]
    factor_lagged = [
        Session(s.day, tuple(replace(b, available_at=b.timestamp + lag) for b in s.bars))
        for s in factors
    ]
    counts: dict[str, int] = {}
    rows = build_return_rows(
        asset_sessions=asset_lagged,
        factor_sessions=factor_lagged,
        horizon_minutes=5,
        beta_lookback=70,
        min_beta_observations=60,
        diagnostics=counts,
    )
    assert not rows
    assert counts["missing_factor_or_residual_feature"] > 0
    assert counts["eligible_rows"] == 0


def test_lagged_return_features_ignore_unpublished_bars() -> None:
    asset, factor = _minute_sessions()
    lag = timedelta(minutes=1)
    asset = [
        Session(s.day, tuple(replace(b, available_at=b.timestamp + lag) for b in s.bars))
        for s in asset
    ]
    factor = [
        Session(s.day, tuple(replace(b, available_at=b.timestamp + lag) for b in s.bars))
        for s in factor
    ]
    options = dict(horizon_minutes=5, beta_lookback=70, min_beta_observations=60)
    before = build_return_rows(asset_sessions=asset, factor_sessions=factor, **options)
    first = before[0]
    session_index = next(i for i, s in enumerate(asset) if s.day == first.day)
    bars = list(asset[session_index].bars)
    # This bar already ENDED but was not PUBLISHED at the decision. Rewriting
    # its close must not affect decision features, even though it changes labels.
    unavailable_index = next(i for i, b in enumerate(bars) if b.timestamp == first.decision_at)
    old = bars[unavailable_index]
    bars[unavailable_index] = replace(old, close=old.close * 1.01)
    modified = asset.copy()
    modified[session_index] = Session(first.day, tuple(bars))
    after = build_return_rows(asset_sessions=modified, factor_sessions=factor, **options)
    matched = next(r for r in after if r.decision_at == first.decision_at)
    assert matched.factor_return == first.factor_return
    assert matched.residual_z == first.residual_z


def test_future_price_change_does_not_change_decision_features() -> None:
    asset, factor = _minute_sessions()
    options = dict(horizon_minutes=5, beta_lookback=70, min_beta_observations=60)
    baseline = build_return_rows(asset_sessions=asset, factor_sessions=factor, **options)
    first = baseline[0]
    day_index = next(i for i, session in enumerate(asset) if session.day == first.day)
    bars = list(asset[day_index].bars)
    index = next(i for i, b in enumerate(bars) if b.available_at == first.decision_at)
    future_idx = index + 6
    future_bar = bars[future_idx]
    bars[future_idx] = replace(future_bar, close=future_bar.close * 1.01)
    changed = asset.copy()
    changed[day_index] = Session(first.day, tuple(bars))
    followup = build_return_rows(asset_sessions=changed, factor_sessions=factor, **options)
    matched = next(r for r in followup if r.decision_at == first.decision_at)
    assert matched.factor_return == first.factor_return
    assert matched.residual_z == first.residual_z
    assert matched.realized_log_return != first.realized_log_return


def test_gap_and_misaligned_factor_do_not_create_false_intervals() -> None:
    assets, factors = _minute_sessions()
    a = list(assets[0].bars)
    a[80] = replace(
        a[80],
        timestamp=a[80].timestamp + timedelta(seconds=5),
        available_at=a[80].available_at + timedelta(seconds=5),
    )
    assets[0] = Session(assets[0].day, tuple(a))
    observations = build_return_rows(
        asset_sessions=assets,
        factor_sessions=factors,
        horizon_minutes=5,
        beta_lookback=70,
        min_beta_observations=60,
    )
    assert all(r.decision_at != a[75].available_at for r in observations)
    with pytest.raises(ValueError, match="dates must match"):
        build_return_rows(asset_sessions=assets, factor_sessions=factors[1:], horizon_minutes=5)


def test_ridge_recovers_explanatory_features_and_intercept_baseline() -> None:
    rows, cutoff = _training_rows()
    intercept = fit_directional_return_model(rows, kind="intercept", train_until=cutoff)
    combined = fit_directional_return_model(
        rows, kind="combined", train_until=cutoff, regularization=0
    )
    held = ReturnResearchRow(
        date(2025, 1, 3),
        cutoff + timedelta(minutes=5),
        cutoff + timedelta(minutes=10),
        5,
        0.00012 + 0.2 * 0.0005 - 0.00045,
        0.0005,
        -1.0,
    )
    assert isclose(combined.predict(held), held.realized_log_return, abs_tol=1e-12)
    assert abs(intercept.predict(held) - held.realized_log_return) > 0.0001
    with pytest.raises(ValueError, match="unmatured"):
        fit_directional_return_model(rows, kind="factor", train_until=rows[0].decision_at)
    with pytest.raises(ValueError, match="after training"):
        combined.predict(rows[0])


def test_paired_scoring_is_on_identical_out_of_fold_decisions() -> None:
    start = datetime(2025, 1, 2, tzinfo=UTC)
    predictions = []
    for i in range(30):
        timestamp = start + timedelta(days=i // 6, minutes=i % 6)
        actual = 0.0005 if i % 2 else -0.0005
        predictions.append(
            ReturnPrediction(
                day=timestamp.date(),
                decision_at=timestamp,
                realized=actual,
                predictions={
                    "intercept": 0.0,
                    "factor": 0.5 * actual,
                    "residual": actual,
                    "combined": actual,
                },
            )
        )
    result = score_return_predictions(predictions, bootstrap_replicates=100)
    assert result["scored"] == 30
    assert result["scored_sessions"] == 5
    assert result["models"]["residual"]["rmse_bps"] == 0
    assert result["intercept_minus_candidate_mse_bps2"]["combined"] > 0
    assert result["intercept_minus_candidate_95pct_session_bootstrap"]["factor"][0] > 0
    with pytest.raises(ValueError, match="duplicate"):
        score_return_predictions([predictions[0], predictions[0]])
