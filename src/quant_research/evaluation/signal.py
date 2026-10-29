"""Exploratory cross-sectional signal evaluation (not executable backtesting).

Signals are joined to labels by (security_id, timestamp). Forward outcomes
are used only for retrospective evaluation, never for decision-time features.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa

from quant_research.exceptions import DataValidationError


@dataclass(frozen=True, slots=True)
class SignalEvaluation:
    observations: pd.DataFrame
    daily_metrics: pd.DataFrame
    quantile_returns: pd.DataFrame
    summary: dict[str, float | int | None]


@dataclass(frozen=True, slots=True)
class SignalEvaluator:
    quantiles: int = 5
    min_cross_section: int = 3

    def __post_init__(self) -> None:
        if isinstance(self.quantiles, bool) or self.quantiles < 2:
            raise ValueError("quantiles must be >= 2")
        if isinstance(self.min_cross_section, bool) or self.min_cross_section < 3:
            raise ValueError("min_cross_section must be >= 3")

    def evaluate(self, signals: pa.Table, labels: pa.Table) -> SignalEvaluation:
        """Evaluate matching signal/label rows using only completed outcomes.

        Returns Pearson IC, Spearman rank IC, and mean forward returns for
        ranked quantile groups. IC is undefined for constant inputs.
        Groups are descriptive, not a simulated portfolio.
        """
        if not isinstance(signals, pa.Table) or not isinstance(labels, pa.Table):
            raise DataValidationError("signals and labels must be Arrow tables")
        required_signal = {"security_id", "timestamp", "available_at", "signal"}
        required_label = {
            "security_id",
            "timestamp",
            "available_at",
            "label_end_time",
            "label_available_at",
            "forward_return",
        }
        if not required_signal.issubset(signals.column_names):
            raise DataValidationError("Signal columns missing")
        if not required_label.issubset(labels.column_names):
            raise DataValidationError("Label columns missing")

        sf = signals.select(sorted(required_signal)).to_pandas()
        lf = labels.select(sorted(required_label)).to_pandas()
        keys = ["security_id", "timestamp"]
        if sf.duplicated(keys).any() or lf.duplicated(keys).any():
            raise DataValidationError("Duplicate security/timestamp keys")
        joined = sf.merge(
            lf,
            on=keys,
            how="left",
            suffixes=("_signal", "_label"),
            validate="one_to_one",
            indicator=True,
        )
        if (joined["_merge"] != "both").any():
            raise DataValidationError("Missing label row for signal observation")
        joined = joined.drop(columns="_merge")
        # Reject logically impossible temporal alignment, including if the
        # outcome is allegedly available when the signal first becomes usable.
        valid_outcomes = joined["forward_return"].notna()
        bad = valid_outcomes & (
            (joined["label_end_time"] <= joined["timestamp"])
            | (joined["label_available_at"] <= joined["available_at_signal"])
            | (joined["available_at_label"] > joined["available_at_signal"])
        )
        if bad.any():
            raise DataValidationError("Invalid signal/label temporal alignment")
        joined["signal"] = pd.to_numeric(joined["signal"], errors="coerce")
        joined["forward_return"] = pd.to_numeric(joined["forward_return"], errors="coerce")
        good = (
            joined["signal"].notna()
            & joined["forward_return"].notna()
            & np.isfinite(joined["signal"])
            & np.isfinite(joined["forward_return"])
        )
        joined["valid_pair"] = good
        joined["quantile"] = pd.Series(pd.NA, index=joined.index, dtype="Int64")
        metric_rows: list[dict] = []
        quantile_rows: list[dict] = []
        for timestamp, frame in joined.groupby("timestamp", sort=True):
            sample = frame.loc[frame["valid_pair"]]
            n = len(sample)
            ic = None
            rank_ic = None
            if n >= self.min_cross_section:
                x, y = sample["signal"], sample["forward_return"]
                if x.nunique() > 1 and y.nunique() > 1:
                    ic = float(x.corr(y, method="pearson"))
                    rank_ic = float(x.corr(y, method="spearman"))
            metric_rows.append({"timestamp": timestamp, "n": n, "ic": ic, "rank_ic": rank_ic})
            if n >= max(self.quantiles, self.min_cross_section):
                # Average tied ranks prevent identical signals being forced
                # into different groups. Some groups may consequently be empty.
                ranks = sample["signal"].rank(method="average", pct=True)
                bins = np.minimum(np.ceil(ranks * self.quantiles).astype(int), self.quantiles)
                joined.loc[sample.index, "quantile"] = bins.to_numpy()
                for q in range(1, self.quantiles + 1):
                    values = sample.loc[bins == q, "forward_return"]
                    quantile_rows.append(
                        {
                            "timestamp": timestamp,
                            "quantile": q,
                            "count": len(values),
                            "mean_forward_return": float(values.mean()) if len(values) else None,
                        }
                    )
        daily = pd.DataFrame(metric_rows, columns=["timestamp", "n", "ic", "rank_ic"])
        quantile = pd.DataFrame(
            quantile_rows, columns=["timestamp", "quantile", "count", "mean_forward_return"]
        )
        total = len(joined)
        summary: dict[str, float | int | None] = {
            "observations": total,
            "valid_pairs": int(good.sum()),
            "coverage": float(good.mean()) if total else 0.0,
            "evaluated_dates": int(daily["ic"].notna().sum()),
            "mean_ic": float(daily["ic"].mean()) if daily["ic"].notna().any() else None,
            "mean_rank_ic": float(daily["rank_ic"].mean())
            if daily["rank_ic"].notna().any()
            else None,
        }
        return SignalEvaluation(joined, daily, quantile, summary)
