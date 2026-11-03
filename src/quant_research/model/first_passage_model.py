"""Adapter connecting mathematical first passage to the public model contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from quant_research.model.barrier import BarrierForecast, BarrierModelInput
from quant_research.model.first_passage import BarrierScenario, first_passage_probabilities
from quant_research.model.volatility import estimate_minute_volatility


@dataclass(frozen=True, slots=True)
class FirstPassageModel:
    """Stateless mathematical model, using historical per-minute volatility."""

    lookback_returns: int = 390
    min_returns: int = 60
    drift_mode: Literal["log_zero", "price_zero"] = "log_zero"

    def predict(self, data: BarrierModelInput) -> BarrierForecast:
        """Evaluate one barrier/horizon combination from the input contract."""
        scenario = BarrierScenario(
            upper_barrier_pct=data.upper_barrier_pct,
            lower_barrier_pct=data.lower_barrier_pct,
            horizon_minutes=data.max_holding_minutes,
        )
        return self.predict_many(data, [scenario])[0]

    def predict_many(
        self,
        data: BarrierModelInput,
        scenarios: Sequence[BarrierScenario],
    ) -> list[BarrierForecast]:
        """Evaluate a grid of scenarios reusing one volatility estimate."""
        if not scenarios:
            return []
        sigma = estimate_minute_volatility(
            data.minute_bars,
            decision_at=data.decision_at,
            lookback_returns=self.lookback_returns,
            min_returns=self.min_returns,
        )
        forecasts: list[BarrierForecast] = []
        for scenario in scenarios:
            probabilities = first_passage_probabilities(
                upper_barrier_pct=scenario.upper_barrier_pct,
                lower_barrier_pct=scenario.lower_barrier_pct,
                horizon_minutes=scenario.horizon_minutes,
                volatility_per_sqrt_minute=sigma,
                drift_mode=self.drift_mode,
            )
            forecasts.append(
                BarrierForecast(
                    security_id=data.security_id,
                    generated_at=data.decision_at,
                    probability_upper_first=probabilities.upper_first,
                    probability_lower_first=probabilities.lower_first,
                    probability_neither=probabilities.neither,
                )
            )
        return forecasts
