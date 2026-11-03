"""Show deterministic first-passage probabilities without historical datasets."""

from __future__ import annotations

from quant_research.model.first_passage import first_passage_probabilities


def main() -> None:
    print("Driftless log-Brownian first-passage examples (NOT validated/calibrated)")
    print(" upper  lower  horizon  sigma/min^0.5    P(upper)  P(lower)  P(neither)")
    for upper, lower, horizon in (
        (0.005, 0.005, 15),
        (0.010, 0.005, 30),
        (0.020, 0.010, 60),
        (0.020, 0.010, 120),
    ):
        volatility = 0.0015
        p = first_passage_probabilities(
            upper_barrier_pct=upper,
            lower_barrier_pct=lower,
            horizon_minutes=horizon,
            volatility_per_sqrt_minute=volatility,
        )
        print(
            f" {upper:>5.1%} {lower:>6.1%} {horizon:>8} {volatility:>14.4%}"
            f" {p.upper_first:>10.4%} {p.lower_first:>9.4%} {p.neither:>11.4%}"
        )


if __name__ == "__main__":
    main()
