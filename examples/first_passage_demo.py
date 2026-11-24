"""Print analytical barrier probabilities without market data or credentials."""

from quant_research.model.first_passage import first_passage_probabilities


def main() -> None:
    print("| Horizon (min) | Upper first | Lower first | Neither |")
    print("| ---: | ---: | ---: | ---: |")
    for horizon in (5, 15, 30, 60):
        result = first_passage_probabilities(
            upper_barrier_pct=0.005,
            lower_barrier_pct=0.005,
            horizon_minutes=horizon,
            volatility_per_sqrt_minute=0.001,
            drift_mode="price_zero",
        )
        print(
            f"| {horizon} | {result.upper_first:.6f} | "
            f"{result.lower_first:.6f} | {result.neither:.6f} |"
        )


if __name__ == "__main__":
    main()
