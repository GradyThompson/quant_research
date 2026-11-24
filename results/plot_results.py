"""Regenerate portfolio figures from published aggregate CSVs.

Requires matplotlib; no vendor data or credentials.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def read_rows(name: str) -> list[dict[str, str]]:
    with (HERE / name).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def main() -> None:
    cost = read_rows("cost_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(8.4, 4.8), layout="constrained")
    for label in dict.fromkeys(row["label"] for row in cost):
        rows = [row for row in cost if row["label"] == label]
        ax.plot(
            [float(row["additional_cost_bps"]) for row in rows],
            [float(row["net_bps_per_session"]) for row in rows],
            marker="o",
            label=label,
        )
    ax.axhline(0, color="0.4", linestyle="--", linewidth=1)
    ax.set(
        title="2024 walk-forward: transaction-cost sensitivity",
        xlabel="Additional cost (bps), beyond estimated commissions",
        ylabel="Net price-proxy bps per session",
        xticks=[4, 8, 12],
    )
    ax.legend(ncol=2, fontsize=8, loc="lower left")
    ax.grid(alpha=0.2)
    fig.savefig(HERE / "cost_sensitivity.png", dpi=180)
    plt.close(fig)

    calibration = read_rows("upper_touch_calibration.csv")
    fig, ax = plt.subplots(figsize=(6.8, 5.6), layout="constrained")
    ax.plot([0, 1], [0, 1], "--", color="0.5", label="Ideal calibration")
    for ticker in dict.fromkeys(row["ticker"] for row in calibration):
        rows = [row for row in calibration if row["ticker"] == ticker]
        ax.scatter(
            [float(row["mean_predicted"]) for row in rows],
            [float(row["observed_fraction"]) for row in rows],
            s=[12 + 0.3 * int(row["count"]) for row in rows],
            alpha=0.75,
            label=ticker,
        )
    ax.set(
        title="Upper-touch calibration: 10 development sessions",
        xlabel="Mean predicted upper-touch probability",
        ylabel="Observed upper-touch frequency",
        xlim=(0, 1),
        ylim=(0, 1),
        aspect="equal",
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)
    fig.savefig(HERE / "upper_touch_calibration.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
