"""Regenerate the US newsvendor figures from the CSV.

Loads results/newsvendor_initial.csv (the US NHSN 2024/25 holdout newsvendor
results, produced by notebooks/09_capacity_planning.ipynb) and renders:
  - newsvendor_cost_vs_ratio.png: cost curves across cost ratios for the four
    NATIONAL-aggregation models.
  - newsvendor_breakdown_3to1.png: stacked under/over-capacity cost breakdown
    at the 3:1 literature anchor.

Usage:
    python src/visualization/replot_us_newsvendor.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
DEFAULT_CSV = RESULTS_DIR / "newsvendor_initial.csv"
DEFAULT_OUTPUT_COST = FIG_DIR / "newsvendor_cost_vs_ratio.png"
DEFAULT_OUTPUT_BREAKDOWN = FIG_DIR / "newsvendor_breakdown_3to1.png"

RATIO_LABELS = ["2:1", "3:1", "5:1", "10:1"]
RATIO_XS = [2, 3, 5, 10]
MODEL_ORDER = ["FluSight-baseline", "FluSight-ensemble", "MEx-ColaGNN", "MEx-EpiGNN"]


def render_cost_vs_ratio(df: pd.DataFrame, output_path: Path) -> None:
    nat = df[df["region"] == "NATIONAL"].copy()
    pivot = nat.pivot_table(index="model", columns="ratio", values="total_cost")[RATIO_LABELS]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in MODEL_ORDER:
        if model not in pivot.index:
            continue
        ax.plot(RATIO_XS, pivot.loc[model, RATIO_LABELS].values,
                marker="o", label=model)
    ax.set_xlabel(r"Cost ratio  $c_u : c_o$")
    ax.set_ylabel("Total newsvendor cost (bed-weeks)")
    ax.set_title("US NHSN 2024/25 - Capacity-planning cost vs cost ratio")
    ax.set_xticks(RATIO_XS)
    ax.set_xticklabels(RATIO_LABELS)
    ax.axvline(3, ls="--", color="grey", alpha=0.5)
    ax.text(3.05, ax.get_ylim()[1] * 0.95,
            "literature anchor (87% reference value)",
            color="grey", fontsize=8)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_breakdown_3to1(df: pd.DataFrame, output_path: Path) -> None:
    nat = df[df["region"] == "NATIONAL"].copy()
    anchor = nat[nat["ratio"] == "3:1"].set_index("model")
    order = anchor.sort_values("total_cost").index.tolist()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(order, anchor.loc[order, "under_cost"], label="under-capacity cost")
    ax.bar(order, anchor.loc[order, "over_cost"],
           bottom=anchor.loc[order, "under_cost"], label="over-capacity cost")
    ax.set_ylabel("Cost (bed-weeks)")
    ax.set_title(r"US NHSN 2024/25 - Cost breakdown at literature anchor $c_u : c_o = 3 : 1$")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    for tick in ax.get_xticklabels():
        tick.set_rotation(15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--cost-output", type=Path, default=DEFAULT_OUTPUT_COST)
    parser.add_argument("--breakdown-output", type=Path, default=DEFAULT_OUTPUT_BREAKDOWN)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    render_cost_vs_ratio(df, args.cost_output)
    print(f"Saved cost-vs-ratio figure to {args.cost_output}")
    render_breakdown_3to1(df, args.breakdown_output)
    print(f"Saved breakdown figure to {args.breakdown_output}")


if __name__ == "__main__":
    main()
