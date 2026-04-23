"""Plot seed-level held-out error dispersion for base and tuned configurations.

This figure is intended for the thesis discussion of tuning through a
bias-variance lens. It reads the authoritative JSON result artifacts on disk
and visualizes seed-level error dispersion for the two datasets where per-seed
artifacts are available in a consistent format: US NHSN and wILI.

Usage:
    python src/visualization/bias_variance_seed_dispersion.py

Output:
    results/figures/bias_variance_seed_dispersion.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
DEFAULT_OUTPUT = FIG_DIR / "bias_variance_seed_dispersion.png"

COLORS = {
    "Base": "#9CA3AF",
    "Tuned": "#2563EB",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def category(label: str, variant: str, values: list[float], mean: float, std: float) -> dict:
    return {
        "label": label,
        "variant": variant,
        "values": np.asarray(values, dtype=float),
        "mean": float(mean),
        "std": float(std),
    }


def sample_std(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def sample_mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def build_panels() -> list[dict]:
    us_base = load_json(RESULTS_DIR / "us_base_eval.json")
    us_colagnn_tuned = load_json(RESULTS_DIR / "colagnn_us_ensemble_metrics.json")
    us_epignn_tuned = load_json(RESULTS_DIR / "epignn_us_ensemble_metrics.json")
    wili_base = load_json(RESULTS_DIR / "wili_base_eval.json")
    wili_tuned = load_json(RESULTS_DIR / "wili_tuned_eval.json")

    us_mae = [
        category(
            "ColaGNN\nBase",
            "Base",
            [row["mae_mean"] for row in us_base["colagnn"]["per_seed"]],
            sample_mean([row["mae_mean"] for row in us_base["colagnn"]["per_seed"]]),
            sample_std([row["mae_mean"] for row in us_base["colagnn"]["per_seed"]]),
        ),
        category(
            "ColaGNN\nTuned",
            "Tuned",
            us_colagnn_tuned["per_seed_mae"],
            sample_mean(us_colagnn_tuned["per_seed_mae"]),
            sample_std(us_colagnn_tuned["per_seed_mae"]),
        ),
        category(
            "EpiGNN\nBase",
            "Base",
            [row["mae_mean"] for row in us_base["epignn"]["per_seed"]],
            sample_mean([row["mae_mean"] for row in us_base["epignn"]["per_seed"]]),
            sample_std([row["mae_mean"] for row in us_base["epignn"]["per_seed"]]),
        ),
        category(
            "EpiGNN\nTuned",
            "Tuned",
            us_epignn_tuned["per_seed_mae"],
            sample_mean(us_epignn_tuned["per_seed_mae"]),
            sample_std(us_epignn_tuned["per_seed_mae"]),
        ),
    ]

    us_rmse = [
        category(
            "ColaGNN\nBase",
            "Base",
            [row["rmse_mean"] for row in us_base["colagnn"]["per_seed"]],
            sample_mean([row["rmse_mean"] for row in us_base["colagnn"]["per_seed"]]),
            sample_std([row["rmse_mean"] for row in us_base["colagnn"]["per_seed"]]),
        ),
        category(
            "ColaGNN\nTuned",
            "Tuned",
            us_colagnn_tuned["per_seed_rmse"],
            sample_mean(us_colagnn_tuned["per_seed_rmse"]),
            sample_std(us_colagnn_tuned["per_seed_rmse"]),
        ),
        category(
            "EpiGNN\nBase",
            "Base",
            [row["rmse_mean"] for row in us_base["epignn"]["per_seed"]],
            sample_mean([row["rmse_mean"] for row in us_base["epignn"]["per_seed"]]),
            sample_std([row["rmse_mean"] for row in us_base["epignn"]["per_seed"]]),
        ),
        category(
            "EpiGNN\nTuned",
            "Tuned",
            us_epignn_tuned["per_seed_rmse"],
            sample_mean(us_epignn_tuned["per_seed_rmse"]),
            sample_std(us_epignn_tuned["per_seed_rmse"]),
        ),
    ]

    wili_rmse_global = [
        category(
            "ColaGNN\nBase",
            "Base",
            [row["rmse_global"] for row in wili_base["colagnn"]["per_seed"]],
            sample_mean([row["rmse_global"] for row in wili_base["colagnn"]["per_seed"]]),
            sample_std([row["rmse_global"] for row in wili_base["colagnn"]["per_seed"]]),
        ),
        category(
            "ColaGNN\nTuned",
            "Tuned",
            [row["rmse_global"] for row in wili_tuned["colagnn"]["per_seed"]],
            sample_mean([row["rmse_global"] for row in wili_tuned["colagnn"]["per_seed"]]),
            sample_std([row["rmse_global"] for row in wili_tuned["colagnn"]["per_seed"]]),
        ),
        category(
            "EpiGNN\nBase",
            "Base",
            [row["rmse_global"] for row in wili_base["epignn"]["per_seed"]],
            sample_mean([row["rmse_global"] for row in wili_base["epignn"]["per_seed"]]),
            sample_std([row["rmse_global"] for row in wili_base["epignn"]["per_seed"]]),
        ),
        category(
            "EpiGNN\nTuned",
            "Tuned",
            [row["rmse_global"] for row in wili_tuned["epignn"]["per_seed"]],
            sample_mean([row["rmse_global"] for row in wili_tuned["epignn"]["per_seed"]]),
            sample_std([row["rmse_global"] for row in wili_tuned["epignn"]["per_seed"]]),
        ),
    ]

    return [
        {
            "title": "US NHSN: single-seed MAE",
            "ylabel": "MAE",
            "categories": us_mae,
        },
        {
            "title": "US NHSN: single-seed RMSE",
            "ylabel": "RMSE",
            "categories": us_rmse,
        },
        {
            "title": "wILI: global RMSE",
            "ylabel": "RMSE_gl",
            "categories": wili_rmse_global,
        },
    ]


def plot_panel(ax: plt.Axes, panel: dict, rng: np.random.Generator) -> None:
    categories = panel["categories"]
    x_positions = np.array([0.0, 0.75, 2.0, 2.75])

    for x, item in zip(x_positions, categories):
        color = COLORS[item["variant"]]
        jitter = rng.uniform(-0.08, 0.08, size=len(item["values"]))
        ax.scatter(
            np.full(len(item["values"]), x) + jitter,
            item["values"],
            color=color,
            alpha=0.62,
            s=32,
            edgecolors="none",
            zorder=2,
        )
        ax.errorbar(
            x,
            item["mean"],
            yerr=item["std"],
            fmt="o",
            color="black",
            ecolor="black",
            elinewidth=1.6,
            capsize=4,
            markersize=4.8,
            zorder=3,
        )

    ymin, ymax = ax.get_ylim()
    pad = 0.04 * (ymax - ymin) if ymax > ymin else 1.0
    for x, item in zip(x_positions, categories):
        ax.text(
            x,
            item["mean"] + item["std"] + pad,
            f"sd = {item['std']:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    ax.set_xticks(x_positions)
    ax.set_xlim(-0.25, 3.0)
    ax.set_xticklabels([item["label"] for item in categories], fontsize=10)
    ax.set_ylabel(panel["ylabel"])
    ax.set_title(panel["title"], fontsize=11.5, pad=8)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.axvline(1.375, color="#E5E7EB", linewidth=1.0, zorder=1)


def make_figure(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    panels = build_panels()

    fig, axes = plt.subplots(3, 1, figsize=(6.8, 10.4))
    for ax, panel in zip(axes, panels):
        plot_panel(ax, panel, rng)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="Base", markerfacecolor=COLORS["Base"], markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Tuned", markerfacecolor=COLORS["Tuned"], markersize=8),
        Line2D([0], [0], marker="o", color="black", label="Mean +/- sd", markerfacecolor="black", markersize=5),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.988))
    fig.suptitle("Seed-level error dispersion before and after tuning", fontsize=13, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.4)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output image path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_figure(args.output)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
