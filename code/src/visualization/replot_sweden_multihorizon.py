"""Regenerate the Sweden multi-horizon newsvendor figure from the CSV.

Loads results/newsvendor_sweden_multihorizon.csv (produced by
notebooks/12_capacity_planning_sweden_multihorizon.ipynb) and renders either:

  all: the original three-panel figure
    Panel A: Sweden CV cost vs horizon at 3:1, per-fold strip + pooled mean.
    Panel B: Calibration at 3:1 vs horizon with 0.75 target tolerance band.
    Panel C: Cost vs ratio at the longest horizon, log y, with SeasonalNaive.

  c: a standalone Panel C figure for thesis use when the ratio-sensitivity
     view is the only one worth keeping in the main text.

The plotting logic mirrors notebook 12; this entry point avoids the
inference-heavy cells when only the figure needs to change.

Usage:
    python src/visualization/replot_sweden_multihorizon.py
    python src/visualization/replot_sweden_multihorizon.py --panel c
    python src/visualization/replot_sweden_multihorizon.py --panel c --output tmp.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
DEFAULT_CSV = RESULTS_DIR / "newsvendor_sweden_multihorizon.csv"
DEFAULT_OUTPUT = FIG_DIR / "newsvendor_sweden_multihorizon.png"
DEFAULT_OUTPUT_PANEL_C = FIG_DIR / "newsvendor_sweden_multihorizon_panel_c.png"

PRIMARY = ["MEx-ColaGNN-SE", "MEx-EpiGNN-SE", "Persistence"]
MODEL_COLORS = {
    "MEx-ColaGNN-SE": "#1f77b4",
    "MEx-EpiGNN-SE":  "#ff7f0e",
    "Persistence":    "#2ca02c",
    "SeasonalNaive":  "#d62728",
}
RATIO_LABELS = ["2:1", "3:1", "5:1", "10:1"]
RATIO_XS = [2, 3, 5, 10]
PLAN_BAND_COLOR = "#fef3c7"
PLAN_BAND_LABEL = "Plan-relevant horizons (h=2-4)"


def draw_panel_c(ax: plt.Axes, pooled_longest: pd.DataFrame, longest: int) -> None:
    all_models = PRIMARY + ["SeasonalNaive"]
    for model in all_models:
        if model not in pooled_longest.index:
            continue
        linestyle = "-" if model in PRIMARY else "-."
        ys = pooled_longest.loc[model, RATIO_LABELS].values
        ax.plot(
            RATIO_XS,
            ys,
            marker="o",
            markersize=7,
            color=MODEL_COLORS[model],
            linewidth=2.0,
            linestyle=linestyle,
            label=model,
        )
    ax.axvline(3, ls="--", color="grey", alpha=0.6, label="Reporting anchor (3:1)")
    ax.set_xlabel(r"Cost ratio $c_u : c_o$")
    ax.set_ylabel(f"Pooled cost (case-weeks) at h={longest}")
    ax.set_title(f"Cost vs ratio at longest horizon (h={longest})")
    ax.set_xticks(RATIO_XS)
    ax.set_xticklabels(RATIO_LABELS)
    ax.set_yscale("log")
    ax.legend(loc="lower right", ncol=2, fontsize=9)
    ax.grid(alpha=0.3, which="both")


def render(df: pd.DataFrame, output_path: Path, panel: str) -> None:
    pooled = df[df.scope == "pooled"]
    folds = df[df.scope == "fold"]
    hs = sorted(df["horizon"].unique())
    longest = max(hs)

    # Pooled rows hold cost summed over all 6 fold-seasons (~4,158 region-weeks).
    # Fold rows hold cost for a single fold-season (~693 region-weeks). The two
    # scales differ by roughly a factor of 6, so Panel A shows per-fold values
    # end-to-end: a mean-across-folds line sitting on top of 6 per-fold points,
    # both on the same axis. Pooled totals remain the authoritative figures
    # cited in Table 4.3 and in prose.
    fold_3to1 = folds[folds.ratio == "3:1"]
    sn_fold_cost_by_h = (fold_3to1[fold_3to1.model == "SeasonalNaive"]
                         .groupby("horizon")["total_cost"].mean())
    cost_3to1 = (fold_3to1.groupby(["model", "horizon"])["total_cost"].mean()
                 .unstack("horizon"))
    svc_3to1 = (pooled[pooled.ratio == "3:1"]
                .pivot_table(index="model", columns="horizon", values="service_level"))
    pooled_longest = (df[(df.scope == "pooled") & (df.horizon == longest)]
                      .pivot_table(index="model", columns="ratio", values="total_cost")
                      [RATIO_LABELS])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if panel == "c":
        fig, ax = plt.subplots(figsize=(10.2, 4.8))
        draw_panel_c(ax, pooled_longest, longest)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 13.5))

    ax = axes[0]
    ax.axvspan(2, 4, color=PLAN_BAND_COLOR, alpha=0.55, zorder=0)
    jitter = {"MEx-ColaGNN-SE": -0.14, "MEx-EpiGNN-SE": 0.0, "Persistence": 0.14}
    for model in PRIMARY:
        colour = MODEL_COLORS[model]
        ax.plot(hs, cost_3to1.loc[model].values,
                marker="o", markersize=7, color=colour, linewidth=2.2, zorder=4, label=model)
        for h in hs:
            fold_vals = fold_3to1[(fold_3to1.model == model) & (fold_3to1.horizon == h)]["total_cost"].values
            ax.scatter([h + jitter[model]] * len(fold_vals), fold_vals,
                       color=colour, alpha=0.35, s=28, zorder=2, edgecolors="none")
    ax.set_xlabel("Horizon $h$ (weeks ahead)")
    ax.set_ylabel("Per-fold newsvendor cost (case-weeks) at 3:1")
    ax.set_title("Panel A: Sweden CV cost vs horizon at 3:1 (6 per-fold points + mean across folds)")
    ax.set_xticks(hs)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor=PLAN_BAND_COLOR, edgecolor="none"))
    labels.append(PLAN_BAND_LABEL)
    ax.legend(handles, labels, loc="upper left")
    ax.grid(alpha=0.3)
    ax.text(0.98, 0.03,
            "SeasonalNaive per-fold mean at 3:1 (not shown):\n  " + "\n  ".join(
                f"h={h}: {int(sn_fold_cost_by_h.loc[h]):,}" for h in hs),
            transform=ax.transAxes, fontsize=8, color="grey",
            va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.85, edgecolor="lightgrey"))

    ax = axes[1]
    ax.axvspan(2, 4, color=PLAN_BAND_COLOR, alpha=0.55, zorder=0)
    ax.axhspan(0.73, 0.77, color="#d1d5db", alpha=0.55, zorder=1,
               label=r"Target CR $\pm$ 0.02 tolerance")
    ax.axhline(0.75, color="k", ls="--", linewidth=1.3, zorder=2,
               label="Target CR (3:1) = 0.75")
    for model in PRIMARY:
        ax.plot(hs, svc_3to1.loc[model].values,
                marker="o", markersize=7, color=MODEL_COLORS[model],
                linewidth=2.2, zorder=3, label=model)
    ax.set_xlabel("Horizon $h$ (weeks ahead)")
    ax.set_ylabel("Pooled service level at 3:1")
    ax.set_title("Panel B: Calibration at 3:1 vs horizon (target 0.75)")
    ax.set_xticks(hs)
    ax.set_ylim(0.60, 0.90)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor=PLAN_BAND_COLOR, edgecolor="none"))
    labels.append(PLAN_BAND_LABEL)
    ax.legend(handles, labels, loc="lower left", ncol=2, fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[2]
    draw_panel_c(ax, pooled_longest, longest)
    ax.set_title(f"Panel C: Cost vs ratio at longest horizon (h={longest}); log $y$ scale")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Source CSV (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (defaults depend on --panel)",
    )
    parser.add_argument(
        "--panel",
        choices=["all", "c"],
        default="all",
        help="Render the original full figure or only Panel C",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    output_path = args.output
    if output_path is None:
        output_path = DEFAULT_OUTPUT if args.panel == "all" else DEFAULT_OUTPUT_PANEL_C
    render(df, output_path, args.panel)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
