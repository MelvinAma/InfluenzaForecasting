"""
Seasonal trend visualizations for US (NHSN) and Swedish (Folkhalsomyndigheten) influenza data.

Produces thesis-quality figures for the data description section:
  1. National aggregate time series (full timeline)
  2. Season overlay (each season aligned week 40 to 20 on same x-axis)
  3. Regional heatmap (regions x season-weeks, color = intensity)
  4. Peak and onset annotations on season overlays

Usage:
    python src/visualization/seasonal_trends.py [--us] [--sweden] [--all]

Output:
    results/figures/us_*.png
    results/figures/sweden_*.png
"""

import colorsys
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "src" / "data"
FIG_DIR = ROOT / "results" / "figures"

FIG_DPI = 300
FIG_FORMAT = "png"

sns.set_theme(style="whitegrid", context="talk", rc={
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 100,
    "axes.titlesize": 20,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "grid.color": "#D1D5DB",
    "grid.linewidth": 0.9,
    "axes.facecolor": "#FCFCFD",
})

LINEWIDTH_MAIN = 2.4
LINEWIDTH_REFERENCE = 1.8
MARKER_SIZE_PEAK = 8
TITLE_PAD = 12
SEASON_LEGEND_FONT = 12.5
SWEDEN_LABEL_MAP = {
    "Blekinge": "Blekinge",
    "Dalarna": "Dalarna",
    "Gotland": "Gotland",
    "Gävleborg": "Gävleb.",
    "Halland": "Halland",
    "Jämtland Härjedalen": "Jämtl.-Härj.",
    "Jönköping": "Jönköp.",
    "Kalmar": "Kalmar",
    "Kronoberg": "Kronob.",
    "Norrbotten": "Norrb.",
    "Skåne": "Skåne",
    "Stockholm": "Stockh.",
    "Södermanland": "Söderm.",
    "Uppsala": "Uppsala",
    "Värmland": "Värml.",
    "Västerbotten": "Västerb.",
    "Västernorrland": "Västern.",
    "Västmanland": "Västmanl.",
    "Västra Götaland": "Västra Göt.",
    "Örebro": "Örebro",
    "Östergötland": "Österg.",
}


def _srgb_to_linear(channel):
    if channel <= 0.04045:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def _rgb_to_lab(rgb):
    rgb = np.asarray(rgb, dtype=float)
    linear = np.array([_srgb_to_linear(c) for c in rgb])
    x = linear[0] * 0.4124564 + linear[1] * 0.3575761 + linear[2] * 0.1804375
    y = linear[0] * 0.2126729 + linear[1] * 0.7151522 + linear[2] * 0.0721750
    z = linear[0] * 0.0193339 + linear[1] * 0.1191920 + linear[2] * 0.9503041

    ref = np.array([0.95047, 1.00000, 1.08883])
    xyz = np.array([x, y, z]) / ref

    def f(t):
        delta = 6 / 29
        if t > delta ** 3:
            return t ** (1 / 3)
        return t / (3 * delta ** 2) + 4 / 29

    fx, fy, fz = [f(v) for v in xyz]
    return np.array([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)])


def _distinct_season_palette(n_colors):
    candidates = []
    for hue in np.linspace(0, 1, 360, endpoint=False):
        for sat in (0.62, 0.78, 0.92):
            for val in (0.56, 0.70, 0.84):
                rgb = colorsys.hsv_to_rgb(float(hue), float(sat), float(val))
                lab = _rgb_to_lab(rgb)
                chroma = float(np.hypot(lab[1], lab[2]))
                lightness = float(lab[0])
                if 32 <= lightness <= 78 and chroma >= 38:
                    candidates.append((rgb, lab))

    anchor_labs = [
        _rgb_to_lab((1.0, 1.0, 1.0)),
        _rgb_to_lab((0.0, 0.0, 0.0)),
        _rgb_to_lab(mcolors.to_rgb("#9CA3AF")),
    ]
    chosen = []

    for _ in range(n_colors):
        distances = []
        for rgb, lab in candidates:
            reference = anchor_labs + [item[1] for item in chosen]
            distances.append(min(np.linalg.norm(lab - ref) for ref in reference))
        best_idx = int(np.argmax(distances))
        chosen.append(candidates.pop(best_idx))

    return [mcolors.to_hex(rgb) for rgb, _ in chosen]


def style_axes(ax, show_x_grid=False):
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(axis="y", alpha=0.85)
    if show_x_grid:
        ax.grid(axis="x", alpha=0.25)
    else:
        ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_matrix(path):
    return np.loadtxt(path, delimiter=",")


def load_us_data():
    matrix = load_matrix(DATA_DIR / "state_flu_admissions.txt")
    dates = pd.read_csv(DATA_DIR / "date_index.csv", header=None,
                        names=["idx", "date"], parse_dates=["date"])["date"]
    states = pd.read_csv(DATA_DIR / "state_index.csv", header=None,
                         names=["idx", "state"])["state"].tolist()

    df = pd.DataFrame(matrix, columns=states)
    df["date"] = dates.values
    df["national"] = matrix.sum(axis=1)

    iso = df["date"].dt.isocalendar()
    df["iso_year"] = iso.year.astype(int)
    df["iso_week"] = iso.week.astype(int)
    return df, states


def load_sweden_data():
    matrix = load_matrix(DATA_DIR / "sweden_flu_cases.txt")
    weeks_df = pd.read_csv(DATA_DIR / "sweden_week_index.csv", header=None,
                           names=["idx", "year", "week"])
    weeks_df["week"] = weeks_df["week"].str.replace("w", "").astype(int)
    regions = pd.read_csv(DATA_DIR / "sweden_region_index.csv", header=None,
                          names=["idx", "region"])["region"].tolist()

    df = pd.DataFrame(matrix, columns=regions)
    df["year"] = weeks_df["year"].values
    df["iso_week"] = weeks_df["week"].values
    df["national"] = matrix.sum(axis=1)

    dates = pd.to_datetime(
        df["year"].astype(str) + df["iso_week"].apply(lambda w: f"-W{w:02d}-1"),
        format="%Y-W%W-%w",
    )
    df["date"] = dates
    return df, regions


# ---------------------------------------------------------------------------
# Season assignment
# ---------------------------------------------------------------------------

def assign_season(df, year_col="iso_year", week_col="iso_week"):
    """Add season_label and week_in_season columns.

    Season: week 40 through week 20.
    week_in_season: 0 = W40, 13 = W53/W1, etc.
    """
    season_start = np.where(df[week_col] >= 40, df[year_col],
                            np.where(df[week_col] <= 20, df[year_col] - 1, np.nan))

    labels = []
    wis = []
    for i, sy in enumerate(season_start):
        if np.isnan(sy):
            labels.append(None)
            wis.append(None)
        else:
            sy = int(sy)
            labels.append(f"{sy}/{str(sy + 1)[-2:]}")
            w = df[week_col].iloc[i]
            wis.append(w - 40 if w >= 40 else (53 - 40) + w)

    df = df.copy()
    df["season"] = labels
    df["week_in_season"] = wis
    return df


# ---------------------------------------------------------------------------
# Onset / peak detection
# ---------------------------------------------------------------------------

def compute_onset_threshold(df):
    """Off-season mean + 2 standard deviations. Off-season = rows with no season assigned."""
    off = df.loc[df["season"].isna(), "national"]
    if len(off) == 0:
        return 0.0
    return float(off.mean() + 2 * off.std())


def find_onset_week(group, threshold):
    """First week_in_season where 3 consecutive weeks exceed threshold."""
    s = group.sort_values("week_in_season")
    vals = s["national"].values
    wks = s["week_in_season"].values
    for i in range(len(vals) - 2):
        if vals[i] > threshold and vals[i + 1] > threshold and vals[i + 2] > threshold:
            return wks[i]
    return None


def find_peak(group):
    """Return (peak_week_in_season, peak_value)."""
    idx = group["national"].idxmax()
    return group.loc[idx, "week_in_season"], group.loc[idx, "national"]


def week_in_season_to_label(wis):
    actual = wis + 40
    if actual > 53:
        actual -= 53
    return f"W{actual}"


# ---------------------------------------------------------------------------
# Plot 1: National aggregate time series
# ---------------------------------------------------------------------------

def plot_national_timeseries(df, title, ylabel, outpath):
    fig, ax = plt.subplots(figsize=(16.0, 5.8))
    color = "#0F766E"
    ax.plot(df["date"], df["national"], color=color, linewidth=3.0)
    ax.fill_between(df["date"], df["national"], alpha=0.18, color=color)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=TITLE_PAD)
    ax.set_xlim(df["date"].iloc[0], df["date"].iloc[-1])
    ax.set_ylim(bottom=0)
    style_axes(ax)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(outpath, dpi=FIG_DPI, format=FIG_FORMAT, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ---------------------------------------------------------------------------
# Plot 2: Season overlay
# ---------------------------------------------------------------------------

def plot_season_overlay(df_season, title, ylabel, outpath, onset_threshold=None):
    fig, ax = plt.subplots(figsize=(15.4, 8.9))
    sorted_seasons = sorted(df_season["season"].dropna().unique())
    season_palette = _distinct_season_palette(len(sorted_seasons))

    for idx, season in enumerate(sorted_seasons):
        color = season_palette[idx]
        sub = df_season[df_season["season"] == season].sort_values("week_in_season")
        linestyle = "--" if season in {"2020/21", "2025/26"} else "-"
        alpha = 0.8 if season in {"2020/21", "2025/26"} else 0.95
        ax.plot(sub["week_in_season"], sub["national"],
                color=color, linewidth=2.8, linestyle=linestyle, label=season, alpha=alpha)
        peak_idx = sub["national"].idxmax()
        ax.plot(sub.loc[peak_idx, "week_in_season"], sub.loc[peak_idx, "national"],
                "v", color=color, markersize=MARKER_SIZE_PEAK, alpha=0.9)

    if onset_threshold is not None:
        ax.axhline(y=onset_threshold, color="gray", linestyle=":",
                    linewidth=LINEWIDTH_REFERENCE, alpha=0.8)
        ax.text(
            0.015, 0.965,
            f"Onset threshold = {onset_threshold:.0f}",
            transform=ax.transAxes,
            fontsize=12.5,
            va="top",
            ha="left",
            color="#4B5563",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D1D5DB", alpha=0.92),
        )

    max_wis = int(df_season["week_in_season"].dropna().max())
    tick_pos = list(range(0, max_wis + 1, 2))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([week_in_season_to_label(w) for w in tick_pos],
                        rotation=35, fontsize=14)
    ax.set_xlabel("Epidemiological week")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=TITLE_PAD)
    ax.set_ylim(bottom=0)
    style_axes(ax, show_x_grid=True)
    ax.legend(
        title="Season",
        title_fontsize=SEASON_LEGEND_FONT,
        fontsize=SEASON_LEGEND_FONT,
        ncol=3 if len(sorted_seasons) <= 9 else 4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
        handlelength=2.8,
        handletextpad=0.6,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(outpath, dpi=FIG_DPI, format=FIG_FORMAT, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ---------------------------------------------------------------------------
# Plot 3: Regional heatmap (seaborn)
# ---------------------------------------------------------------------------

def plot_season_heatmap(
    df_season,
    region_names,
    title_prefix,
    outpath,
    preferred_cols=None,
    display_names=None,
    min_fig_height=None,
    ytick_fontsize=None,
    ytick_pad=10,
    left_margin=None,
    right_margin=None,
):
    sorted_seasons = sorted(df_season["season"].dropna().unique())
    n_seasons = len(sorted_seasons)
    if preferred_cols is not None:
        n_cols = min(preferred_cols, n_seasons)
    else:
        n_cols = 1 if n_seasons <= 3 else (3 if n_seasons > 8 else 2)
    n_rows = (n_seasons + n_cols - 1) // n_cols
    fig_height = 2.35 * n_rows + 0.26 * len(region_names)
    if min_fig_height is not None:
        fig_height = max(fig_height, float(min_fig_height))
    yticklabels = display_names if display_names is not None else region_names

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6.1 * n_cols, fig_height),
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    global_max = df_season[region_names].max().max()

    for idx, season in enumerate(sorted_seasons):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        sub = df_season[df_season["season"] == season].sort_values("week_in_season")
        heatmap_data = sub[region_names].values.T

        week_labels = [week_in_season_to_label(int(w)) for w in sub["week_in_season"]]

        sns.heatmap(
            heatmap_data, ax=ax, cmap="YlOrRd", vmin=0, vmax=global_max,
            cbar=False, xticklabels=False,
            linewidths=0.15, linecolor=(1, 1, 1, 0.25),
            yticklabels=yticklabels if c == 0 else False,
        )
        ax.set_title(season, fontsize=13, pad=8)

        tick_step = max(1, len(week_labels) // 6)
        tick_pos = list(range(0, len(week_labels), tick_step))
        ax.set_xticks([p + 0.5 for p in tick_pos])
        if r == n_rows - 1:
            ax.set_xticklabels([week_labels[p] for p in tick_pos],
                                fontsize=10, rotation=35, color="black")
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0)

        ytick_size = ytick_fontsize if ytick_fontsize is not None else (9 if len(region_names) > 25 else 10)
        if c == 0:
            ax.tick_params(axis="y", labelsize=ytick_size, colors="black", pad=ytick_pad)
            for label in ax.get_yticklabels():
                label.set_fontweight("semibold")
        ax.tick_params(axis="x", labelsize=11, colors="black")

    for idx in range(len(sorted_seasons), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].set_visible(False)

    if left_margin is None:
        left_margin = 0.24 if n_cols <= 2 else 0.19
    if right_margin is None:
        right_margin = 0.88 if n_cols <= 2 else 0.90
    fig.subplots_adjust(left=left_margin, right=right_margin, bottom=0.06, top=0.92, hspace=0.12, wspace=0.12)
    fig.suptitle(f"{title_prefix}: Seasonal Intensity by Region", fontsize=17, y=0.972)

    cbar_x = 0.90 if n_cols <= 2 else 0.915
    cbar_ax = fig.add_axes([cbar_x, 0.15, 0.015, 0.7])
    norm = mcolors.Normalize(vmin=0, vmax=global_max)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="YlOrRd"),
                        cax=cbar_ax, label="Weekly count")
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("Weekly count", size=13)

    fig.savefig(outpath, dpi=FIG_DPI, format=FIG_FORMAT, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ---------------------------------------------------------------------------
# Plot 4: Peak and onset summary
# ---------------------------------------------------------------------------

def plot_peak_onset_summary(df_season, onset_threshold, title, outpath):
    sorted_seasons = sorted(df_season["season"].dropna().unique())
    records = []
    for season in sorted_seasons:
        sub = df_season[df_season["season"] == season]
        pw, pv = find_peak(sub)
        ow = find_onset_week(sub, onset_threshold)
        records.append({"season": season, "peak_val": pv, "peak_wis": pw, "onset_wis": ow})
    summary = pd.DataFrame(records)
    x = np.arange(len(summary))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(13.2, 8.4), sharex=True, layout="constrained",
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08},
    )

    pal = sns.color_palette("muted")
    ax_top.bar(x, summary["peak_val"], color=pal[0], alpha=0.85, width=0.72)
    ax_top.set_ylabel("Peak weekly count\n(national)")
    ax_top.set_title(title, pad=TITLE_PAD)
    style_axes(ax_top)

    valid_onset = summary.dropna(subset=["onset_wis"])
    valid_peak = summary.dropna(subset=["peak_wis"])

    onset_labels = [week_in_season_to_label(int(w)) for w in valid_onset["onset_wis"]]
    peak_labels = [week_in_season_to_label(int(w)) for w in valid_peak["peak_wis"]]

    ax_bot.plot(valid_onset.index, valid_onset["onset_wis"], "s-",
                color=pal[2], markersize=10, linewidth=2.4, label="Onset week")
    ax_bot.plot(valid_peak.index, valid_peak["peak_wis"], "^-",
                color=pal[3], markersize=10, linewidth=2.4, label="Peak week")

    for i, (xi, wi) in enumerate(zip(valid_onset.index, valid_onset["onset_wis"])):
        ax_bot.annotate(onset_labels[i], (xi, wi), textcoords="offset points",
                         xytext=(0, -18), fontsize=12, fontweight="semibold",
                         ha="center", color=pal[2])
    for i, (xi, wi) in enumerate(zip(valid_peak.index, valid_peak["peak_wis"])):
        ax_bot.annotate(peak_labels[i], (xi, wi), textcoords="offset points",
                         xytext=(0, 14), fontsize=12, fontweight="semibold",
                         ha="center", color=pal[3])

    ax_bot.set_ylabel("Week in season\n(0 = W40)")
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(summary["season"], rotation=25, fontsize=14)
    ax_bot.set_xlabel("Season")
    style_axes(ax_bot)
    ax_bot.legend(fontsize=11, loc="upper right", frameon=False)
    fig.savefig(outpath, dpi=FIG_DPI, format=FIG_FORMAT, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def generate_us_figures():
    print("\n=== US Data Visualizations ===")
    df, states = load_us_data()
    df = assign_season(df)
    onset_threshold = compute_onset_threshold(df)
    print(f"  Onset threshold (off-season mean + 2sd): {onset_threshold:.1f}")

    plot_national_timeseries(
        df,
        title="US Weekly Influenza Hospital Admissions (NHSN)",
        ylabel="Total weekly admissions",
        outpath=FIG_DIR / "us_national_timeseries.png",
    )

    df_season = df.dropna(subset=["season"]).copy()
    plot_season_overlay(
        df_season,
        title="US Influenza Admissions: Season Overlay",
        ylabel="Total weekly admissions",
        outpath=FIG_DIR / "us_season_overlay.png",
        onset_threshold=onset_threshold,
    )

    top_states = (df[states].sum().nlargest(15).index.tolist())
    plot_season_heatmap(
        df_season, top_states,
        title_prefix="US Influenza Admissions (Top 15 States)",
        outpath=FIG_DIR / "us_regional_heatmap.png",
        preferred_cols=2,
    )

    plot_peak_onset_summary(
        df_season, onset_threshold,
        title="US Influenza: Peak Intensity and Timing by Season",
        outpath=FIG_DIR / "us_peak_onset_summary.png",
    )
    print("  US figures complete.\n")


def generate_sweden_figures():
    print("\n=== Swedish Data Visualizations ===")
    df, regions = load_sweden_data()
    df = assign_season(df, year_col="year", week_col="iso_week")
    onset_threshold = compute_onset_threshold(df)
    print(f"  Onset threshold (off-season mean + 2sd): {onset_threshold:.1f}")

    plot_national_timeseries(
        df,
        title="Swedish Weekly Lab-Confirmed Influenza Cases",
        ylabel="Total weekly cases",
        outpath=FIG_DIR / "sweden_national_timeseries.png",
    )

    df_season = df.dropna(subset=["season"]).copy()
    plot_season_overlay(
        df_season,
        title="Swedish Influenza Cases: Season Overlay",
        ylabel="Total weekly cases",
        outpath=FIG_DIR / "sweden_season_overlay.png",
        onset_threshold=onset_threshold,
    )

    plot_season_heatmap(
        df_season, regions,
        title_prefix="Swedish Influenza Cases",
        outpath=FIG_DIR / "sweden_regional_heatmap.png",
        preferred_cols=3,
        display_names=[SWEDEN_LABEL_MAP.get(name, name) for name in regions],
        min_fig_height=19.8,
        ytick_fontsize=13,
        ytick_pad=22,
        left_margin=0.31,
        right_margin=0.90,
    )

    plot_peak_onset_summary(
        df_season, onset_threshold,
        title="Swedish Influenza: Peak Intensity and Timing by Season",
        outpath=FIG_DIR / "sweden_peak_onset_summary.png",
    )
    print("  Swedish figures complete.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    args = sys.argv[1:]
    run_us = "--us" in args or "--all" in args or not args
    run_sweden = "--sweden" in args or "--all" in args or not args

    if run_us:
        generate_us_figures()
    if run_sweden:
        generate_sweden_figures()

    print("Done. All figures saved to:", FIG_DIR)


if __name__ == "__main__":
    main()
