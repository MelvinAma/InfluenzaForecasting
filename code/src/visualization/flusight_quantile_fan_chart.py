"""Generate a shared FluSight-style quantile forecast comparison figure.

By default the script compares MEx-ColaGNN and MEx-EpiGNN on the same
state/reference-date pair. The automatic selector avoids the earlier
"largest surge" rule and instead picks a representative high-activity example:

1. restrict to common complete submissions across all requested models;
2. keep only pairs with realized 3-horizon burden at or above the median;
3. choose the pair whose combined 3-horizon mean WIS is closest to the median.

Usage:
    python src/visualization/flusight_quantile_fan_chart.py
    python src/visualization/flusight_quantile_fan_chart.py --state CA --reference-date 2025-01-11
    python src/visualization/flusight_quantile_fan_chart.py --model-dirs results/vintage_submissions/MEx-ColaGNN

Output:
    results/figures/flusight_quantile_fan_chart.png
    results/figures/flusight_quantile_fan_chart_metadata.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
DEFAULT_OUTPUT = FIG_DIR / "flusight_quantile_fan_chart.png"
DEFAULT_MODEL_DIRS = [
    RESULTS_DIR / "vintage_submissions" / "MEx-ColaGNN",
    RESULTS_DIR / "vintage_submissions" / "MEx-EpiGNN",
]
DEFAULT_OVERLAY_DIR = ROOT / "data" / "flusight_2024_25" / "model-output" / "FluSight-ensemble"
DEFAULT_TARGET_PATH = ROOT / "data" / "flusight_2024_25" / "target-hospital-admissions.csv"
DEFAULT_HISTORY_WEEKS = 6

sys.path.insert(0, str(ROOT))
from src.eval.quantile import FLUSIGHT_LEVELS, compute_wis  # noqa: E402


PLOT_LEVELS = [0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975]
FIPS_TO_ABBR = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY",
}
ABBR_TO_FIPS = {abbr: fips for fips, abbr in FIPS_TO_ABBR.items()}
INTERVAL_COLORS = {
    "95": "#BFDBFE",
    "80": "#93C5FD",
    "50": "#60A5FA",
    "median": "#1D4ED8",
}
OVERLAY_COLORS = {
    "50": "#FCA5A5",
    "median": "#DC2626",
}


def load_truth(path: Path) -> pd.DataFrame:
    truth = pd.read_csv(path, parse_dates=["date"], dtype={"location": str})
    truth["location"] = truth["location"].astype(str).str.zfill(2)
    truth = truth[truth["location"].ne("US")].copy()
    return truth


def load_submissions(model_dir: Path) -> pd.DataFrame:
    frames = []
    for csv_path in sorted(model_dir.glob("*.csv")):
        frame = pd.read_csv(
            csv_path,
            parse_dates=["reference_date", "target_end_date"],
            dtype={"location": str, "output_type_id": str},
        )
        frame.columns = frame.columns.str.lower()
        frame["location"] = frame["location"].astype(str).str.zfill(2)
        frame = frame[
            frame["output_type"].str.lower().eq("quantile")
            & frame["horizon"].isin([1, 2, 3])
            & frame["location"].ne("US")
        ].copy()
        frame["quantile"] = frame["output_type_id"].astype(float)
        frames.append(
            frame[["reference_date", "target_end_date", "horizon", "location", "quantile", "value"]]
        )

    if not frames:
        raise FileNotFoundError(f"No submission CSVs found in {model_dir}")

    return pd.concat(frames, ignore_index=True)


def resolve_location(location_arg: str | None, truth: pd.DataFrame) -> str | None:
    if location_arg is None:
        return None

    value = location_arg.strip()
    upper = value.upper()
    if upper in ABBR_TO_FIPS:
        return ABBR_TO_FIPS[upper]
    if value.isdigit():
        padded = value.zfill(2)
        if padded in FIPS_TO_ABBR:
            return padded

    matches = truth.loc[
        truth["location_name"].str.lower().eq(value.lower()),
        "location",
    ].drop_duplicates()
    if len(matches) == 1:
        return matches.iloc[0]

    raise ValueError(f"Could not resolve state/location argument: {location_arg}")


def complete_pairs_for_model(submissions: pd.DataFrame) -> set[tuple[pd.Timestamp, str]]:
    complete = (
        submissions.groupby(["reference_date", "location", "horizon"])["quantile"]
        .nunique()
        .reset_index(name="n_quantiles")
    )
    complete = complete[complete["n_quantiles"].eq(len(FLUSIGHT_LEVELS))]
    complete = (
        complete.groupby(["reference_date", "location"])["horizon"]
        .nunique()
        .reset_index(name="n_horizons")
    )
    complete = complete[complete["n_horizons"].eq(3)]
    return set(zip(complete["reference_date"], complete["location"]))


def build_forecast_frame(submissions: pd.DataFrame, location: str, reference_date: pd.Timestamp) -> pd.DataFrame:
    subset = submissions[
        submissions["location"].eq(location)
        & submissions["reference_date"].eq(reference_date)
    ].copy()
    if subset.empty:
        raise ValueError("No submission rows found for the selected example.")

    forecast = (
        subset.pivot_table(
            index="target_end_date",
            columns="quantile",
            values="value",
            aggfunc="first",
        )
        .sort_index()
    )
    missing = [level for level in FLUSIGHT_LEVELS if level not in forecast.columns]
    if missing:
        raise ValueError(f"Missing quantile levels for selected example: {missing}")
    return forecast


def truth_for_targets(truth: pd.DataFrame, location: str, target_dates: pd.Index) -> pd.Series:
    rows = truth[truth["location"].eq(location)][["date", "value"]].copy()
    rows = rows.set_index("date").sort_index()
    values = rows.reindex(target_dates)["value"]
    if values.isna().any():
        raise ValueError("Ground truth is missing for one or more selected target dates.")
    return values


def wis_for_frame(forecast: pd.DataFrame, true_series: pd.Series) -> float:
    qpred = forecast[sorted(FLUSIGHT_LEVELS)].to_numpy(dtype=float)[:, None, :]
    truth_vals = true_series.to_numpy(dtype=float)[:, None]
    return float(compute_wis(qpred, truth_vals, FLUSIGHT_LEVELS)["wis_mean"])


def choose_quad_examples(
    submissions_by_model: dict[str, pd.DataFrame],
    truth: pd.DataFrame,
    overlay_submissions: pd.DataFrame | None = None,
) -> list[dict]:
    """Pick four representative examples spanning (early|late) x (easier|harder).

    Filter to above-median realised burden, split by reference-date median into
    early vs late, and within each half pick the submission closest to the 25th
    and 75th percentile of combined WIS across the configured models. The
    resulting 2x2 is representative, not cherry-picked: easier/harder sit at
    typical-low and typical-high difficulty rather than at pathological extremes.
    """
    common_pairs: set[tuple[pd.Timestamp, str]] | None = None
    for submissions in submissions_by_model.values():
        pairs = complete_pairs_for_model(submissions)
        common_pairs = pairs if common_pairs is None else common_pairs & pairs
    if overlay_submissions is not None:
        common_pairs = common_pairs & complete_pairs_for_model(overlay_submissions) if common_pairs is not None else complete_pairs_for_model(overlay_submissions)

    if not common_pairs:
        raise ValueError("No common complete 1-to-3-week submissions found across the selected models.")

    scored = []
    for ref_date, loc in sorted(common_pairs):
        model_wis = {}
        target_dates = None
        for model_name, submissions in submissions_by_model.items():
            forecast = build_forecast_frame(submissions, loc, ref_date)
            if target_dates is None:
                target_dates = forecast.index
            true_series = truth_for_targets(truth, loc, forecast.index)
            model_wis[model_name] = wis_for_frame(forecast, true_series)

        overlay_wis: float | None = None
        if overlay_submissions is not None:
            overlay_forecast = build_forecast_frame(overlay_submissions, loc, ref_date)
            overlay_truth = truth_for_targets(truth, loc, overlay_forecast.index)
            overlay_wis = wis_for_frame(overlay_forecast, overlay_truth)

        assert target_dates is not None
        true_series = truth_for_targets(truth, loc, target_dates)
        realized_total = float(true_series.sum())
        combined_wis = float(sum(model_wis.values()) / len(model_wis))
        scored.append(
            {
                "reference_date": ref_date,
                "location": loc,
                "realized_total": realized_total,
                "combined_wis": combined_wis,
                "model_wis": model_wis,
                "overlay_wis": overlay_wis,
            }
        )

    scored_df = pd.DataFrame(scored)
    burden_threshold = float(scored_df["realized_total"].median())
    high_burden = scored_df[scored_df["realized_total"] >= burden_threshold].copy()

    date_median = high_burden["reference_date"].median()
    early_df = high_burden[high_burden["reference_date"] < date_median].copy()
    late_df = high_burden[high_burden["reference_date"] >= date_median].copy()

    def _pick(sub_df: pd.DataFrame, quantile: float, bucket: str, taken_states: set[str]) -> dict | None:
        """Pick the row whose combined_wis is closest to the target quantile,
        skipping states already chosen so the four quadrants land on four
        distinct states. If every candidate state is already taken (rare on the
        FluSight 51-jurisdiction set), fall back to the global closest match."""
        if sub_df.empty:
            return None
        target = sub_df["combined_wis"].quantile(quantile)
        candidates = sub_df.copy()
        candidates["distance"] = (candidates["combined_wis"] - target).abs()
        candidates = candidates.sort_values(["distance", "reference_date", "location"])
        for _, row in candidates.iterrows():
            if row["location"] not in taken_states:
                chosen = row.to_dict()
                chosen["bucket"] = bucket
                taken_states.add(row["location"])
                return chosen
        chosen = candidates.iloc[0].to_dict()
        chosen["bucket"] = bucket
        return chosen

    taken: set[str] = set()
    quadrants = [
        _pick(early_df, 0.25, "Early season, easier", taken),
        _pick(early_df, 0.75, "Early season, harder", taken),
        _pick(late_df, 0.25, "Late season, easier", taken),
        _pick(late_df, 0.75, "Late season, harder", taken),
    ]
    examples = [ex for ex in quadrants if ex is not None]
    if len(examples) < 4:
        raise ValueError(
            f"Could not select four distinct quadrant examples (got {len(examples)}); "
            "check that above-median-burden submissions are present in both halves of the season."
        )
    return examples


def choose_example(
    submissions_by_model: dict[str, pd.DataFrame],
    truth: pd.DataFrame,
    location: str | None,
    reference_date: pd.Timestamp | None,
    overlay_submissions: pd.DataFrame | None = None,
) -> tuple[str, pd.Timestamp, str, dict[str, float], float]:
    common_pairs: set[tuple[pd.Timestamp, str]] | None = None
    for submissions in submissions_by_model.values():
        pairs = complete_pairs_for_model(submissions)
        common_pairs = pairs if common_pairs is None else common_pairs & pairs
    if overlay_submissions is not None:
        common_pairs = common_pairs & complete_pairs_for_model(overlay_submissions) if common_pairs is not None else complete_pairs_for_model(overlay_submissions)

    if not common_pairs:
        raise ValueError("No common complete 1-to-3-week submissions found across the selected models.")

    if location is not None:
        common_pairs = {pair for pair in common_pairs if pair[1] == location}
    if reference_date is not None:
        common_pairs = {pair for pair in common_pairs if pair[0] == reference_date}
    if not common_pairs:
        raise ValueError("No common complete submissions remain after applying the requested state/date filters.")

    scored = []
    for ref_date, loc in sorted(common_pairs):
        model_wis = {}
        target_dates = None
        for model_name, submissions in submissions_by_model.items():
            forecast = build_forecast_frame(submissions, loc, ref_date)
            if target_dates is None:
                target_dates = forecast.index
            true_series = truth_for_targets(truth, loc, forecast.index)
            model_wis[model_name] = wis_for_frame(forecast, true_series)

        assert target_dates is not None
        true_series = truth_for_targets(truth, loc, target_dates)
        realized_total = float(true_series.sum())
        combined_wis = float(sum(model_wis.values()) / len(model_wis))
        scored.append(
            {
                "reference_date": ref_date,
                "location": loc,
                "realized_total": realized_total,
                "combined_wis": combined_wis,
                "model_wis": model_wis,
            }
        )

    if location is not None and reference_date is not None:
        best = scored[0]
        method = "user-specified"
    else:
        scored_df = pd.DataFrame(scored)
        burden_threshold = float(scored_df["realized_total"].median())
        high_burden = scored_df[scored_df["realized_total"] >= burden_threshold].copy()
        target_wis = float(high_burden["combined_wis"].median())
        high_burden["distance"] = (high_burden["combined_wis"] - target_wis).abs()
        high_burden = high_burden.sort_values(
            ["distance", "reference_date", "location"],
            ascending=[True, True, True],
        )
        best = high_burden.iloc[0].to_dict()
        method = "above-median burden, median combined WIS"

    return (
        best["location"],
        pd.Timestamp(best["reference_date"]),
        method,
        dict(best["model_wis"]),
        float(best["realized_total"]),
    )


EPIGNN_MEDIAN_COLOR = "#EA580C"  # orange, distinct from blue ColaGNN + red ensemble


def plot_quad_comparison(
    examples: list[dict],
    submissions_by_model: dict[str, pd.DataFrame],
    truth: pd.DataFrame,
    history_weeks: int,
    output_path: Path,
    primary_model: str,
    overlay_name: str | None = None,
    overlay_submissions: pd.DataFrame | None = None,
) -> dict:
    """Render a 2x2 grid of four representative examples.

    Each panel shows the primary model (default MEx-ColaGNN) as a full blue
    probabilistic fan chart, the secondary MEx model as an orange dashed
    median line, and the FluSight-ensemble overlay as dashed red 50%
    boundary lines plus a dashed red median line. The four panels span the
    (early|late) x (easier|harder) quadrant classification produced by
    choose_quad_examples.
    """
    if len(examples) != 4:
        raise ValueError(f"plot_quad_comparison requires exactly 4 examples, got {len(examples)}")

    secondary_models = [m for m in submissions_by_model.keys() if m != primary_model]
    secondary_model = secondary_models[0] if secondary_models else None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    axes_flat = axes.flatten()

    legend_handles: list | None = None
    legend_labels: list[str] | None = None
    metadata_examples: list[dict] = []

    for ax, example in zip(axes_flat, examples):
        location = example["location"]
        reference_date = pd.Timestamp(example["reference_date"])
        location_rows = truth[truth["location"].eq(location)]
        location_name = location_rows["location_name"].dropna().iloc[0]

        primary_forecast = build_forecast_frame(submissions_by_model[primary_model], location, reference_date)
        target_dates = primary_forecast.index
        history_start = reference_date - pd.Timedelta(weeks=history_weeks - 1)
        plot_end = max(target_dates)

        observed = location_rows[
            location_rows["date"].between(history_start, plot_end)
        ].sort_values("date")

        max_y = float(observed["value"].max()) if not observed.empty else 0.0
        max_y = max(max_y, float(primary_forecast[0.975].max()))

        secondary_forecast = None
        if secondary_model is not None:
            secondary_forecast = build_forecast_frame(
                submissions_by_model[secondary_model], location, reference_date
            )
            max_y = max(max_y, float(secondary_forecast[0.50].max()))

        overlay_forecast = None
        if overlay_submissions is not None:
            overlay_forecast = build_forecast_frame(overlay_submissions, location, reference_date)
            max_y = max(max_y, float(overlay_forecast[0.975].max()))

        ax.axvspan(reference_date, plot_end, color="#DBEAFE", alpha=0.25, zorder=0)
        line_obs, = ax.plot(
            observed["date"], observed["value"],
            color="black", linewidth=1.6, marker="o", markersize=4,
            label="Observed admissions", zorder=4,
        )
        band95 = ax.fill_between(
            primary_forecast.index, primary_forecast[0.025], primary_forecast[0.975],
            color=INTERVAL_COLORS["95"], alpha=0.28,
            label=f"{primary_model} 95% interval", zorder=1,
        )
        band80 = ax.fill_between(
            primary_forecast.index, primary_forecast[0.10], primary_forecast[0.90],
            color=INTERVAL_COLORS["80"], alpha=0.28,
            label=f"{primary_model} 80% interval", zorder=2,
        )
        band50 = ax.fill_between(
            primary_forecast.index, primary_forecast[0.25], primary_forecast[0.75],
            color=INTERVAL_COLORS["50"], alpha=0.25,
            label=f"{primary_model} 50% interval", zorder=3,
        )
        line_med, = ax.plot(
            primary_forecast.index, primary_forecast[0.50],
            color=INTERVAL_COLORS["median"], linewidth=2.0,
            marker="o", markersize=5, label=f"{primary_model} median", zorder=5,
        )
        line_sec = None
        if secondary_forecast is not None:
            line_sec, = ax.plot(
                secondary_forecast.index, secondary_forecast[0.50],
                color=EPIGNN_MEDIAN_COLOR, linewidth=1.6, linestyle=":",
                marker="^", markersize=5,
                label=f"{secondary_model} median", zorder=5.1,
            )

        overlay_lower = None
        overlay_med = None
        if overlay_forecast is not None:
            overlay_lower, = ax.plot(
                overlay_forecast.index, overlay_forecast[0.25],
                color=OVERLAY_COLORS["median"], linewidth=1.3, linestyle="--",
                marker="s", markersize=3.5, markerfacecolor="white",
                markeredgecolor=OVERLAY_COLORS["median"], markeredgewidth=1.0,
                label=f"{overlay_name} 50% interval", zorder=5.0,
            )
            ax.plot(
                overlay_forecast.index, overlay_forecast[0.75],
                color=OVERLAY_COLORS["median"], linewidth=1.3, linestyle="--",
                marker="s", markersize=3.5, markerfacecolor="white",
                markeredgecolor=OVERLAY_COLORS["median"], markeredgewidth=1.0,
                zorder=5.0,
            )
            overlay_med, = ax.plot(
                overlay_forecast.index, overlay_forecast[0.50],
                color=OVERLAY_COLORS["median"], linewidth=1.6, linestyle="--",
                marker="s", markersize=4, label=f"{overlay_name} median",
                zorder=5.2,
            )

        ax.axvline(reference_date, color="#374151", linestyle="--", linewidth=1.0)
        bucket = example.get("bucket", "")
        mex_wis = example["model_wis"].get(primary_model)
        sec_wis = example["model_wis"].get(secondary_model) if secondary_model else None
        ov_wis = example.get("overlay_wis")
        wis_chunks: list[str] = []
        if mex_wis is not None:
            wis_chunks.append(f"{primary_model} {mex_wis:.1f}")
        if sec_wis is not None:
            wis_chunks.append(f"{secondary_model} {sec_wis:.1f}")
        if ov_wis is not None and overlay_name is not None:
            wis_chunks.append(f"{overlay_name} {ov_wis:.1f}")
        title_top = f"{bucket}: {location_name} ({FIPS_TO_ABBR[location]}), ref {reference_date.strftime('%Y-%m-%d')}"
        title_bottom = "3-horizon mean WIS: " + "  |  ".join(wis_chunks) if wis_chunks else ""
        ax.set_title(title_top + ("\n" + title_bottom if title_bottom else ""), fontsize=10)
        ax.set_xlim(history_start - pd.Timedelta(days=2), plot_end + pd.Timedelta(days=2))
        ax.set_ylim(0, max_y * 1.12 if max_y > 0 else 1.0)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.SA, interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        for tick in ax.xaxis.get_majorticklabels():
            tick.set_rotation(30)
            tick.set_horizontalalignment("right")

        if legend_handles is None:
            legend_handles = [line_obs, band95, band80, band50, line_med]
            legend_labels = [
                "Observed admissions",
                f"{primary_model} 95% interval",
                f"{primary_model} 80% interval",
                f"{primary_model} 50% interval",
                f"{primary_model} median",
            ]
            if line_sec is not None:
                legend_handles.append(line_sec)
                legend_labels.append(f"{secondary_model} median")
            if overlay_lower is not None and overlay_med is not None:
                legend_handles.extend([overlay_lower, overlay_med])
                legend_labels.extend([f"{overlay_name} 50% interval", f"{overlay_name} median"])

        metadata_examples.append({
            "bucket": bucket,
            "location_fips": location,
            "location_abbr": FIPS_TO_ABBR[location],
            "location_name": location_name,
            "reference_date": reference_date.strftime("%Y-%m-%d"),
            "target_end_dates": [ts.strftime("%Y-%m-%d") for ts in target_dates],
            "model_wis": example["model_wis"],
            "overlay_wis": ov_wis,
            "realized_total": example["realized_total"],
        })

    for row_axes in axes:
        row_axes[0].set_ylabel("Weekly hospital admissions")
    for col_ax in axes[-1, :]:
        col_ax.set_xlabel("Week ending date")

    fig.suptitle(
        "FluSight 2024/25 quantile forecast comparison: four representative examples",
        fontsize=14, y=0.995,
    )
    fig.legend(
        legend_handles or [],
        legend_labels or [],
        loc="upper center", ncol=min(4, len(legend_handles or [])),
        frameon=False, bbox_to_anchor=(0.5, 0.965), fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=3.2, w_pad=2.5)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "mode": "quad",
        "primary_model": primary_model,
        "secondary_model": secondary_model,
        "overlay_model": overlay_name,
        "examples": metadata_examples,
    }


def plot_comparison(
    forecasts_by_model: dict[str, pd.DataFrame],
    truth: pd.DataFrame,
    location: str,
    reference_date: pd.Timestamp,
    history_weeks: int,
    output_path: Path,
    model_wis: dict[str, float],
    overlay_name: str | None = None,
    overlay_forecast: pd.DataFrame | None = None,
    overlay_wis: float | None = None,
) -> dict:
    location_rows = truth[truth["location"].eq(location)]
    location_name = location_rows["location_name"].dropna().iloc[0]
    history_start = reference_date - pd.Timedelta(weeks=history_weeks - 1)
    all_target_dates = sorted({dt for frame in forecasts_by_model.values() for dt in frame.index})
    plot_end = max(all_target_dates)

    observed = location_rows[
        location_rows["date"].between(history_start, plot_end)
    ].sort_values("date")
    if observed.empty:
        raise ValueError("No observed series found for the selected example window.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_models = len(forecasts_by_model)
    fig, axes = plt.subplots(n_models, 1, figsize=(11.5, 4.8 * n_models), sharex=True, sharey=True)
    if n_models == 1:
        axes = [axes]

    max_y = float(observed["value"].max())
    for frame in forecasts_by_model.values():
        max_y = max(max_y, float(frame[0.975].max()))
    if overlay_forecast is not None:
        max_y = max(max_y, float(overlay_forecast[0.975].max()))

    legend_handles = None
    for ax, (model_name, forecast) in zip(axes, forecasts_by_model.items()):
        ax.axvspan(reference_date, plot_end, color="#DBEAFE", alpha=0.25, zorder=0)
        line_obs, = ax.plot(
            observed["date"],
            observed["value"],
            color="black",
            linewidth=1.8,
            marker="o",
            markersize=4,
            label="Observed admissions",
            zorder=4,
        )
        band95 = ax.fill_between(
            forecast.index,
            forecast[0.025],
            forecast[0.975],
            color=INTERVAL_COLORS["95"],
            alpha=0.28,
            label="95% interval",
            zorder=1,
        )
        band80 = ax.fill_between(
            forecast.index,
            forecast[0.10],
            forecast[0.90],
            color=INTERVAL_COLORS["80"],
            alpha=0.28,
            label="80% interval",
            zorder=2,
        )
        band50 = ax.fill_between(
            forecast.index,
            forecast[0.25],
            forecast[0.75],
            color=INTERVAL_COLORS["50"],
            alpha=0.25,
            label="50% interval",
            zorder=3,
        )
        line_med, = ax.plot(
            forecast.index,
            forecast[0.50],
            color=INTERVAL_COLORS["median"],
            linewidth=2.0,
            marker="o",
            markersize=5,
            label="Median forecast",
            zorder=5,
        )
        overlay_band50 = None
        overlay_upper = None
        overlay_med = None
        if overlay_forecast is not None:
            # The FluSight-ensemble 50% central interval is drawn as two dashed
            # boundary lines rather than a filled band. A fill would have to
            # compete visually with the MEx model's three overlapping fills
            # (95%/80%/50%), which conflates the two forecasters. Hollow square
            # markers on the 0.25 and 0.75 quantile lines distinguish them from
            # the FluSight-ensemble median (solid-filled square markers).
            overlay_band50, = ax.plot(
                overlay_forecast.index,
                overlay_forecast[0.25],
                color=OVERLAY_COLORS["median"],
                linewidth=1.4,
                linestyle="--",
                marker="s",
                markersize=4,
                markerfacecolor="white",
                markeredgecolor=OVERLAY_COLORS["median"],
                markeredgewidth=1.2,
                label=f"{overlay_name} 50% interval",
                zorder=5.0,
            )
            overlay_upper, = ax.plot(
                overlay_forecast.index,
                overlay_forecast[0.75],
                color=OVERLAY_COLORS["median"],
                linewidth=1.4,
                linestyle="--",
                marker="s",
                markersize=4,
                markerfacecolor="white",
                markeredgecolor=OVERLAY_COLORS["median"],
                markeredgewidth=1.2,
                zorder=5.0,
            )
            overlay_med, = ax.plot(
                overlay_forecast.index,
                overlay_forecast[0.50],
                color=OVERLAY_COLORS["median"],
                linewidth=1.8,
                linestyle="--",
                marker="s",
                markersize=4,
                label=f"{overlay_name} median",
                zorder=5.2,
            )

        ax.axvline(reference_date, color="#374151", linestyle="--", linewidth=1.2)
        ax.text(
            reference_date,
            max_y * 1.02,
            "Reference date",
            rotation=90,
            va="top",
            ha="right",
            fontsize=9,
            color="#374151",
        )
        title = f"{model_name}  |  3-horizon mean WIS = {model_wis[model_name]:.1f}"
        if overlay_name is not None and overlay_wis is not None:
            title += f"  |  {overlay_name}: {overlay_wis:.1f}"
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Week ending date")
        ax.set_xlim(history_start - pd.Timedelta(days=2), plot_end + pd.Timedelta(days=2))
        ax.set_ylim(0, max_y * 1.1)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.SA, interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        if legend_handles is None:
            legend_handles = [line_obs, band95, band80, band50, line_med]
            if overlay_band50 is not None and overlay_med is not None:
                legend_handles.extend([overlay_band50, overlay_med])

    axes[0].set_ylabel("Weekly hospital admissions")
    fig.suptitle(
        f"Quantile forecast comparison: {location_name} ({FIPS_TO_ABBR[location]})",
        fontsize=15,
        y=1.01,
    )
    fig.legend(
        legend_handles,
        [
            "Observed admissions",
            "95% interval",
            "80% interval",
            "50% interval",
            "Median forecast",
            f"{overlay_name} 50% interval" if overlay_name is not None else "",
            f"{overlay_name} median" if overlay_name is not None else "",
        ][: len(legend_handles)],
        loc="upper left",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.06, 0.985),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "location_fips": location,
        "location_abbr": FIPS_TO_ABBR[location],
        "location_name": location_name,
        "reference_date": reference_date.strftime("%Y-%m-%d"),
        "history_start": history_start.strftime("%Y-%m-%d"),
        "target_end_dates": [ts.strftime("%Y-%m-%d") for ts in all_target_dates],
        "model_wis": model_wis,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dirs",
        nargs="+",
        type=Path,
        default=DEFAULT_MODEL_DIRS,
        help="One or more model submission directories to compare.",
    )
    parser.add_argument(
        "--target-path",
        type=Path,
        default=DEFAULT_TARGET_PATH,
        help=f"Authoritative target-hospital-admissions CSV (default: {DEFAULT_TARGET_PATH})",
    )
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=DEFAULT_OVERLAY_DIR,
        help=f"Optional benchmark submission directory to overlay (default: {DEFAULT_OVERLAY_DIR})",
    )
    parser.add_argument(
        "--state",
        type=str,
        default=None,
        help="Optional state selector (abbreviation, FIPS, or full name).",
    )
    parser.add_argument(
        "--reference-date",
        type=str,
        default=None,
        help="Optional reference date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--history-weeks",
        type=int,
        default=DEFAULT_HISTORY_WEEKS,
        help=f"Number of historical weeks to show before the forecast origin (default: {DEFAULT_HISTORY_WEEKS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output image path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--mode",
        choices=("single", "quad"),
        default="quad",
        help=(
            "single: one state/reference-date panel per model stacked vertically "
            "(legacy behaviour; requires explicit --state/--reference-date or falls "
            "back to above-median-burden median-WIS selection). "
            "quad: 2x2 of four representative examples spanning "
            "(early|late season) x (easier|harder WIS). Default: quad."
        ),
    )
    parser.add_argument(
        "--primary-model",
        type=str,
        default="MEx-ColaGNN",
        help="Primary model (full fan chart) for quad mode. Default: MEx-ColaGNN.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    truth = load_truth(args.target_path)
    submissions_by_model = {model_dir.name: load_submissions(model_dir) for model_dir in args.model_dirs}
    overlay_submissions = load_submissions(args.overlay_dir) if args.overlay_dir else None
    overlay_name = args.overlay_dir.name if args.overlay_dir else None

    if args.mode == "quad":
        if args.primary_model not in submissions_by_model:
            raise ValueError(
                f"--primary-model '{args.primary_model}' not found in model-dirs "
                f"{list(submissions_by_model.keys())}"
            )
        examples = choose_quad_examples(
            submissions_by_model,
            truth,
            overlay_submissions=overlay_submissions,
        )
        metadata = plot_quad_comparison(
            examples=examples,
            submissions_by_model=submissions_by_model,
            truth=truth,
            history_weeks=args.history_weeks,
            output_path=args.output,
            primary_model=args.primary_model,
            overlay_name=overlay_name,
            overlay_submissions=overlay_submissions,
        )
        meta_path = args.output.with_name(f"{args.output.stem}_metadata.json")
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, default=str)
        print(f"Saved figure to {args.output}")
        print(f"Saved metadata to {meta_path}")
        for ex in metadata["examples"]:
            print(f"  [{ex['bucket']}] {ex['location_name']} ({ex['location_abbr']}) ref {ex['reference_date']}")
        return

    location = resolve_location(args.state, truth)
    reference_date = pd.Timestamp(args.reference_date) if args.reference_date else None
    location, reference_date, selection_method, model_wis, realized_total = choose_example(
        submissions_by_model,
        truth,
        location,
        reference_date,
        overlay_submissions=overlay_submissions,
    )

    forecasts_by_model = {
        model_name: build_forecast_frame(submissions, location, reference_date)
        for model_name, submissions in submissions_by_model.items()
    }
    overlay_forecast = None
    overlay_wis = None
    if overlay_submissions is not None:
        overlay_forecast = build_forecast_frame(overlay_submissions, location, reference_date)
        overlay_truth = truth_for_targets(truth, location, overlay_forecast.index)
        overlay_wis = wis_for_frame(overlay_forecast, overlay_truth)
    metadata = plot_comparison(
        forecasts_by_model=forecasts_by_model,
        truth=truth,
        location=location,
        reference_date=reference_date,
        history_weeks=args.history_weeks,
        output_path=args.output,
        model_wis=model_wis,
        overlay_name=overlay_name,
        overlay_forecast=overlay_forecast,
        overlay_wis=overlay_wis,
    )
    metadata["models"] = list(forecasts_by_model.keys())
    metadata["selection_method"] = selection_method
    metadata["realized_total"] = realized_total
    if overlay_name is not None and overlay_wis is not None:
        metadata["overlay_model"] = overlay_name
        metadata["overlay_wis"] = overlay_wis

    meta_path = args.output.with_name(f"{args.output.stem}_metadata.json")
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)

    print(f"Saved figure to {args.output}")
    print(f"Saved metadata to {meta_path}")
    print(
        "Selected example:",
        metadata["location_name"],
        metadata["location_abbr"],
        metadata["reference_date"],
        f"({selection_method})",
    )


if __name__ == "__main__":
    main()
