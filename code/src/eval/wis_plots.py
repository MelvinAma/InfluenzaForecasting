# MEx Thesis Adaptation
# Per-week WIS comparison: ColaGNN vs FluSight-ensemble (2024/25 season).
# Uses cached predictions from results/flusight_checkpoints/.
# Runs a two-sided sign test and saves a two-panel comparison figure.
from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest


def run_wis_per_week_analysis(project_root: Path, save_fig: bool = True) -> dict:
    """Generate per-week WIS comparison: ColaGNN vs FluSight-ensemble.

    Derives overlap window, loads cached ColaGNN predictions, fits quantile
    regression on calibration data, then computes and compares per-week WIS
    against FluSight-ensemble over the 24-week overlap (2024-11-30 → 2025-05-17).

    Parameters
    ----------
    project_root:
        Absolute path to the MEx project root.
    save_fig:
        If True, save figure to results/figures/wis_per_week_comparison.png.

    Returns
    -------
    dict with keys:
        colagnn_wis_per_week  -- np.ndarray (24,)
        flusight_wis_per_week -- np.ndarray (24,)
        colagnn_wins          -- int
        flusight_wins         -- int
        pvalue                -- float  (two-sided sign test)
        dates                 -- list[str]  YYYY-MM-DD, one per overlap week
    """
    sys.path.insert(0, str(project_root))

    from src.data.loader import _load_date_index, assign_seasons
    from src.eval.quantile import (
        FLUSIGHT_LEVELS,
        compute_wis,
        fit_quantile_model,
        predict_quantiles,
    )

    # --- Constants ---
    SEEDS = [42, 123, 456]
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
    ABBR_TO_FIPS = {v: k for k, v in FIPS_TO_ABBR.items()}

    _sidx = pd.read_csv(
        project_root / "src" / "data" / "state_index.csv",
        header=None, names=["idx", "abbr"],
    )
    STATES = list(_sidx["abbr"])
    assert len(STATES) == 51

    Q = len(FLUSIGHT_LEVELS)
    level_to_q = {round(lvl, 4): i for i, lvl in enumerate(FLUSIGHT_LEVELS)}
    level_set = set(level_to_q)

    # --- Step 1: Derive TEST_DATES (same logic as FluDataLoader._split_by_season) ---
    date_idx_path = project_root / "src" / "data" / "date_index.csv"
    date_df = pd.read_csv(
        date_idx_path, header=None, names=["idx", "date"], parse_dates=["date"]
    )
    idx_to_date = dict(zip(date_df["idx"], date_df["date"]))

    iso_years, iso_weeks = _load_date_index(date_idx_path, "date")
    seasons = assign_seasons(iso_years, iso_weeks)

    P, h = 20, 1
    test_indices = [
        i for i in range(P + h - 1, len(seasons))
        if seasons[i] == "2024/25"
    ]
    TEST_DATES = pd.DatetimeIndex([idx_to_date[i] for i in test_indices])
    T_TEST = len(test_indices)

    # Build authoritative ground truth matrix (T_TEST, 51) from FluSight NHSN CSV
    target_df = pd.read_csv(
        project_root / "data" / "flusight_2024_25" / "target-hospital-admissions.csv",
        parse_dates=["date"],
    )
    target_df["location"] = target_df["location"].astype(str).str.zfill(2)

    gt = np.full((T_TEST, len(STATES)), np.nan)
    for t, d in enumerate(TEST_DATES):
        day_rows = target_df[target_df["date"] == d]
        for r, abbr in enumerate(STATES):
            match = day_rows[day_rows["location"] == ABBR_TO_FIPS[abbr]]
            if not match.empty:
                gt[t, r] = float(match["value"].values[0])

    # --- Step 2: Load FluSight-ensemble; derive overlap_test_rows ---
    FLUSIGHT_DIR = project_root / "data" / "flusight_2024_25" / "model-output"

    def _load_flusight_model(model_name: str):
        """Parse cached FluSight CSVs into (qpred, gt_aligned, dates, tidx) arrays."""
        model_dir = FLUSIGHT_DIR / model_name
        qpred_rows, gt_rows, date_rows, tidx_rows = [], [], [], []

        for t, target_date in enumerate(TEST_DATES):
            ref_date = target_date - timedelta(days=7)
            cache_path = model_dir / f"{ref_date.strftime('%Y-%m-%d')}-{model_name}.csv"
            if not cache_path.exists():
                continue
            try:
                df = pd.read_csv(cache_path)
            except Exception:
                continue

            df.columns = df.columns.str.lower()
            df["location"] = df["location"].astype(str).str.zfill(2)
            mask = (
                (df["horizon"] == 1)
                & (df["output_type"].str.lower() == "quantile")
                & (df["location"].isin(FIPS_TO_ABBR))
            )
            df_h1 = df[mask].copy()
            if df_h1.empty:
                continue

            df_h1["q_level"] = df_h1["output_type_id"].apply(
                lambda x: round(float(x), 4)
            )
            df_h1 = df_h1[df_h1["q_level"].isin(level_set)]

            frame = np.full((len(STATES), Q), np.nan)
            for _, row in df_h1.iterrows():
                abbr = FIPS_TO_ABBR.get(row["location"])
                q_idx = level_to_q.get(row["q_level"])
                if abbr in STATES and q_idx is not None:
                    frame[STATES.index(abbr), q_idx] = float(row["value"])

            if np.sum(~np.isnan(frame[:, 0])) < 26:
                continue

            for r in range(len(STATES)):
                q_row = frame[r]
                if np.isnan(q_row).any() and not np.isnan(q_row).all():
                    known = np.where(~np.isnan(q_row))[0]
                    frame[r] = np.interp(np.arange(Q), known, q_row[known])

            qpred_rows.append(frame)
            gt_rows.append(gt[t])
            date_rows.append(target_date)
            tidx_rows.append(t)

        if not qpred_rows:
            return (
                np.empty((0, len(STATES), Q)),
                np.empty((0, len(STATES))),
                pd.DatetimeIndex([]),
                [],
            )
        return (
            np.stack(qpred_rows),
            np.stack(gt_rows),
            pd.DatetimeIndex(date_rows),
            tidx_rows,
        )

    qpred_fs, gt_fs, _, fs_overlap_tidx = _load_flusight_model("FluSight-ensemble")

    # Restrict to rows where our gt is fully observed (no NaN any state)
    overlap_test_rows = [
        t for t in fs_overlap_tidx
        if t < T_TEST and not np.isnan(gt[t]).any()
    ]
    N_OVERLAP = len(overlap_test_rows)

    # --- Step 3: Load ColaGNN cached predictions; seed-average for test ---
    ckpt_dir = project_root / "results" / "flusight_checkpoints"

    def _load_json(path: Path) -> np.ndarray:
        with open(path) as f:
            return np.array(json.load(f))

    seed_preds = [
        _load_json(ckpt_dir / f"preds_colagnn_test_seed{s}.json") for s in SEEDS
    ]
    test_avg = np.mean(seed_preds, axis=0)          # (33, 51)
    test_overlap = test_avg[overlap_test_rows]       # (N_OVERLAP, 51)

    # --- Step 4: Load cal predictions; fit quantile regression on all 99 rows ---
    cal_pred = _load_json(ckpt_dir / "preds_colagnn_cal_pred.json")   # (99, 51)
    cal_true = _load_json(ckpt_dir / "preds_colagnn_cal_true.json")   # (99, 51)
    qmodel = fit_quantile_model(cal_pred, cal_true, FLUSIGHT_LEVELS, alpha=0.0)

    # --- Step 5: ColaGNN per-week WIS on overlap window ---
    gt_overlap = gt[overlap_test_rows]                                 # (N_OVERLAP, 51)
    qpred_colagnn = predict_quantiles(qmodel, test_overlap, enforce_monotone=True)
    colagnn_wis_per_week = compute_wis(
        qpred_colagnn, gt_overlap, FLUSIGHT_LEVELS
    )["wis_per_week"]                                                  # (N_OVERLAP,)

    # --- Step 6: FluSight-ensemble per-week WIS on same overlap window ---
    overlap_set = set(overlap_test_rows)
    valid_fs_idx = [i for i, t in enumerate(fs_overlap_tidx) if t in overlap_set]
    flusight_wis_per_week = compute_wis(
        qpred_fs[valid_fs_idx],
        gt_fs[valid_fs_idx],
        FLUSIGHT_LEVELS,
    )["wis_per_week"]                                                  # (N_OVERLAP,)

    # --- Step 7: Two-sided sign test ---
    colagnn_wins = int(np.sum(colagnn_wis_per_week < flusight_wis_per_week))
    flusight_wins = int(np.sum(flusight_wis_per_week < colagnn_wis_per_week))
    pvalue = float(binomtest(colagnn_wins, n=N_OVERLAP, p=0.5).pvalue)

    # --- Step 8: Figure ---
    overlap_dates = [TEST_DATES[t] for t in overlap_test_rows]
    x_labels = [d.strftime("%b %d") for d in overlap_dates]
    x_pos = np.arange(N_OVERLAP)

    COLOR_COLAGNN = "#2563EB"
    COLOR_FLUSIGHT = "#DC2626"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.subplots_adjust(hspace=0.4)

    # Panel 1: per-week WIS lines
    ax1.plot(
        x_pos, colagnn_wis_per_week,
        color=COLOR_COLAGNN, marker="o", markersize=4, linewidth=1.5,
        label="ColaGNN",
    )
    ax1.plot(
        x_pos, flusight_wis_per_week,
        color=COLOR_FLUSIGHT, marker="s", markersize=4, linewidth=1.5,
        label="FluSight-ensemble",
    )
    ax1.set_ylabel("Mean WIS (across 51 states)")
    ax1.set_title("Per-week WIS: ColaGNN vs FluSight-ensemble  (2024/25 overlap window)")
    ax1.legend(framealpha=0.9, fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: WIS difference (positive = FluSight better)
    diff = colagnn_wis_per_week - flusight_wis_per_week
    bar_colors = [COLOR_COLAGNN if d < 0 else COLOR_FLUSIGHT for d in diff]
    ax2.bar(x_pos, diff, color=bar_colors, alpha=0.75)
    ax2.axhline(0, linestyle="--", color="black", linewidth=0.9)
    ax2.set_ylabel("ColaGNN − FluSight WIS\n(positive = FluSight better)")
    ax2.set_title(
        f"WIS Difference per Week  |  sign test p = {pvalue:.3f}"
        f"  (ColaGNN wins: {colagnn_wins}/{N_OVERLAP})"
    )
    ax2.grid(True, alpha=0.3)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)

    if save_fig:
        fig_path = project_root / "results" / "figures" / "wis_per_week_comparison.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure saved to: {fig_path}")

    return {
        "colagnn_wis_per_week":  colagnn_wis_per_week,
        "flusight_wis_per_week": flusight_wis_per_week,
        "colagnn_wins":          colagnn_wins,
        "flusight_wins":         flusight_wins,
        "pvalue":                pvalue,
        "dates":                 [d.strftime("%Y-%m-%d") for d in overlap_dates],
    }


if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[2]
    results = run_wis_per_week_analysis(_root)
    n = len(results["dates"])
    print(f"\nSign test (two-sided, H0: equal win probability, n={n})")
    print(f"  ColaGNN wins  : {results['colagnn_wins']} / {n}")
    print(f"  FluSight wins : {results['flusight_wins']} / {n}")
    print(f"  p-value       : {results['pvalue']:.4f}")
    print(f"\nFigure saved to: results/figures/wis_per_week_comparison.png")
