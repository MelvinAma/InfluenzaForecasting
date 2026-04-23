"""Metrics formatting utilities for thesis reporting.

Converts the dicts returned by compute_metrics(), evaluate_epignn/colagnn(),
and cv_epignn/colagnn() into pandas DataFrames and LaTeX table fragments.

Usage:
    from src.eval.cross_validate import cv_epignn
    from src.eval.report import summary_table, fold_table, to_latex

    result = cv_epignn(config_path, hparam_config, ...)
    df = summary_table(result)          # aggregate row per metric
    fdf = fold_table(result)            # one row per fold
    print(to_latex(df, caption="EpiGNN US results"))
"""
from typing import Optional

import numpy as np
import pandas as pd


def summary_table(cv_result: dict) -> pd.DataFrame:
    """Build a one-row summary DataFrame from a cv_epignn/cv_colagnn result.

    Columns: mae_mean, mae_std, rmse_mean, rmse_std, mape_mean, mape_std,
             peak_intensity_mae, peak_week_mae, onset_week_mae, n_folds.
    """
    agg = cv_result["aggregate"]
    row = {
        "MAE":               f"{agg['mae_mean']:.1f} +/- {agg['mae_std']:.1f}",
        "RMSE":              f"{agg['rmse_mean']:.1f} +/- {agg['rmse_std']:.1f}",
        "MAPE (%)":          f"{agg['mape_mean']:.1f} +/- {agg['mape_std']:.1f}",
        "PCC":               f"{agg['pcc_mean']:.3f} +/- {agg['pcc_std']:.3f}",
        "Peak intensity MAE": f"{agg['peak_intensity_mae']:.1f}",
        "Peak week MAE":     f"{agg['peak_week_mae']:.1f}",
        "Onset week MAE":    f"{agg['onset_week_mae']:.1f}" if agg["onset_week_mae"] is not None else "N/A",
        "n folds":           agg["n_folds"],
    }
    if "n_seasons" in agg:
        row["n seasons"] = agg["n_seasons"]
    if "n_seeds" in agg:
        row["n seeds"] = agg["n_seeds"]
    return pd.DataFrame([row])


def fold_table(cv_result: dict) -> pd.DataFrame:
    """Build a per-fold DataFrame from a cv_epignn/cv_colagnn result.

    One row per fold. Columns: test_season, val_season, mae, rmse, mape,
    peak_intensity_mae, peak_week_mae.
    """
    rows = []
    for fold in cv_result["folds"]:
        m = fold["metrics"]
        rows.append({
            "Test season":       fold["test_season"],
            "Val season":        fold["val_season"],
            "MAE":               round(m["mae_mean"], 2),
            "RMSE":              round(m["rmse_mean"], 2),
            "MAPE (%)":          round(float(np.nanmean(m["mape"])), 2),
            "Peak intensity MAE": round(float(np.mean(np.abs(m["peak_intensity_error"]))), 2),
            "Peak week MAE":     round(float(np.mean(np.abs(m["peak_week_error"]))), 2),
        })
    return pd.DataFrame(rows)


def region_table(metrics: dict, region_names: Optional[list[str]] = None) -> pd.DataFrame:
    """Build a per-region DataFrame from a compute_metrics() result.

    Parameters
    ----------
    metrics : dict
        Output of compute_metrics() or evaluate_epignn/colagnn().
    region_names : list of str, optional
        Region labels. If None, regions are labelled 0, 1, 2, ...

    Returns
    -------
    DataFrame with columns: region, mae, rmse, mape, peak_intensity_error,
    peak_week_error, onset_week_error.
    """
    m = metrics["n_regions"]
    names = region_names if region_names is not None else list(range(m))
    if region_names is not None and len(names) != m:
        raise ValueError(
            f"region_names has {len(names)} entries but metrics has {m} regions"
        )
    data = {
        "Region":              names,
        "MAE":                 metrics["mae"].round(2),
        "RMSE":                metrics["rmse"].round(2),
        "MAPE (%)":            np.round(metrics["mape"], 2),
        "Peak intensity error": np.round(metrics["peak_intensity_error"], 2),
        "Peak week error":     metrics["peak_week_error"].astype(int),
    }
    if metrics["onset_week_error"] is not None:
        data["Onset week error"] = metrics["onset_week_error"].round(1)
    return pd.DataFrame(data)


def to_latex(df: pd.DataFrame, caption: str = "", label: str = "",
             index: bool = False) -> str:
    """Convert a DataFrame to a LaTeX table string.

    Produces a table environment with booktabs-style rules. Requires the
    ``booktabs`` package in the LaTeX preamble (standard in the KTH template).

    Parameters
    ----------
    df : pd.DataFrame
    caption : str
        Text for the \\caption{} command.
    label : str
        Text for the \\label{} command (without the ``tab:`` prefix).
    index : bool
        Whether to include the DataFrame index as a column.

    Returns
    -------
    str -- complete LaTeX table environment.
    """
    col_fmt = "l" + "r" * (len(df.columns) - (1 if index else 0))
    body = df.to_latex(index=index, column_format=col_fmt,
                       escape=True, na_rep="N/A",
                       caption=caption if caption else None,
                       label=f"tab:{label}" if label else None)
    return body
