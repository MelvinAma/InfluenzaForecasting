"""Fetch historical NHSN data vintages for FluSight-aligned evaluation.

For each FluSight reference_date, retrieves the preliminary NHSN data that
was available to participants at the submission deadline (Wednesday before
the Saturday reference_date). The vintage data extends through ref_date - 7.

Sources (priority order):
  1. Cached: data/vintages/target_{ref_date}.csv
  2. FluSight hub archive: auxiliary-data/target-data-archive/
     Mapping: archive file named _(ref_date - 7).csv is the correct vintage
  3. Delphi Epidata API: as_of=epiweek(ref_date) gives data through ref_date - 7

Output per reference_date:
  data/vintages/target_{ref_date}.csv             — hub-format CSV
  data/vintages/state_flu_admissions_{ref_date}.txt — model-input format
  data/vintages/date_index_{ref_date}.csv          — date index for model

Usage:
    python scripts/fetch_data_vintages.py [--season 2024/25|2025/26|both]
    python scripts/fetch_data_vintages.py --validate  # check Delphi vs archive
"""
import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VINTAGE_DIR = PROJECT_ROOT / "data" / "vintages"
VINTAGE_DIR.mkdir(parents=True, exist_ok=True)

ARCHIVE_DIR = (
    PROJECT_ROOT / "data" / "FluSight-forecast-hub"
    / "auxiliary-data" / "target-data-archive"
)

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

state_idx = pd.read_csv(
    PROJECT_ROOT / "src" / "data" / "state_index.csv",
    header=None, names=["idx", "abbr"],
)
STATES = list(state_idx["abbr"])

DELPHI_URL = (
    "https://api.delphi.cmu.edu/epidata/covidcast/"
    "?data_source=nhsn"
    "&signal=confirmed_admissions_flu_ew_prelim"
    "&geo_type=state&geo_value=*"
    "&time_type=week"
    "&time_values={time_range}"
    "&as_of={as_of}"
)

SEASON_WINDOWS = {
    "2024/25": ("2024-10-26", "2025-05-31"),
    "2025/26": ("2025-10-25", "2026-03-28"),
}


def saturdays_in_range(start_str, end_str):
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    if start.weekday() != 5:
        start += timedelta(days=(5 - start.weekday()) % 7)
    dates = []
    while start <= end:
        dates.append(start.strftime("%Y-%m-%d"))
        start += timedelta(weeks=1)
    return dates


def cdc_epiweek(date_str):
    """Convert a date string (YYYY-MM-DD) to CDC epiweek YYYYWW.

    CDC epiweeks run Sunday-Saturday. Week 1 of a year is the week
    containing January 4th.
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    # Find the Sunday of this CDC week
    dow = dt.weekday()  # Mon=0, Sun=6
    days_since_sunday = (dow + 1) % 7
    sunday = dt - timedelta(days=days_since_sunday)

    # Week 1 contains Jan 4. Find Sunday of week 1.
    jan4 = datetime(sunday.year, 1, 4)
    jan4_dow = jan4.weekday()
    week1_sunday = jan4 - timedelta(days=(jan4_dow + 1) % 7)

    diff = (sunday - week1_sunday).days
    week_num = diff // 7 + 1
    year = sunday.year

    if week_num < 1:
        prev_jan4 = datetime(year - 1, 1, 4)
        prev_jan4_dow = prev_jan4.weekday()
        prev_week1_sun = prev_jan4 - timedelta(days=(prev_jan4_dow + 1) % 7)
        week_num = (sunday - prev_week1_sun).days // 7 + 1
        year -= 1
    elif week_num > 52:
        next_jan4 = datetime(year + 1, 1, 4)
        next_jan4_dow = next_jan4.weekday()
        next_week1_sun = next_jan4 - timedelta(days=(next_jan4_dow + 1) % 7)
        if sunday >= next_week1_sun:
            year += 1
            week_num = 1

    return f"{year}{week_num:02d}"


def epiweek_to_saturday(ew_str):
    """Convert CDC epiweek YYYYWW to the Saturday ending that week."""
    year = int(ew_str[:4])
    week = int(ew_str[4:])
    jan4 = datetime(year, 1, 4)
    jan4_dow = jan4.weekday()
    week1_sunday = jan4 - timedelta(days=(jan4_dow + 1) % 7)
    saturday = week1_sunday + timedelta(weeks=week - 1, days=6)
    return saturday


def load_from_archive(ref_date):
    """Load vintage from FluSight hub archive.

    For ref_date X, the correct archive file is _(X-7).csv because
    at the Wednesday deadline, participants had data through the
    previous Saturday (ref_date - 7). The archive file is named
    by the max date in its data.
    """
    needed_date = (
        datetime.strptime(ref_date, "%Y-%m-%d") - timedelta(days=7)
    ).strftime("%Y-%m-%d")
    path = ARCHIVE_DIR / f"target-hospital-admissions_{needed_date}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["date"])


def fetch_from_delphi(ref_date):
    """Fetch vintage data from Delphi Epidata API.

    Uses as_of=epiweek(ref_date), which returns data through
    ref_date - 7 (1-week NHSN publication lag).
    """
    ew = cdc_epiweek(ref_date)
    # Request data going back ~4 years to cover full training history
    dt = datetime.strptime(ref_date, "%Y-%m-%d")
    start_ew = cdc_epiweek((dt - timedelta(weeks=200)).strftime("%Y-%m-%d"))
    url = DELPHI_URL.format(time_range=f"{start_ew}-{ew}", as_of=ew)

    try:
        response = urllib.request.urlopen(url, timeout=120)
        data = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"    Delphi API error: {e}")
        return None

    if data.get("result") != 1 or not data.get("epidata"):
        print(f"    Delphi API: no data (result={data.get('result')})")
        return None

    rows = []
    for rec in data["epidata"]:
        if rec.get("missing_value") and rec["missing_value"] != 0:
            continue
        saturday = epiweek_to_saturday(str(rec["time_value"]))
        geo = rec["geo_value"].upper()
        fips = ABBR_TO_FIPS.get(geo)
        if fips is None:
            continue
        if geo not in [s for s in STATES]:
            continue
        rows.append({
            "date": saturday,
            "location": fips,
            "location_name": "",
            "value": rec["value"],
            "weekly_rate": 0.0,
        })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def save_vintage(ref_date, df):
    """Save vintage in all required formats."""
    df = df.copy()
    df["location"] = df["location"].astype(str).str.zfill(2)

    csv_path = VINTAGE_DIR / f"target_{ref_date}.csv"
    df.to_csv(csv_path, index=False)

    dates = sorted(df["date"].unique())
    rows = []
    for d in dates:
        day_data = df[df["date"] == d]
        row = []
        for abbr in STATES:
            fips = ABBR_TO_FIPS[abbr]
            match = day_data[day_data["location"] == fips]
            row.append(float(match["value"].values[0]) if not match.empty else 0.0)
        rows.append(row)

    data_path = VINTAGE_DIR / f"state_flu_admissions_{ref_date}.txt"
    with open(data_path, "w") as f:
        for row in rows:
            f.write(",".join(f"{v:.1f}" for v in row) + "\n")

    dates_path = VINTAGE_DIR / f"date_index_{ref_date}.csv"
    with open(dates_path, "w") as f:
        for i, d in enumerate(dates):
            f.write(f"{i},{pd.Timestamp(d).strftime('%Y-%m-%d')}\n")

    return len(dates)


def compare_vintages(prelim_df, final_df, ref_date):
    """Compare preliminary vs final values for recent weeks."""
    prelim_df = prelim_df.copy()
    final_df = final_df.copy()
    prelim_df["location"] = prelim_df["location"].astype(str).str.zfill(2)
    final_df["location"] = final_df["location"].astype(str).str.zfill(2)

    ref_dt = pd.Timestamp(ref_date)
    dates_to_check = [ref_dt - pd.Timedelta(weeks=i) for i in range(1, 5)]

    results = []
    for d in dates_to_check:
        p_day = prelim_df[prelim_df["date"] == d]
        f_day = final_df[final_df["date"] == d]
        if p_day.empty or f_day.empty:
            continue

        revisions = []
        for abbr in STATES:
            fips = ABBR_TO_FIPS[abbr]
            p_val = p_day[p_day["location"] == fips]["value"]
            f_val = f_day[f_day["location"] == fips]["value"]
            if not p_val.empty and not f_val.empty:
                pv, fv = float(p_val.values[0]), float(f_val.values[0])
                rev = (fv - pv) / pv * 100 if pv > 0 else 0
                revisions.append({"state": abbr, "prelim": pv, "final": fv, "revision_pct": rev})

        if not revisions:
            continue

        p_sum = sum(r["prelim"] for r in revisions)
        f_sum = sum(r["final"] for r in revisions)
        rev_pcts = [abs(r["revision_pct"]) for r in revisions]

        results.append({
            "reference_date": ref_date,
            "date": d,
            "weeks_before_ref": int((ref_dt - d).days / 7),
            "prelim_sum": p_sum,
            "final_sum": f_sum,
            "national_revision_pct": (f_sum - p_sum) / p_sum * 100 if p_sum > 0 else 0,
            "mean_abs_state_revision_pct": np.mean(rev_pcts),
            "max_state_revision_pct": max(rev_pcts),
            "n_states": len(revisions),
        })

    return results


def validate_delphi_vs_archive():
    """Compare Delphi API values against hub archive for quality assurance."""
    test_ref = "2025-02-08"
    archive_date = "2025-02-01"

    archive_path = ARCHIVE_DIR / f"target-hospital-admissions_{archive_date}.csv"
    if not archive_path.exists():
        print("Validation skipped: archive file not found")
        return

    archive_df = pd.read_csv(archive_path, parse_dates=["date"])
    archive_df["location"] = archive_df["location"].astype(str).str.zfill(2)

    print(f"Fetching Delphi data for ref_date {test_ref}...")
    delphi_df = fetch_from_delphi(test_ref)
    if delphi_df is None:
        print("Validation failed: Delphi returned no data")
        return

    delphi_df["location"] = delphi_df["location"].astype(str).str.zfill(2)

    check_date = pd.Timestamp(archive_date)
    a_day = archive_df[archive_df["date"] == check_date]
    d_day = delphi_df[delphi_df["date"] == check_date]

    if a_day.empty or d_day.empty:
        print(f"Validation: no overlap on {archive_date}")
        return

    n_match, n_diff, diffs = 0, 0, []
    for abbr in STATES:
        fips = ABBR_TO_FIPS[abbr]
        a_val = a_day[a_day["location"] == fips]["value"]
        d_val = d_day[d_day["location"] == fips]["value"]
        if a_val.empty or d_val.empty:
            continue
        av, dv = float(a_val.values[0]), float(d_val.values[0])
        if av == dv:
            n_match += 1
        else:
            n_diff += 1
            diffs.append(f"    {abbr}: archive={av}, delphi={dv}, diff={dv-av:+.1f}")

    print(f"Validation for {archive_date}: {n_match} match, {n_diff} differ out of {n_match+n_diff}")
    if diffs:
        for d in diffs[:10]:
            print(d)
        if len(diffs) > 10:
            print(f"    ... and {len(diffs)-10} more")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", default="both", choices=["2024/25", "2025/26", "both"])
    parser.add_argument("--validate", action="store_true", help="Run Delphi vs archive validation")
    args = parser.parse_args()

    if args.validate:
        validate_delphi_vs_archive()
        return

    seasons = list(SEASON_WINDOWS.keys()) if args.season == "both" else [args.season]

    final_path = (
        PROJECT_ROOT / "data" / "FluSight-forecast-hub"
        / "target-data" / "target-hospital-admissions.csv"
    )
    final_df = pd.read_csv(final_path, parse_dates=["date"])

    all_revisions = []
    stats = {"cached": 0, "archive": 0, "delphi": 0, "failed": 0}

    for season in seasons:
        start, end = SEASON_WINDOWS[season]
        ref_dates = saturdays_in_range(start, end)
        print(f"\n{'='*60}")
        print(f"Season {season}: {len(ref_dates)} reference dates ({start} to {end})")
        print(f"{'='*60}")

        for ref_date in ref_dates:
            cache_path = VINTAGE_DIR / f"target_{ref_date}.csv"

            if cache_path.exists():
                df = pd.read_csv(cache_path, parse_dates=["date"])
                max_d = df["date"].max().strftime("%Y-%m-%d")
                print(f"\n{ref_date}: cached (through {max_d})")
                revisions = compare_vintages(df, final_df, ref_date)
                all_revisions.extend(revisions)
                stats["cached"] += 1
                continue

            needed_archive = (
                datetime.strptime(ref_date, "%Y-%m-%d") - timedelta(days=7)
            ).strftime("%Y-%m-%d")

            df = load_from_archive(ref_date)
            if df is not None:
                source = f"archive _{needed_archive}"
                stats["archive"] += 1
            else:
                ew = cdc_epiweek(ref_date)
                print(f"\n{ref_date}: archive _{needed_archive} missing, querying Delphi (as_of={ew})...")
                df = fetch_from_delphi(ref_date)
                source = f"Delphi (as_of={ew})"
                if df is not None:
                    stats["delphi"] += 1
                else:
                    stats["failed"] += 1
                    print(f"  FAILED — no data available")
                    continue
                time.sleep(1)

            n_dates = save_vintage(ref_date, df)
            max_d = df["date"].max()
            if isinstance(max_d, pd.Timestamp):
                max_d = max_d.strftime("%Y-%m-%d")
            print(f"\n{ref_date}: {source} — {n_dates} dates (through {max_d})")

            revisions = compare_vintages(df, final_df, ref_date)
            all_revisions.extend(revisions)
            for r in revisions:
                age = r["weeks_before_ref"]
                print(f"  ref-{age}w: national {r['national_revision_pct']:+.1f}%, "
                      f"mean |state rev|={r['mean_abs_state_revision_pct']:.1f}%")

    if all_revisions:
        rev_df = pd.DataFrame(all_revisions)
        rev_df.to_csv(VINTAGE_DIR / "revision_analysis.csv", index=False)

        print(f"\n{'='*60}")
        print("REVISION ANALYSIS SUMMARY")
        print(f"{'='*60}")
        for age in sorted(rev_df["weeks_before_ref"].unique()):
            age_df = rev_df[rev_df["weeks_before_ref"] == age]
            print(f"\n  Week ref-{age} (n={len(age_df)}):")
            print(f"    National revision: {age_df['national_revision_pct'].mean():+.1f}% "
                  f"(range {age_df['national_revision_pct'].min():+.1f}% to "
                  f"{age_df['national_revision_pct'].max():+.1f}%)")
            print(f"    Mean |state rev|:  {age_df['mean_abs_state_revision_pct'].mean():.1f}%")

    print(f"\n{'='*60}")
    print(f"Sources: {stats['cached']} cached, {stats['archive']} archive, "
          f"{stats['delphi']} Delphi, {stats['failed']} failed")
    print(f"Vintage data: {VINTAGE_DIR}")


if __name__ == "__main__":
    main()
