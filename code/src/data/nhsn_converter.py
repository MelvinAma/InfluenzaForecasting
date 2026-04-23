#!/usr/bin/env python3
"""
Convert NHSN Weekly Hospital Respiratory Data to EpiGNN/ColaGNN format.

Input: NHSN CSV (semicolon-separated) with columns:
  - "Week Ending Date"
  - "Geographic aggregation" (state codes)
  - "Total Influenza Admissions"

Output: state_flu_admissions.txt
  - Rows = weeks (chronological)
  - Columns = states (alphabetical, excluding territories)
  - Values = comma-separated float counts
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

# US States + DC (51 total), alphabetical order
US_STATES = [
    "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL",
    "GA", "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA",
    "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE",
    "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY"
]

# Excluded: AS (American Samoa), GU (Guam), MP (N. Mariana Islands),
#           PR (Puerto Rico), VI (Virgin Islands), Region 1-10, USA


def parse_number(val: str) -> float:
    """Parse number string, handling commas and empty values."""
    if not val or val.strip() == '':
        return 0.0
    val = val.replace(',', '').replace('"', '').strip()
    try:
        return float(val)
    except ValueError:
        return 0.0


def load_nhsn_data(filepath: Path) -> dict[str, dict[str, float]]:
    """
    Load NHSN CSV and extract flu admissions by week and state.

    Returns: {week_date: {state: admissions}}
    """
    data = defaultdict(dict)

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)

        # Validate required columns exist
        required_cols = ['Week Ending Date', 'Geographic aggregation', 'Total Influenza Admissions']
        missing = [col for col in required_cols if col not in header]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Available: {header}")

        # Find column indices
        date_col = header.index('Week Ending Date')
        geo_col = header.index('Geographic aggregation')
        flu_col = header.index('Total Influenza Admissions')

        for row in reader:
            if len(row) <= max(date_col, geo_col, flu_col):
                continue

            date = row[date_col].strip('"')
            state = row[geo_col].strip('"')
            flu_val = row[flu_col]

            # Only keep US states
            if state not in US_STATES:
                continue

            admissions = parse_number(flu_val)

            # Detect duplicates
            if state in data[date]:
                print(f"WARNING: Duplicate data for {state} on {date}. "
                      f"Old: {data[date][state]}, New: {admissions}")

            data[date][state] = admissions

    return data


def convert_to_matrix(data: dict[str, dict[str, float]],
                      states: list[str]) -> tuple[list[str], list[list[float]]]:
    """
    Convert data dict to time series matrix.

    Returns: (sorted_dates, matrix)
      - matrix[t][s] = admissions for week t, state s
    """
    # Sort dates chronologically
    sorted_dates = sorted(data.keys())

    matrix = []
    for date in sorted_dates:
        row = []
        for state in states:
            val = data[date].get(state, 0.0)
            row.append(val)
        matrix.append(row)

    return sorted_dates, matrix


def write_matrix(matrix: list[list[float]], output_path: Path):
    """Write matrix to EpiGNN format (comma-separated, no header)."""
    with open(output_path, 'w') as f:
        for row in matrix:
            line = ','.join(f'{v:.1f}' for v in row)
            f.write(line + '\n')


def write_dates(dates: list[str], output_path: Path):
    """Write date index for reference."""
    with open(output_path, 'w') as f:
        for i, date in enumerate(dates):
            f.write(f'{i},{date}\n')


def write_states(states: list[str], output_path: Path):
    """Write state index for reference."""
    with open(output_path, 'w') as f:
        for i, state in enumerate(states):
            f.write(f'{i},{state}\n')


def main():
    # Paths - Data is in main repo, not worktree
    main_repo = Path('C:/Users/Admin/Desktop/MEx')
    data_dir = main_repo / 'Data'
    nhsn_files = list(data_dir.glob('Weekly_Hospital_Respiratory_Data*.csv'))

    if not nhsn_files:
        print(f"Error: No NHSN CSV found in {data_dir}")
        sys.exit(1)

    nhsn_path = nhsn_files[0]
    print(f"Loading: {nhsn_path.name}")

    output_dir = Path(__file__).parent

    # Load and convert
    data = load_nhsn_data(nhsn_path)
    print(f"Loaded {len(data)} weeks of data")

    dates, matrix = convert_to_matrix(data, US_STATES)

    # Validate matrix dimensions
    expected_cols = len(US_STATES)
    actual_rows = len(matrix)
    actual_cols = len(matrix[0]) if matrix else 0

    if actual_cols != expected_cols:
        raise ValueError(f"Matrix dimension mismatch: expected {expected_cols} columns, got {actual_cols}")

    if actual_rows == 0:
        raise ValueError("Matrix is empty - no data was converted")

    print(f"Matrix shape validated: {actual_rows} weeks x {actual_cols} states")
    print(f"Date range: {dates[0]} to {dates[-1]}")

    # Check for missing data
    missing_count = sum(1 for row in matrix for val in row if val == 0.0)
    total_cells = len(matrix) * len(US_STATES)
    print(f"Zero values: {missing_count}/{total_cells} ({100*missing_count/total_cells:.1f}%)")

    # Write outputs
    write_matrix(matrix, output_dir / 'state_flu_admissions.txt')
    write_dates(dates, output_dir / 'date_index.csv')
    write_states(US_STATES, output_dir / 'state_index.csv')

    print(f"\nOutputs written to {output_dir}:")
    print("  - state_flu_admissions.txt (model input)")
    print("  - date_index.csv (week reference)")
    print("  - state_index.csv (state order reference)")


if __name__ == '__main__':
    main()
