#!/usr/bin/env python3
"""
Convert Swedish Folkhälsomyndigheten data to EpiGNN/ColaGNN format.

Input: Wide-format CSV (comma-separated) with columns:
  - "Region" (row labels for 21 Swedish counties)
  - "Totalt influensa Antal fall YYYY v WW" (one column per week)

Output: sweden_flu_cases.txt
  - Rows = weeks (chronological)
  - Columns = regions (alphabetical order)
  - Values = comma-separated float counts
"""

import csv
import sys
import re
from pathlib import Path

SWEDISH_REGIONS = [
    "Blekinge",
    "Dalarna",
    "Gotland",
    "Gävleborg",
    "Halland",
    "Jämtland Härjedalen",
    "Jönköping",
    "Kalmar",
    "Kronoberg",
    "Norrbotten",
    "Skåne",
    "Stockholm",
    "Södermanland",
    "Uppsala",
    "Värmland",
    "Västerbotten",
    "Västernorrland",
    "Västmanland",
    "Västra Götaland",
    "Örebro",
    "Östergötland"
]


def parse_week_column(col_name: str) -> tuple[int, int] | None:
    """
    Parse column name like 'Totalt influensa Antal fall 2015 v 40'.

    Returns: (year, week) or None if not a week column
    """
    match = re.search(r'(\d{4})\s+v\s+(\d{2})', col_name)
    if match:
        year = int(match.group(1))
        week = int(match.group(2))

        # Validate ISO week number
        if not (1 <= week <= 53):
            print(f"WARNING: Invalid week number {week} in year {year}")
            return None

        # Validate year is reasonable
        if not (2000 <= year <= 2030):
            print(f"WARNING: Suspicious year {year}")
            return None

        return (year, week)
    return None


def parse_number(val: str) -> float:
    """Parse number string, handling empty values."""
    if not val or val.strip() == '':
        return 0.0
    val = val.replace(',', '').replace('"', '').strip()
    try:
        return float(val)
    except ValueError:
        return 0.0


def load_swedish_data(filepath: Path) -> tuple[list[tuple[int, int]], dict[str, list[float]]]:
    """
    Load Swedish wide-format CSV.

    Returns: (week_list, region_data)
      - week_list: [(year, week), ...] in order
      - region_data: {region: [values for each week]}
    """
    with open(filepath, 'r', encoding='iso-8859-1') as f:
        reader = csv.reader(f)
        header = next(reader)

        week_columns = []
        for i, col in enumerate(header):
            parsed = parse_week_column(col)
            if parsed:
                week_columns.append((i, parsed))

        week_columns.sort(key=lambda x: x[1])
        week_list = [wc[1] for wc in week_columns]
        col_indices = [wc[0] for wc in week_columns]

        region_data = {}
        for row in reader:
            if len(row) == 0:
                continue

            region = row[0].strip('"')

            if region not in SWEDISH_REGIONS:
                if region.strip():
                    print(f"WARNING: Unknown region '{region}' - skipping")
                continue

            values = []
            for col_idx in col_indices:
                if col_idx < len(row):
                    val = parse_number(row[col_idx])
                else:
                    val = 0.0
                values.append(val)

            region_data[region] = values

        # Verify all expected regions found
        found_regions = set(region_data.keys())
        expected_regions = set(SWEDISH_REGIONS)
        missing = expected_regions - found_regions
        if missing:
            print(f"WARNING: Missing regions in data: {missing}")
        extra = found_regions - expected_regions
        if extra:
            print(f"WARNING: Unexpected regions in data: {extra}")

    return week_list, region_data


def convert_to_matrix(week_list: list[tuple[int, int]],
                     region_data: dict[str, list[float]],
                     regions: list[str]) -> list[list[float]]:
    """
    Convert region_data to time series matrix.

    Returns: matrix[t][r] = cases for week t, region r
    """
    n_weeks = len(week_list)

    matrix = []
    for t in range(n_weeks):
        row = []
        for region in regions:
            if region in region_data and t < len(region_data[region]):
                val = region_data[region][t]
            else:
                val = 0.0
            row.append(val)
        matrix.append(row)

    return matrix


def write_matrix(matrix: list[list[float]], output_path: Path):
    """Write matrix to EpiGNN format (comma-separated, no header)."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in matrix:
            line = ','.join(f'{v:.1f}' for v in row)
            f.write(line + '\n')


def write_weeks(week_list: list[tuple[int, int]], output_path: Path):
    """Write week index for reference."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, (year, week) in enumerate(week_list):
            f.write(f'{i},{year},w{week:02d}\n')


def write_regions(regions: list[str], output_path: Path):
    """Write region index for reference."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, region in enumerate(regions):
            f.write(f'{i},{region}\n')


def main():
    main_repo = Path('C:/Users/Admin/Desktop/MEx')
    data_dir = main_repo / 'Data'
    swedish_path = data_dir / 'Swedish_Folkhalsomyndigheten.csv'

    if not swedish_path.exists():
        print(f"Error: File not found: {swedish_path}")
        sys.exit(1)

    print(f"Loading: {swedish_path.name}")

    output_dir = Path(__file__).parent

    week_list, region_data = load_swedish_data(swedish_path)
    print(f"Loaded {len(week_list)} weeks of data")
    print(f"Regions found: {len(region_data)}")
    print(f"Date range: {week_list[0][0]} w{week_list[0][1]:02d} to {week_list[-1][0]} w{week_list[-1][1]:02d}")

    matrix = convert_to_matrix(week_list, region_data, SWEDISH_REGIONS)
    print(f"Matrix shape: {len(matrix)} weeks x {len(SWEDISH_REGIONS)} regions")

    missing_count = sum(1 for row in matrix for val in row if val == 0.0)
    total_cells = len(matrix) * len(SWEDISH_REGIONS)
    print(f"Zero values: {missing_count}/{total_cells} ({100*missing_count/total_cells:.1f}%)")

    write_matrix(matrix, output_dir / 'sweden_flu_cases.txt')
    write_weeks(week_list, output_dir / 'sweden_week_index.csv')
    write_regions(SWEDISH_REGIONS, output_dir / 'sweden_region_index.csv')

    print(f"\nOutputs written to {output_dir}:")
    print("  - sweden_flu_cases.txt (model input)")
    print("  - sweden_week_index.csv (week reference)")
    print("  - sweden_region_index.csv (region order reference)")


if __name__ == '__main__':
    main()
