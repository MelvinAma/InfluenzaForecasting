#!/usr/bin/env python3
"""
Validate generated data formats match EpiGNN/ColaGNN requirements.

Expected formats:
  - Time series: T×N matrix (T weeks, N regions), comma-separated floats
  - Adjacency: N×N binary matrix, comma-separated integers
  - Both: no header, no row indices
"""

from pathlib import Path
import sys


def load_matrix(filepath: Path) -> list[list[str]]:
    """Load matrix from CSV file (returns strings for validation)."""
    with open(filepath, 'r') as f:
        return [line.strip().split(',') for line in f if line.strip()]


def validate_time_series(filepath: Path, expected_regions: int) -> bool:
    """
    Validate time series matrix format.

    Args:
        filepath: Path to time series file
        expected_regions: Expected number of regions (columns)

    Returns: True if valid
    """
    print(f"\nValidating: {filepath.name}")
    print(f"  Expected format: T weeks × {expected_regions} regions")

    try:
        matrix = load_matrix(filepath)
        n_weeks = len(matrix)

        if n_weeks == 0:
            print("  [ERROR] File is empty")
            return False

        n_cols = len(matrix[0])
        print(f"  Actual shape: {n_weeks} × {n_cols}")

        if n_cols != expected_regions:
            print(f"  [ERROR] Expected {expected_regions} columns, got {n_cols}")
            return False

        col_counts = set(len(row) for row in matrix)
        if len(col_counts) > 1:
            print(f"  [ERROR] Inconsistent column counts: {col_counts}")
            return False

        for i, row in enumerate(matrix[:5]):
            for j, val in enumerate(row):
                try:
                    float(val)
                except ValueError:
                    print(f"  [ERROR] Non-numeric value at row {i}, col {j}: '{val}'")
                    return False

        min_val = min(float(val) for row in matrix for val in row)
        max_val = max(float(val) for row in matrix for val in row)
        print(f"  Value range: [{min_val:.1f}, {max_val:.1f}]")

        zero_count = sum(1 for row in matrix for val in row if float(val) == 0.0)
        total = n_weeks * n_cols
        print(f"  Zero values: {zero_count}/{total} ({100*zero_count/total:.1f}%)")

        print("  [OK] Format valid")
        return True

    except Exception as e:
        print(f"  [ERROR] {str(e)}")
        return False


def validate_adjacency(filepath: Path, expected_size: int) -> bool:
    """
    Validate adjacency matrix format.

    Args:
        filepath: Path to adjacency file
        expected_size: Expected N×N size

    Returns: True if valid
    """
    print(f"\nValidating: {filepath.name}")
    print(f"  Expected format: {expected_size} × {expected_size} binary matrix")

    try:
        matrix = load_matrix(filepath)
        n_rows = len(matrix)

        if n_rows != expected_size:
            print(f"  [ERROR] Expected {expected_size} rows, got {n_rows}")
            return False

        for i, row in enumerate(matrix):
            if len(row) != expected_size:
                print(f"  [ERROR] Row {i} has {len(row)} columns, expected {expected_size}")
                return False

        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                if val not in ['0', '1']:
                    print(f"  [ERROR] Non-binary value at ({i},{j}): '{val}'")
                    return False

        for i in range(n_rows):
            if matrix[i][i] != '0':
                print(f"  [ERROR] Diagonal element ({i},{i}) is not 0")
                return False

        for i in range(n_rows):
            for j in range(i+1, n_rows):
                if matrix[i][j] != matrix[j][i]:
                    print(f"  [ERROR] Matrix not symmetric at ({i},{j})")
                    return False

        edges = sum(int(val) for row in matrix for val in row)
        density = edges / (expected_size * expected_size) * 100
        print(f"  Edges: {edges} (density: {density:.1f}%)")
        print(f"  Average degree: {edges/expected_size:.1f}")

        print("  [OK] Format valid")
        return True

    except Exception as e:
        print(f"  [ERROR] {str(e)}")
        return False


def main():
    data_dir = Path(__file__).parent

    print("="*60)
    print("EpiGNN/ColaGNN Data Format Validation")
    print("="*60)

    all_valid = True

    all_valid &= validate_time_series(
        data_dir / 'state_flu_admissions.txt',
        expected_regions=51
    )

    all_valid &= validate_adjacency(
        data_dir / 'state-adj-51.txt',
        expected_size=51
    )

    all_valid &= validate_time_series(
        data_dir / 'sweden_flu_cases.txt',
        expected_regions=21
    )

    all_valid &= validate_adjacency(
        data_dir / 'sweden-adj-21.txt',
        expected_size=21
    )

    print("\n" + "="*60)
    if all_valid:
        print("[OK] All data formats valid for EpiGNN/ColaGNN")
        print("="*60)
        return 0
    else:
        print("[ERROR] Some validation checks failed")
        print("="*60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
