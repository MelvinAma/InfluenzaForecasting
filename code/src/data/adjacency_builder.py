#!/usr/bin/env python3
"""
Generate geographic adjacency matrices for US states and Swedish counties.

Adjacency is defined as sharing a land border.
Output format: N×N binary matrix (comma-separated, no header)
  - 1 = states/counties share a border
  - 0 = no shared border
  - Diagonal = 0 (state not adjacent to itself)
"""

from pathlib import Path
import sys

US_STATE_ADJACENCIES = {
    "AK": [],
    "AL": ["FL", "GA", "MS", "TN"],
    "AR": ["LA", "MO", "MS", "OK", "TN", "TX"],
    "AZ": ["CA", "CO", "NM", "NV", "UT"],
    "CA": ["AZ", "NV", "OR"],
    "CO": ["AZ", "KS", "NE", "NM", "OK", "UT", "WY"],
    "CT": ["MA", "NY", "RI"],
    "DC": ["MD", "VA"],
    "DE": ["MD", "NJ", "PA"],
    "FL": ["AL", "GA"],
    "GA": ["AL", "FL", "NC", "SC", "TN"],
    "HI": [],
    "IA": ["IL", "MN", "MO", "NE", "SD", "WI"],
    "ID": ["MT", "NV", "OR", "UT", "WA", "WY"],
    "IL": ["IA", "IN", "KY", "MO", "WI"],
    "IN": ["IL", "KY", "MI", "OH"],
    "KS": ["CO", "MO", "NE", "OK"],
    "KY": ["IL", "IN", "MO", "OH", "TN", "VA", "WV"],
    "LA": ["AR", "MS", "TX"],
    "MA": ["CT", "NH", "NY", "RI", "VT"],
    "MD": ["DC", "DE", "PA", "VA", "WV"],
    "ME": ["NH"],
    "MI": ["IN", "OH", "WI"],
    "MN": ["IA", "ND", "SD", "WI"],
    "MO": ["AR", "IA", "IL", "KS", "KY", "NE", "OK", "TN"],
    "MS": ["AL", "AR", "LA", "TN"],
    "MT": ["ID", "ND", "SD", "WY"],
    "NC": ["GA", "SC", "TN", "VA"],
    "ND": ["MN", "MT", "SD"],
    "NE": ["CO", "IA", "KS", "MO", "SD", "WY"],
    "NH": ["MA", "ME", "VT"],
    "NJ": ["DE", "NY", "PA"],
    "NM": ["AZ", "CO", "OK", "TX", "UT"],
    "NV": ["AZ", "CA", "ID", "OR", "UT"],
    "NY": ["CT", "MA", "NJ", "PA", "VT"],
    "OH": ["IN", "KY", "MI", "PA", "WV"],
    "OK": ["AR", "CO", "KS", "MO", "NM", "TX"],
    "OR": ["CA", "ID", "NV", "WA"],
    "PA": ["DE", "MD", "NJ", "NY", "OH", "WV"],
    "RI": ["CT", "MA"],
    "SC": ["GA", "NC"],
    "SD": ["IA", "MN", "MT", "ND", "NE", "WY"],
    "TN": ["AL", "AR", "GA", "KY", "MO", "MS", "NC", "VA"],
    "TX": ["AR", "LA", "NM", "OK"],
    "UT": ["AZ", "CO", "ID", "NM", "NV", "WY"],
    "VA": ["DC", "KY", "MD", "NC", "TN", "WV"],
    "VT": ["MA", "NH", "NY"],
    "WA": ["ID", "OR"],
    "WI": ["IA", "IL", "MI", "MN"],
    "WV": ["KY", "MD", "OH", "PA", "VA"],
    "WY": ["CO", "ID", "MT", "NE", "SD", "UT"],
}

SWEDISH_REGION_ADJACENCIES = {
    "Blekinge": ["Kalmar", "Kronoberg", "Skåne"],
    "Dalarna": ["Gävleborg", "Jämtland Härjedalen", "Västmanland", "Värmland"],
    "Gotland": [],
    "Gävleborg": ["Dalarna", "Jämtland Härjedalen", "Uppsala", "Västernorrland"],
    "Halland": ["Skåne", "Västra Götaland"],
    "Jämtland Härjedalen": ["Dalarna", "Gävleborg", "Västerbotten", "Västernorrland"],
    "Jönköping": ["Kalmar", "Kronoberg", "Västra Götaland", "Östergötland"],
    "Kalmar": ["Blekinge", "Jönköping", "Kronoberg", "Östergötland"],
    "Kronoberg": ["Blekinge", "Jönköping", "Kalmar", "Skåne", "Västra Götaland"],
    "Norrbotten": ["Västerbotten"],
    "Skåne": ["Blekinge", "Halland", "Kronoberg"],
    "Stockholm": ["Södermanland", "Uppsala"],
    "Södermanland": ["Stockholm", "Uppsala", "Västmanland", "Östergötland", "Örebro"],
    "Uppsala": ["Gävleborg", "Stockholm", "Södermanland", "Västmanland"],
    "Värmland": ["Dalarna", "Västra Götaland", "Örebro"],
    "Västerbotten": ["Jämtland Härjedalen", "Norrbotten", "Västernorrland"],
    "Västernorrland": ["Gävleborg", "Jämtland Härjedalen", "Västerbotten"],
    "Västmanland": ["Dalarna", "Södermanland", "Uppsala", "Örebro"],
    "Västra Götaland": ["Halland", "Jönköping", "Kronoberg", "Värmland", "Örebro", "Östergötland"],
    "Örebro": ["Södermanland", "Värmland", "Västmanland", "Västra Götaland", "Östergötland"],
    "Östergötland": ["Jönköping", "Kalmar", "Södermanland", "Västra Götaland", "Örebro"],
}


def build_adjacency_matrix(entities: list[str],
                          adjacency_dict: dict[str, list[str]]) -> list[list[int]]:
    """
    Build binary adjacency matrix from adjacency dictionary.

    Args:
        entities: Ordered list of states/regions
        adjacency_dict: Dict mapping each entity to list of adjacent entities

    Returns: N×N binary matrix
    """
    entity_set = set(entities)
    n = len(entities)
    matrix = [[0 for _ in range(n)] for _ in range(n)]

    for i, entity_i in enumerate(entities):
        neighbors = adjacency_dict.get(entity_i, [])

        # Validate neighbors exist in entities list
        for neighbor in neighbors:
            if neighbor not in entity_set:
                print(f"ERROR: {entity_i} references unknown neighbor '{neighbor}'")
                sys.exit(1)

        for j, entity_j in enumerate(entities):
            if entity_j in neighbors:
                matrix[i][j] = 1

    return matrix


def write_adjacency_matrix(matrix: list[list[int]], output_path: Path):
    """Write adjacency matrix (comma-separated, no header)."""
    with open(output_path, 'w') as f:
        for row in matrix:
            line = ','.join(str(v) for v in row)
            f.write(line + '\n')


def validate_symmetry(matrix: list[list[int]], entities: list[str]) -> bool:
    """Verify adjacency matrix is symmetric."""
    n = len(matrix)
    for i in range(n):
        for j in range(i+1, n):
            if matrix[i][j] != matrix[j][i]:
                print(f"WARNING: Asymmetric adjacency between {entities[i]} and {entities[j]}")
                return False
    return True


def main():
    output_dir = Path(__file__).parent

    us_states = [
        "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL",
        "GA", "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA",
        "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE",
        "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI",
        "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY"
    ]

    swedish_regions = [
        "Blekinge", "Dalarna", "Gotland", "Gävleborg", "Halland",
        "Jämtland Härjedalen", "Jönköping", "Kalmar", "Kronoberg",
        "Norrbotten", "Skåne", "Stockholm", "Södermanland", "Uppsala",
        "Värmland", "Västerbotten", "Västernorrland", "Västmanland",
        "Västra Götaland", "Örebro", "Östergötland"
    ]

    print("Generating US state adjacency matrix (51×51)...")
    us_matrix = build_adjacency_matrix(us_states, US_STATE_ADJACENCIES)

    if validate_symmetry(us_matrix, us_states):
        print("  [OK] Matrix is symmetric")
    else:
        print("  [ERROR] Matrix has asymmetries")
        sys.exit(1)

    us_edges = sum(sum(row) for row in us_matrix)
    print(f"  Total adjacencies: {us_edges} (avg {us_edges/51:.1f} per state)")

    us_output = output_dir / 'state-adj-51.txt'
    write_adjacency_matrix(us_matrix, us_output)
    print(f"  Written to: {us_output}")

    print("\nGenerating Swedish county adjacency matrix (21×21)...")
    swedish_matrix = build_adjacency_matrix(swedish_regions, SWEDISH_REGION_ADJACENCIES)

    if validate_symmetry(swedish_matrix, swedish_regions):
        print("  [OK] Matrix is symmetric")
    else:
        print("  [ERROR] Matrix has asymmetries")
        sys.exit(1)

    swedish_edges = sum(sum(row) for row in swedish_matrix)
    print(f"  Total adjacencies: {swedish_edges} (avg {swedish_edges/21:.1f} per region)")

    swedish_output = output_dir / 'sweden-adj-21.txt'
    write_adjacency_matrix(swedish_matrix, swedish_output)
    print(f"  Written to: {swedish_output}")

    print("\n[OK] All adjacency matrices generated successfully")


if __name__ == '__main__':
    main()
