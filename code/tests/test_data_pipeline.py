#!/usr/bin/env python3
"""
Test suite for Phase 1: Data Pipeline

Tests all converters, adjacency builders, and format validators.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'data'))

from nhsn_converter import parse_number, US_STATES
from sweden_converter import parse_week_column, SWEDISH_REGIONS
from adjacency_builder import build_adjacency_matrix, validate_symmetry


class TestNHSNConverter:
    """Tests for US NHSN data converter."""

    def test_parse_number_valid(self):
        assert parse_number("100") == 100.0
        assert parse_number("1,234.5") == 1234.5
        assert parse_number('"567"') == 567.0

    def test_parse_number_empty(self):
        assert parse_number("") == 0.0
        assert parse_number("   ") == 0.0
        assert parse_number(None) == 0.0

    def test_parse_number_invalid(self):
        assert parse_number("N/A") == 0.0
        assert parse_number("abc") == 0.0

    def test_us_states_count(self):
        assert len(US_STATES) == 51

    def test_us_states_alphabetical(self):
        assert US_STATES == sorted(US_STATES)

    def test_us_states_includes_dc(self):
        assert "DC" in US_STATES

    def test_output_file_exists(self):
        output_dir = Path(__file__).parent.parent / 'src' / 'data'
        assert (output_dir / 'state_flu_admissions.txt').exists()
        assert (output_dir / 'state_index.csv').exists()
        assert (output_dir / 'date_index.csv').exists()


class TestSwedishConverter:
    """Tests for Swedish data converter."""

    def test_parse_week_column_valid(self):
        result = parse_week_column("Totalt influensa Antal fall 2015 v 40")
        assert result == (2015, 40)

        result = parse_week_column("Totalt influensa Antal fall 2026 v 03")
        assert result == (2026, 3)

    def test_parse_week_column_invalid(self):
        assert parse_week_column("Region") is None
        assert parse_week_column("Some other column") is None

    def test_swedish_regions_count(self):
        assert len(SWEDISH_REGIONS) == 21

    def test_swedish_regions_alphabetical(self):
        assert SWEDISH_REGIONS == sorted(SWEDISH_REGIONS)

    def test_swedish_regions_encoding(self):
        assert "Gävleborg" in SWEDISH_REGIONS
        assert "Jämtland Härjedalen" in SWEDISH_REGIONS
        assert "Örebro" in SWEDISH_REGIONS

    def test_output_file_exists(self):
        output_dir = Path(__file__).parent.parent / 'src' / 'data'
        assert (output_dir / 'sweden_flu_cases.txt').exists()
        assert (output_dir / 'sweden_region_index.csv').exists()
        assert (output_dir / 'sweden_week_index.csv').exists()


class TestAdjacencyBuilder:
    """Tests for adjacency matrix builder."""

    def test_build_adjacency_us(self):
        entities = ["CA", "NV", "OR"]
        adjacencies = {
            "CA": ["NV", "OR"],
            "NV": ["CA", "OR"],
            "OR": ["CA", "NV"]
        }
        matrix = build_adjacency_matrix(entities, adjacencies)

        assert len(matrix) == 3
        assert len(matrix[0]) == 3
        assert matrix[0][0] == 0
        assert matrix[0][1] == 1
        assert matrix[0][2] == 1

    def test_validate_symmetry_valid(self):
        matrix = [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ]
        entities = ["A", "B", "C"]
        assert validate_symmetry(matrix, entities) is True

    def test_validate_symmetry_invalid(self):
        matrix = [
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0]
        ]
        entities = ["A", "B", "C"]
        assert validate_symmetry(matrix, entities) is False

    def test_diagonal_is_zero(self):
        entities = ["CA", "NV", "OR"]
        adjacencies = {
            "CA": ["NV", "OR"],
            "NV": ["CA", "OR"],
            "OR": ["CA", "NV"]
        }
        matrix = build_adjacency_matrix(entities, adjacencies)

        for i in range(len(matrix)):
            assert matrix[i][i] == 0

    def test_us_adjacency_file_exists(self):
        output_dir = Path(__file__).parent.parent / 'src' / 'data'
        assert (output_dir / 'state-adj-51.txt').exists()

    def test_swedish_adjacency_file_exists(self):
        output_dir = Path(__file__).parent.parent / 'src' / 'data'
        assert (output_dir / 'sweden-adj-21.txt').exists()


class TestDataFormats:
    """Tests for data format validation."""

    def load_matrix(self, filepath):
        with open(filepath, 'r') as f:
            return [line.strip().split(',') for line in f if line.strip()]

    def count_rows(self, filepath):
        with open(filepath, 'r') as f:
            return sum(1 for line in f if line.strip())

    def test_us_time_series_dimensions(self):
        data_dir = Path(__file__).parent.parent / 'src' / 'data'
        matrix = self.load_matrix(data_dir / 'state_flu_admissions.txt')
        expected_rows = self.count_rows(data_dir / 'date_index.csv')

        assert len(matrix) == expected_rows
        assert len(matrix[0]) == 51

    def test_us_time_series_values(self):
        data_dir = Path(__file__).parent.parent / 'src' / 'data'
        matrix = self.load_matrix(data_dir / 'state_flu_admissions.txt')

        for row in matrix[:10]:
            for val in row:
                assert float(val) >= 0.0

    def test_us_time_series_consistent_columns(self):
        data_dir = Path(__file__).parent.parent / 'src' / 'data'
        matrix = self.load_matrix(data_dir / 'state_flu_admissions.txt')

        col_counts = set(len(row) for row in matrix)
        assert len(col_counts) == 1
        assert col_counts.pop() == 51

    def test_swedish_time_series_dimensions(self):
        data_dir = Path(__file__).parent.parent / 'src' / 'data'
        matrix = self.load_matrix(data_dir / 'sweden_flu_cases.txt')
        expected_rows = self.count_rows(data_dir / 'sweden_week_index.csv')

        assert len(matrix) == expected_rows
        assert len(matrix[0]) == 21

    def test_swedish_time_series_values(self):
        data_dir = Path(__file__).parent.parent / 'src' / 'data'
        matrix = self.load_matrix(data_dir / 'sweden_flu_cases.txt')

        for row in matrix[:10]:
            for val in row:
                assert float(val) >= 0.0

    def test_us_adjacency_dimensions(self):
        data_dir = Path(__file__).parent.parent / 'src' / 'data'
        matrix = self.load_matrix(data_dir / 'state-adj-51.txt')

        assert len(matrix) == 51
        for row in matrix:
            assert len(row) == 51

    def test_us_adjacency_binary(self):
        data_dir = Path(__file__).parent.parent / 'src' / 'data'
        matrix = self.load_matrix(data_dir / 'state-adj-51.txt')

        for row in matrix:
            for val in row:
                assert val in ['0', '1']

    def test_us_adjacency_symmetric(self):
        data_dir = Path(__file__).parent.parent / 'src' / 'data'
        matrix = self.load_matrix(data_dir / 'state-adj-51.txt')

        for i in range(len(matrix)):
            for j in range(i+1, len(matrix)):
                assert matrix[i][j] == matrix[j][i]

    def test_us_adjacency_zero_diagonal(self):
        data_dir = Path(__file__).parent.parent / 'src' / 'data'
        matrix = self.load_matrix(data_dir / 'state-adj-51.txt')

        for i in range(len(matrix)):
            assert matrix[i][i] == '0'

    def test_swedish_adjacency_dimensions(self):
        data_dir = Path(__file__).parent.parent / 'src' / 'data'
        matrix = self.load_matrix(data_dir / 'sweden-adj-21.txt')

        assert len(matrix) == 21
        for row in matrix:
            assert len(row) == 21

    def test_swedish_adjacency_binary(self):
        data_dir = Path(__file__).parent.parent / 'src' / 'data'
        matrix = self.load_matrix(data_dir / 'sweden-adj-21.txt')

        for row in matrix:
            for val in row:
                assert val in ['0', '1']

    def test_swedish_adjacency_symmetric(self):
        data_dir = Path(__file__).parent.parent / 'src' / 'data'
        matrix = self.load_matrix(data_dir / 'sweden-adj-21.txt')

        for i in range(len(matrix)):
            for j in range(i+1, len(matrix)):
                assert matrix[i][j] == matrix[j][i]

    def test_swedish_adjacency_zero_diagonal(self):
        data_dir = Path(__file__).parent.parent / 'src' / 'data'
        matrix = self.load_matrix(data_dir / 'sweden-adj-21.txt')

        for i in range(len(matrix)):
            assert matrix[i][i] == '0'


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_isolated_nodes_us(self):
        from adjacency_builder import US_STATE_ADJACENCIES
        assert US_STATE_ADJACENCIES["AK"] == []
        assert US_STATE_ADJACENCIES["HI"] == []

    def test_isolated_nodes_swedish(self):
        from adjacency_builder import SWEDISH_REGION_ADJACENCIES
        assert SWEDISH_REGION_ADJACENCIES["Gotland"] == []

    def test_us_adjacency_degree(self):
        data_dir = Path(__file__).parent.parent / 'src' / 'data'
        with open(data_dir / 'state-adj-51.txt', 'r') as f:
            matrix = [[int(x) for x in line.strip().split(',')] for line in f]

        degrees = [sum(row) for row in matrix]
        avg_degree = sum(degrees) / len(degrees)

        assert 3.0 <= avg_degree <= 6.0

    def test_swedish_adjacency_degree(self):
        data_dir = Path(__file__).parent.parent / 'src' / 'data'
        with open(data_dir / 'sweden-adj-21.txt', 'r') as f:
            matrix = [[int(x) for x in line.strip().split(',')] for line in f]

        degrees = [sum(row) for row in matrix]
        avg_degree = sum(degrees) / len(degrees)

        assert 2.5 <= avg_degree <= 5.0


class TestDataCorrectness:
    """Test actual data values match known geographic facts."""

    def test_ca_borders_nevada(self):
        from adjacency_builder import US_STATE_ADJACENCIES
        data_dir = Path(__file__).parent.parent / 'src' / 'data'
        with open(data_dir / 'state-adj-51.txt', 'r') as f:
            matrix = [[int(x) for x in line.strip().split(',')] for line in f]

        ca_idx = US_STATES.index("CA")
        nv_idx = US_STATES.index("NV")
        assert matrix[ca_idx][nv_idx] == 1
        assert matrix[nv_idx][ca_idx] == 1

    def test_ca_borders_oregon(self):
        from adjacency_builder import US_STATE_ADJACENCIES
        data_dir = Path(__file__).parent.parent / 'src' / 'data'
        with open(data_dir / 'state-adj-51.txt', 'r') as f:
            matrix = [[int(x) for x in line.strip().split(',')] for line in f]

        ca_idx = US_STATES.index("CA")
        or_idx = US_STATES.index("OR")
        assert matrix[ca_idx][or_idx] == 1
        assert matrix[or_idx][ca_idx] == 1

    def test_ca_borders_arizona(self):
        from adjacency_builder import US_STATE_ADJACENCIES
        data_dir = Path(__file__).parent.parent / 'src' / 'data'
        with open(data_dir / 'state-adj-51.txt', 'r') as f:
            matrix = [[int(x) for x in line.strip().split(',')] for line in f]

        ca_idx = US_STATES.index("CA")
        az_idx = US_STATES.index("AZ")
        assert matrix[ca_idx][az_idx] == 1
        assert matrix[az_idx][ca_idx] == 1

    def test_ca_does_not_border_texas(self):
        data_dir = Path(__file__).parent.parent / 'src' / 'data'
        with open(data_dir / 'state-adj-51.txt', 'r') as f:
            matrix = [[int(x) for x in line.strip().split(',')] for line in f]

        ca_idx = US_STATES.index("CA")
        tx_idx = US_STATES.index("TX")
        assert matrix[ca_idx][tx_idx] == 0
        assert matrix[tx_idx][ca_idx] == 0

    def test_ny_has_five_neighbors(self):
        data_dir = Path(__file__).parent.parent / 'src' / 'data'
        with open(data_dir / 'state-adj-51.txt', 'r') as f:
            matrix = [[int(x) for x in line.strip().split(',')] for line in f]

        ny_idx = US_STATES.index("NY")
        neighbors = sum(matrix[ny_idx])
        assert neighbors == 5

    def test_ny_neighbors_are_correct(self):
        data_dir = Path(__file__).parent.parent / 'src' / 'data'
        with open(data_dir / 'state-adj-51.txt', 'r') as f:
            matrix = [[int(x) for x in line.strip().split(',')] for line in f]

        ny_idx = US_STATES.index("NY")
        expected_neighbors = ["CT", "MA", "NJ", "PA", "VT"]

        for state in expected_neighbors:
            state_idx = US_STATES.index(state)
            assert matrix[ny_idx][state_idx] == 1

    def test_alaska_has_no_neighbors(self):
        data_dir = Path(__file__).parent.parent / 'src' / 'data'
        with open(data_dir / 'state-adj-51.txt', 'r') as f:
            matrix = [[int(x) for x in line.strip().split(',')] for line in f]

        ak_idx = US_STATES.index("AK")
        neighbors = sum(matrix[ak_idx])
        assert neighbors == 0

    def test_hawaii_has_no_neighbors(self):
        data_dir = Path(__file__).parent.parent / 'src' / 'data'
        with open(data_dir / 'state-adj-51.txt', 'r') as f:
            matrix = [[int(x) for x in line.strip().split(',')] for line in f]

        hi_idx = US_STATES.index("HI")
        neighbors = sum(matrix[hi_idx])
        assert neighbors == 0

    def test_gotland_has_no_neighbors(self):
        data_dir = Path(__file__).parent.parent / 'src' / 'data'
        with open(data_dir / 'sweden-adj-21.txt', 'r') as f:
            matrix = [[int(x) for x in line.strip().split(',')] for line in f]

        gotland_idx = SWEDISH_REGIONS.index("Gotland")
        neighbors = sum(matrix[gotland_idx])
        assert neighbors == 0

    def test_stockholm_borders_uppsala(self):
        data_dir = Path(__file__).parent.parent / 'src' / 'data'
        with open(data_dir / 'sweden-adj-21.txt', 'r') as f:
            matrix = [[int(x) for x in line.strip().split(',')] for line in f]

        stockholm_idx = SWEDISH_REGIONS.index("Stockholm")
        uppsala_idx = SWEDISH_REGIONS.index("Uppsala")
        assert matrix[stockholm_idx][uppsala_idx] == 1
        assert matrix[uppsala_idx][stockholm_idx] == 1
