import json
from pathlib import Path
import sys

import pytest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import (
    FluDataLoader,
    US_CONFIG,
    SWEDEN_CONFIG,
    assign_seasons,
)
from src.eval.cross_validate import _write_fold_config

US_DATE_INDEX = ROOT / "src" / "data" / "date_index.csv"
SWEDEN_WEEK_INDEX = ROOT / "src" / "data" / "sweden_week_index.csv"


def _line_count(path):
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _season_start_year(season):
    return int(season.split("/")[0])


@pytest.fixture
def us_loader():
    return FluDataLoader(US_CONFIG)


@pytest.fixture
def sweden_loader():
    return FluDataLoader(SWEDEN_CONFIG)


@pytest.fixture
def us_ratio_loader():
    cfg = {**US_CONFIG, "split_mode": "ratio"}
    return FluDataLoader(cfg)


class TestUSData:

    def test_shape(self, us_loader):
        assert us_loader.n == _line_count(US_DATE_INDEX)
        assert us_loader.m == 51

    def test_adjacency_shape(self, us_loader):
        assert us_loader.adj.shape == (51, 51)
        assert us_loader.orig_adj.shape == (51, 51)

    def test_degree_adj(self, us_loader):
        expected = torch.sum(us_loader.orig_adj, dim=-1)
        assert torch.allclose(us_loader.degree_adj, expected)

    def test_sliding_window_shapes(self, us_loader):
        X_train, Y_train = us_loader.train
        assert X_train.ndim == 3
        assert X_train.shape[1] == US_CONFIG["window"]
        assert X_train.shape[2] == 51
        assert Y_train.ndim == 2
        assert Y_train.shape[1] == 51
        assert X_train.shape[0] == Y_train.shape[0]


class TestSwedenData:

    def test_shape(self, sweden_loader):
        assert sweden_loader.n == _line_count(SWEDEN_WEEK_INDEX)
        assert sweden_loader.m == 21

    def test_adjacency_shape(self, sweden_loader):
        assert sweden_loader.adj.shape == (21, 21)

    def test_excludes_covid(self, sweden_loader):
        for i in sweden_loader.train_set:
            assert sweden_loader.seasons[i] != "2020/21"
        for i in sweden_loader.val_set:
            assert sweden_loader.seasons[i] != "2020/21"
        for i in sweden_loader.test_set:
            assert sweden_loader.seasons[i] != "2020/21"

    def test_excludes_future_partial_season(self, sweden_loader):
        all_idx = sweden_loader.train_set + sweden_loader.val_set + sweden_loader.test_set
        for i in all_idx:
            assert sweden_loader.seasons[i] != "2025/26"


class TestSeasonSplitting:

    def test_no_overlap(self, us_loader):
        train = set(us_loader.train_set)
        val = set(us_loader.val_set)
        test = set(us_loader.test_set)
        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)

    def test_test_season_in_test_set(self, us_loader):
        for i in us_loader.test_set:
            assert us_loader.seasons[i] == "2024/25"

    def test_val_season_in_val_set(self, us_loader):
        for i in us_loader.val_set:
            assert us_loader.seasons[i] == "2023/24"

    def test_train_excludes_test_val(self, us_loader):
        for i in us_loader.train_set:
            assert us_loader.seasons[i] not in ("2023/24", "2024/25")

    def test_train_seasons_precede_validation(self, us_loader):
        train_years = {
            _season_start_year(us_loader.seasons[i])
            for i in us_loader.train_set
            if us_loader.seasons[i] is not None
        }
        assert train_years
        assert max(train_years) < _season_start_year("2023/24")


class TestRatioSplitting:

    def test_covers_all_valid(self, us_ratio_loader):
        all_idx = (
            us_ratio_loader.train_set
            + us_ratio_loader.val_set
            + us_ratio_loader.test_set
        )
        min_idx = US_CONFIG["window"] + US_CONFIG["horizon"] - 1
        expected = list(range(min_idx, _line_count(US_DATE_INDEX)))
        assert sorted(all_idx) == expected

    def test_ordered(self, us_ratio_loader):
        combined = (
            us_ratio_loader.train_set
            + us_ratio_loader.val_set
            + us_ratio_loader.test_set
        )
        assert combined == sorted(combined)


class TestNormalization:

    def test_training_target_range(self, us_loader):
        _, Y_train = us_loader.train
        vals = Y_train.numpy()
        assert np.all(vals >= -0.01)
        assert np.all(vals <= 1.01)

    def test_min_max_from_train(self, us_loader):
        assert us_loader.max.shape == (51,)
        assert us_loader.min.shape == (51,)

    def test_peak_thold_shape(self, us_loader):
        assert us_loader.peak_thold.shape == (51,)


    def test_normalization_uses_first_window_and_targets(self, us_loader):
        """Pin: min/max derived from first training window + all training
        targets (P + n rows), not from all training windows (n*P rows).
        This is an inherited quirk preserved for reproducibility."""
        raw = us_loader._batchify(us_loader.train_set, use_raw=True)
        train_mx = torch.cat((raw[0][0], raw[1]), 0).numpy()
        np.testing.assert_array_equal(us_loader.max, np.max(train_mx, 0))
        np.testing.assert_array_equal(us_loader.min, np.min(train_mx, 0))
        np.testing.assert_array_equal(us_loader.peak_thold, np.mean(train_mx, 0))


class TestGetBatches:

    def test_yields_three_elements(self, us_loader):
        for batch in us_loader.get_batches(us_loader.train, batch_size=32, shuffle=False):
            X, Y, idx = batch
            assert X.ndim == 3
            assert Y.ndim == 2
            assert idx.ndim == 1
            assert X.shape[0] == Y.shape[0] == idx.shape[0]
            break

    def test_covers_all_samples(self, us_loader):
        total = 0
        for X, Y, idx in us_loader.get_batches(us_loader.train, batch_size=64, shuffle=False):
            total += X.shape[0]
        assert total == len(us_loader.train[0])


class TestHorizon:

    def test_horizon_offset(self):
        cfg = {**US_CONFIG, "horizon": 3}
        loader = FluDataLoader(cfg)
        min_idx = cfg["window"] + cfg["horizon"] - 1
        for i in loader.train_set + loader.val_set + loader.test_set:
            assert i >= min_idx


class TestAssignSeasons:

    def test_season_w40_plus(self):
        result = assign_seasons([2023], [42])
        assert result == ["2023/24"]

    def test_season_w1_to_w20(self):
        result = assign_seasons([2024], [5])
        assert result == ["2023/24"]

    def test_off_season_returns_none(self):
        result = assign_seasons([2024], [30])
        assert result == [None]


class TestCrossValidationSplits:

    def _load_fold(self, base_config_name, test_season, val_season):
        config_path = ROOT / "src" / "configs" / base_config_name
        fold_cfg_path = _write_fold_config(str(config_path), test_season, val_season)
        try:
            with open(fold_cfg_path, "r", encoding="utf-8") as handle:
                cfg = json.load(handle)
            loader = FluDataLoader(cfg)
        finally:
            Path(fold_cfg_path).unlink(missing_ok=True)
        return loader, cfg

    def test_us_cv_fold_train_stops_before_validation(self):
        loader, cfg = self._load_fold("us_epignn.json", "2024/25", "2023/24")
        assert cfg["train_cutoff_row"] == min(loader.val_set)
        train_years = {
            _season_start_year(loader.seasons[i])
            for i in loader.train_set
            if loader.seasons[i] is not None
        }
        assert train_years
        assert max(train_years) < _season_start_year("2023/24")
        assert max(loader.train_set) < cfg["train_cutoff_row"]

    def test_sweden_cv_fold_train_stops_before_validation(self):
        loader, cfg = self._load_fold("sweden_epignn.json", "2018/19", "2017/18")
        assert cfg["train_cutoff_row"] == min(loader.val_set)
        train_years = {
            _season_start_year(loader.seasons[i])
            for i in loader.train_set
            if loader.seasons[i] is not None
        }
        assert train_years
        assert max(train_years) < _season_start_year("2017/18")
        assert max(loader.train_set) < cfg["train_cutoff_row"]
