import sys
import pytest
import numpy as np
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

US_EPIGNN_CFG  = str(PROJ_ROOT / "src" / "configs" / "us_epignn.json")
US_COLAGNN_CFG = str(PROJ_ROOT / "src" / "configs" / "us_colagnn.json")

FAST_CFG = {
    "lr": 1e-3, "k": 4, "hidP": 1, "hidA": 32, "hidR": 64, "n_hidden": 16,
    "n_layer": 1, "n": 1, "dropout": 0.2, "weight_decay": 5e-4,
    "batch_size": 32, "res": 0, "s": 2, "hw": 0,
}

FAST_COLA_CFG = {
    "lr": 1e-3, "k": 5, "hidsp": 8, "n_hidden": 16,
    "n_layer": 1, "dropout": 0.2, "weight_decay": 5e-4,
    "batch_size": 32, "bi": False, "rnn_model": "RNN",
}

EXPECTED_KEYS = {
    "mae", "rmse", "mape", "pcc",
    "mae_mean", "rmse_mean", "mape_mean", "pcc_mean",
    "rmse_global", "pcc_global",
    "peak_intensity_error", "peak_week_error", "onset_week_error",
    "n_weeks", "n_regions",
}


class TestOnsetWeek:
    def setup_method(self):
        from src.eval.evaluate import _onset_week
        self._onset_week = _onset_week

    def test_detects_onset_at_start(self):
        series = np.array([5.0, 6.0, 7.0, 4.0])
        assert self._onset_week(series, 4.0) == 0.0

    def test_detects_onset_midway(self):
        series = np.array([1.0, 1.0, 5.0, 5.0, 5.0])
        assert self._onset_week(series, 4.0) == 2.0

    def test_non_consecutive_does_not_qualify(self):
        series = np.array([5.0, 1.0, 5.0, 5.0, 5.0])
        assert self._onset_week(series, 4.0) == 2.0

    def test_returns_nan_when_never_detected(self):
        series = np.array([1.0, 5.0, 5.0, 1.0])
        assert np.isnan(self._onset_week(series, 4.0))

    def test_series_too_short_returns_nan(self):
        series = np.array([5.0, 5.0])
        assert np.isnan(self._onset_week(series, 4.0))

    def test_returns_first_not_last(self):
        series = np.array([5.0, 5.0, 5.0, 1.0, 5.0, 5.0, 5.0])
        assert self._onset_week(series, 4.0) == 0.0


class TestComputeMetrics:
    def setup_method(self):
        from src.eval.evaluate import compute_metrics
        self._compute_metrics = compute_metrics
        rng = np.random.default_rng(0)
        self.T, self.m = 30, 5
        self.true = rng.uniform(100, 2000, size=(self.T, self.m)).astype(np.float32)
        self.pred = self.true + rng.normal(0, 50, size=(self.T, self.m)).astype(np.float32)

    def test_returns_all_expected_keys(self):
        result = self._compute_metrics(self.pred, self.true)
        assert EXPECTED_KEYS == set(result.keys())

    def test_mae_nonnegative(self):
        result = self._compute_metrics(self.pred, self.true)
        assert np.all(result["mae"] >= 0)

    def test_rmse_geq_mae(self):
        result = self._compute_metrics(self.pred, self.true)
        assert np.all(result["rmse"] >= result["mae"] - 1e-6)

    def test_perfect_predictions_zero_mae(self):
        result = self._compute_metrics(self.true, self.true)
        assert result["mae_mean"] < 1e-6

    def test_mae_shape(self):
        result = self._compute_metrics(self.pred, self.true)
        assert result["mae"].shape == (self.m,)

    def test_national_aggregates_are_scalars(self):
        result = self._compute_metrics(self.pred, self.true)
        for key in ("mae_mean", "rmse_mean", "mape_mean"):
            assert isinstance(result[key], float)

    def test_mape_handles_zero_true_values(self):
        true = np.zeros((10, 3))
        pred = np.ones((10, 3))
        result = self._compute_metrics(pred, true)
        assert not np.any(np.isinf(result["mape"]))

    def test_onset_none_by_default(self):
        result = self._compute_metrics(self.pred, self.true)
        assert result["onset_week_error"] is None

    def test_onset_error_with_threshold(self):
        result = self._compute_metrics(self.pred, self.true, onset_threshold=200.0)
        assert result["onset_week_error"] is not None
        assert result["onset_week_error"].shape == (self.m,)

    def test_peak_week_error_shape(self):
        result = self._compute_metrics(self.pred, self.true)
        assert result["peak_week_error"].shape == (self.m,)

    def test_n_weeks_n_regions(self):
        result = self._compute_metrics(self.pred, self.true)
        assert result["n_weeks"] == self.T
        assert result["n_regions"] == self.m

    def test_known_mae(self):
        true = np.ones((4, 2)) * 100.0
        pred = np.ones((4, 2)) * 110.0
        result = self._compute_metrics(pred, true)
        assert abs(result["mae_mean"] - 10.0) < 1e-4

    def test_known_rmse(self):
        true = np.ones((4, 1)) * 0.0
        pred = np.array([[3.0], [4.0], [0.0], [0.0]])
        result = self._compute_metrics(pred, true)
        expected_rmse = np.sqrt((9 + 16) / 4)
        assert abs(result["rmse"][0] - expected_rmse) < 1e-4

    def test_pcc_shape(self):
        result = self._compute_metrics(self.pred, self.true)
        assert result["pcc"].shape == (self.m,)

    def test_pcc_mean_is_scalar(self):
        result = self._compute_metrics(self.pred, self.true)
        assert isinstance(result["pcc_mean"], float)

    def test_pcc_range(self):
        result = self._compute_metrics(self.pred, self.true)
        assert np.all(result["pcc"] >= -1.0 - 1e-6)
        assert np.all(result["pcc"] <= 1.0 + 1e-6)

    def test_pcc_perfect_predictions(self):
        result = self._compute_metrics(self.true, self.true)
        assert abs(result["pcc_mean"] - 1.0) < 1e-6

    def test_pcc_known_value(self):
        rng = np.random.default_rng(42)
        t = rng.uniform(100, 1000, size=(20, 1)).astype(np.float64)
        p = t * 1.5  # perfect linear scaling -- PCC should be 1.0
        result = self._compute_metrics(p.astype(np.float32), t.astype(np.float32))
        assert abs(result["pcc"][0] - 1.0) < 1e-5

    def test_rmse_global_nonneg(self):
        result = self._compute_metrics(self.pred, self.true)
        assert result["rmse_global"] >= 0.0

    def test_pcc_global_range(self):
        result = self._compute_metrics(self.pred, self.true)
        assert -1.0 - 1e-6 <= result["pcc_global"] <= 1.0 + 1e-6


class TestEpiGNNScoring:
    def test_scoring_returns_correct_keys(self):
        from src.tuning.runner import train_epignn
        from src.eval.evaluate import evaluate_epignn
        history = train_epignn(FAST_CFG, US_EPIGNN_CFG, num_epochs=1, patience=999)
        result = evaluate_epignn(history["best_state"], FAST_CFG, US_EPIGNN_CFG)
        assert EXPECTED_KEYS == set(result.keys())

    def test_mae_nonneg(self):
        from src.tuning.runner import train_epignn
        from src.eval.evaluate import evaluate_epignn
        history = train_epignn(FAST_CFG, US_EPIGNN_CFG, num_epochs=1, patience=999)
        result = evaluate_epignn(history["best_state"], FAST_CFG, US_EPIGNN_CFG)
        assert result["mae_mean"] >= 0.0

    def test_rmse_geq_mae(self):
        from src.tuning.runner import train_epignn
        from src.eval.evaluate import evaluate_epignn
        history = train_epignn(FAST_CFG, US_EPIGNN_CFG, num_epochs=1, patience=999)
        result = evaluate_epignn(history["best_state"], FAST_CFG, US_EPIGNN_CFG)
        assert result["rmse_mean"] >= result["mae_mean"] - 1e-4

    def test_onset_threshold_populates_onset_error(self):
        from src.tuning.runner import train_epignn
        from src.eval.evaluate import evaluate_epignn
        history = train_epignn(FAST_CFG, US_EPIGNN_CFG, num_epochs=1, patience=999)
        result = evaluate_epignn(
            history["best_state"], FAST_CFG, US_EPIGNN_CFG, onset_threshold=1600.0
        )
        assert result["onset_week_error"] is not None

    def test_n_regions_matches_us_dataset(self):
        from src.tuning.runner import train_epignn
        from src.eval.evaluate import evaluate_epignn
        history = train_epignn(FAST_CFG, US_EPIGNN_CFG, num_epochs=1, patience=999)
        result = evaluate_epignn(history["best_state"], FAST_CFG, US_EPIGNN_CFG)
        assert result["n_regions"] == 51


class TestColaGNNScoring:
    def test_scoring_returns_correct_keys(self):
        from src.tuning.runner import train_colagnn
        from src.eval.evaluate import evaluate_colagnn
        history = train_colagnn(FAST_COLA_CFG, US_COLAGNN_CFG, num_epochs=1, patience=999)
        result = evaluate_colagnn(history["best_state"], FAST_COLA_CFG, US_COLAGNN_CFG)
        assert EXPECTED_KEYS == set(result.keys())

    def test_mae_nonneg(self):
        from src.tuning.runner import train_colagnn
        from src.eval.evaluate import evaluate_colagnn
        history = train_colagnn(FAST_COLA_CFG, US_COLAGNN_CFG, num_epochs=1, patience=999)
        result = evaluate_colagnn(history["best_state"], FAST_COLA_CFG, US_COLAGNN_CFG)
        assert result["mae_mean"] >= 0.0

    def test_n_regions_matches_us_dataset(self):
        from src.tuning.runner import train_colagnn
        from src.eval.evaluate import evaluate_colagnn
        history = train_colagnn(FAST_COLA_CFG, US_COLAGNN_CFG, num_epochs=1, patience=999)
        result = evaluate_colagnn(history["best_state"], FAST_COLA_CFG, US_COLAGNN_CFG)
        assert result["n_regions"] == 51


class TestInferenceEpiGNN:
    def setup_method(self):
        from src.tuning.runner import train_epignn
        self._history = train_epignn(FAST_CFG, US_EPIGNN_CFG, num_epochs=1, patience=999)

    def test_inference_test_split_shapes(self):
        from src.eval.evaluate import inference_epignn
        pred, true = inference_epignn(self._history["best_state"], FAST_CFG, US_EPIGNN_CFG, split="test")
        assert pred.ndim == 2 and true.ndim == 2
        assert pred.shape == true.shape
        assert pred.shape[1] == 51

    def test_inference_val_split_shapes(self):
        from src.eval.evaluate import inference_epignn
        pred, true = inference_epignn(self._history["best_state"], FAST_CFG, US_EPIGNN_CFG, split="val")
        assert pred.ndim == 2 and true.ndim == 2
        assert pred.shape == true.shape
        assert pred.shape[1] == 51

    def test_val_and_test_return_different_data(self):
        from src.eval.evaluate import inference_epignn
        import numpy as np
        _, t_test = inference_epignn(self._history["best_state"], FAST_CFG, US_EPIGNN_CFG, split="test")
        _, t_val = inference_epignn(self._history["best_state"], FAST_CFG, US_EPIGNN_CFG, split="val")
        assert not np.array_equal(t_test, t_val)


class TestInferenceColaGNN:
    def setup_method(self):
        from src.tuning.runner import train_colagnn
        self._history = train_colagnn(FAST_COLA_CFG, US_COLAGNN_CFG, num_epochs=1, patience=999)

    def test_inference_test_split_shapes(self):
        from src.eval.evaluate import inference_colagnn
        pred, true = inference_colagnn(self._history["best_state"], FAST_COLA_CFG, US_COLAGNN_CFG, split="test")
        assert pred.ndim == 2 and true.ndim == 2
        assert pred.shape == true.shape
        assert pred.shape[1] == 51

    def test_inference_val_split_shapes(self):
        from src.eval.evaluate import inference_colagnn
        pred, true = inference_colagnn(self._history["best_state"], FAST_COLA_CFG, US_COLAGNN_CFG, split="val")
        assert pred.ndim == 2 and true.ndim == 2
        assert pred.shape == true.shape
        assert pred.shape[1] == 51


class TestComputeOnsetThreshold:
    def test_returns_correct_shape(self):
        from src.data.loader import FluDataLoader
        import json
        with open(US_EPIGNN_CFG) as f:
            cfg = json.load(f)
        loader = FluDataLoader(cfg)
        from src.eval.evaluate import compute_onset_threshold
        thresholds = compute_onset_threshold(loader, cfg)
        assert thresholds.shape == (51,)
        assert np.all(np.isfinite(thresholds))
        assert np.all(thresholds >= 0)


class TestAvailableSeasons:
    def test_returns_sorted_list(self):
        from src.eval.cross_validate import _available_seasons
        seasons = _available_seasons(US_EPIGNN_CFG)
        years = [int(s.split("/")[0]) for s in seasons]
        assert years == sorted(years)

    def test_no_none_entries(self):
        from src.eval.cross_validate import _available_seasons
        seasons = _available_seasons(US_EPIGNN_CFG)
        assert all(s is not None for s in seasons)

    def test_sweden_excludes_covid_season(self):
        from src.eval.cross_validate import _available_seasons
        se_cfg = str(PROJ_ROOT / "src" / "configs" / "sweden_epignn.json")
        seasons = _available_seasons(se_cfg)
        assert "2020/21" not in seasons

    def test_us_has_at_least_four_seasons(self):
        from src.eval.cross_validate import _available_seasons
        seasons = _available_seasons(US_EPIGNN_CFG)
        assert len(seasons) >= 4

    def test_season_format(self):
        from src.eval.cross_validate import _available_seasons
        seasons = _available_seasons(US_EPIGNN_CFG)
        for s in seasons:
            parts = s.split("/")
            assert len(parts) == 2
            assert len(parts[0]) == 4 and parts[0].isdigit()
            assert len(parts[1]) == 2 and parts[1].isdigit()


class TestCrossValidate:
    def test_cv_epignn_returns_folds_and_aggregate(self):
        from src.eval.cross_validate import cv_epignn
        result = cv_epignn(
            US_EPIGNN_CFG, FAST_CFG,
            min_train_seasons=1, num_epochs=1, patience=999,
        )
        assert "folds" in result and "aggregate" in result
        assert len(result["folds"]) >= 1

    def test_cv_epignn_fold_has_test_season_key(self):
        from src.eval.cross_validate import cv_epignn
        result = cv_epignn(
            US_EPIGNN_CFG, FAST_CFG,
            min_train_seasons=1, num_epochs=1, patience=999,
        )
        for fold in result["folds"]:
            assert "test_season" in fold
            assert "val_season" in fold
            assert "metrics" in fold

    def test_cv_epignn_aggregate_keys(self):
        from src.eval.cross_validate import cv_epignn
        result = cv_epignn(
            US_EPIGNN_CFG, FAST_CFG,
            min_train_seasons=1, num_epochs=1, patience=999,
        )
        agg = result["aggregate"]
        for key in ("mae_mean", "mae_std", "rmse_mean", "rmse_std",
                    "mape_mean", "mape_std", "peak_intensity_mae",
                    "peak_week_mae", "onset_week_mae", "n_folds",
                    "n_seasons", "n_runs"):
            assert key in agg

    def test_cv_epignn_mae_nonneg(self):
        from src.eval.cross_validate import cv_epignn
        result = cv_epignn(
            US_EPIGNN_CFG, FAST_CFG,
            min_train_seasons=1, num_epochs=1, patience=999,
        )
        assert result["aggregate"]["mae_mean"] >= 0.0

    def test_cv_colagnn_returns_structure(self):
        from src.eval.cross_validate import cv_colagnn
        result = cv_colagnn(
            US_COLAGNN_CFG, FAST_COLA_CFG,
            min_train_seasons=1, num_epochs=1, patience=999,
        )
        assert "folds" in result and "aggregate" in result
        assert result["aggregate"]["mae_mean"] >= 0.0

    def test_cv_epignn_splits_are_valid(self):
        from src.eval.cross_validate import cv_epignn, _available_seasons
        result = cv_epignn(
            US_EPIGNN_CFG, FAST_CFG,
            min_train_seasons=1, num_epochs=1, patience=999,
        )
        seasons = _available_seasons(US_EPIGNN_CFG)
        for fold in result["folds"]:
            ts, vs = fold["test_season"], fold["val_season"]
            assert ts != vs, f"test and val season identical: {ts}"
            ts_year = int(ts.split("/")[0])
            vs_year = int(vs.split("/")[0])
            assert vs_year < ts_year, f"val {vs} not before test {ts}"
            assert ts in seasons, f"test season {ts} not in available seasons"
            assert vs in seasons, f"val season {vs} not in available seasons"

    def test_cv_colagnn_splits_are_valid(self):
        from src.eval.cross_validate import cv_colagnn, _available_seasons
        result = cv_colagnn(
            US_COLAGNN_CFG, FAST_COLA_CFG,
            min_train_seasons=1, num_epochs=1, patience=999,
        )
        seasons = _available_seasons(US_COLAGNN_CFG)
        for fold in result["folds"]:
            ts, vs = fold["test_season"], fold["val_season"]
            assert ts != vs
            assert int(vs.split("/")[0]) < int(ts.split("/")[0])
            assert ts in seasons
            assert vs in seasons

    def test_cv_raises_when_too_few_seasons(self):
        from src.eval.cross_validate import cv_epignn
        with pytest.raises(ValueError, match="Not enough seasons"):
            cv_epignn(
                US_EPIGNN_CFG, FAST_CFG,
                min_train_seasons=100, num_epochs=1, patience=999,
            )


class TestAggregateMultiSeed:
    def _make_fold(self, mae_mean, rmse_mean, pcc_mean, rng):
        T, m = 10, 3
        true = rng.uniform(100, 500, (T, m)).astype(np.float32)
        pred = true + rng.normal(0, mae_mean, (T, m)).astype(np.float32)
        from src.eval.evaluate import compute_metrics
        metrics = compute_metrics(pred, true)
        metrics["mae_mean"] = float(mae_mean)
        metrics["rmse_mean"] = float(rmse_mean)
        metrics["pcc_mean"] = float(pcc_mean)
        metrics["mape_mean"] = 10.0
        return metrics

    def test_multi_seed_keys(self):
        from src.eval.cross_validate import _aggregate
        rng = np.random.default_rng(99)
        folds = [self._make_fold(10, 15, 0.9, rng) for _ in range(4)]
        seasons = ["2022/23", "2022/23", "2023/24", "2023/24"]
        agg = _aggregate(folds, None, fold_seasons=seasons)
        for key in ("mae_seed_std", "rmse_seed_std", "pcc_seed_std",
                     "n_seasons", "n_runs", "n_seeds"):
            assert key in agg, f"missing key: {key}"

    def test_multi_seed_counts(self):
        from src.eval.cross_validate import _aggregate
        rng = np.random.default_rng(99)
        folds = [self._make_fold(10, 15, 0.9, rng) for _ in range(6)]
        seasons = ["2021/22"] * 3 + ["2022/23"] * 3
        agg = _aggregate(folds, None, fold_seasons=seasons)
        assert agg["n_seasons"] == 2
        assert agg["n_runs"] == 6
        assert agg["n_seeds"] == 3
        assert agg["n_folds"] == 6

    def test_season_std_vs_seed_std(self):
        from src.eval.cross_validate import _aggregate
        rng = np.random.default_rng(42)
        folds = []
        seasons = []
        for base in (10.0, 50.0):
            for seed_offset in (0.0, 1.0):
                folds.append(self._make_fold(
                    base + seed_offset, base * 1.5 + seed_offset, 0.9, rng))
                seasons.append(f"{2020 + int(base) // 10}/23")
        agg = _aggregate(folds, None, fold_seasons=seasons)
        assert agg["mae_std"] > agg["mae_seed_std"]

    def test_single_seed_no_seed_std(self):
        from src.eval.cross_validate import _aggregate
        rng = np.random.default_rng(7)
        folds = [self._make_fold(10, 15, 0.9, rng) for _ in range(3)]
        seasons = ["2021/22", "2022/23", "2023/24"]
        agg = _aggregate(folds, None, fold_seasons=seasons)
        assert "mae_seed_std" not in agg
        assert agg["n_seasons"] == 3
        assert agg["n_runs"] == 3


class TestBaselines:
    def test_persistence_returns_correct_shapes(self):
        from src.eval.baselines import persistence_forecast
        pred, true = persistence_forecast(US_EPIGNN_CFG)
        assert pred.shape == true.shape
        assert pred.ndim == 2
        assert pred.shape[1] == 51

    def test_persistence_pred_is_lagged_true(self):
        from src.eval.baselines import persistence_forecast
        pred, true = persistence_forecast(US_EPIGNN_CFG)
        assert pred.shape[0] == true.shape[0]

    def test_persistence_values_nonneg(self):
        from src.eval.baselines import persistence_forecast
        pred, true = persistence_forecast(US_EPIGNN_CFG)
        assert np.all(pred >= 0)
        assert np.all(true >= 0)

    def test_seasonal_naive_returns_correct_shapes(self):
        from src.eval.baselines import seasonal_naive_forecast
        pred, true = seasonal_naive_forecast(US_EPIGNN_CFG, season_length=52)
        assert pred.shape == true.shape
        assert pred.ndim == 2
        assert pred.shape[1] == 51

    def test_seasonal_naive_shorter_than_full_test(self):
        from src.eval.baselines import persistence_forecast, seasonal_naive_forecast
        _, true_p = persistence_forecast(US_EPIGNN_CFG)
        pred_s, true_s = seasonal_naive_forecast(US_EPIGNN_CFG, season_length=52)
        assert pred_s.shape[0] <= true_p.shape[0]

    def test_cv_persistence_returns_structure(self):
        from src.eval.baselines import cv_persistence
        result = cv_persistence(US_EPIGNN_CFG, min_train_seasons=1)
        assert "folds" in result and "aggregate" in result
        assert result["aggregate"]["mae_mean"] >= 0.0

    def test_cv_seasonal_naive_returns_structure(self):
        from src.eval.baselines import cv_seasonal_naive
        result = cv_seasonal_naive(US_EPIGNN_CFG, min_train_seasons=1)
        assert "folds" in result and "aggregate" in result
        assert result["aggregate"]["mae_mean"] >= 0.0

    def test_baselines_compatible_with_compute_metrics(self):
        from src.eval.baselines import persistence_forecast
        from src.eval.evaluate import compute_metrics
        pred, true = persistence_forecast(US_EPIGNN_CFG)
        metrics = compute_metrics(pred, true)
        assert metrics["n_regions"] == 51


class TestReport:
    def setup_method(self):
        rng = np.random.default_rng(1)
        T, m = 20, 5
        true = rng.uniform(100, 2000, (T, m)).astype(np.float32)
        pred = true + rng.normal(0, 100, (T, m)).astype(np.float32)
        from src.eval.evaluate import compute_metrics
        self._metrics = compute_metrics(pred, true)

    def _make_cv_result(self):
        rng = np.random.default_rng(2)
        folds = []
        from src.eval.evaluate import compute_metrics
        for season in ["2022/23", "2023/24"]:
            T, m = 20, 5
            true = rng.uniform(100, 2000, (T, m)).astype(np.float32)
            pred = true + rng.normal(0, 100, (T, m)).astype(np.float32)
            folds.append({"test_season": season, "val_season": "prev",
                           "metrics": compute_metrics(pred, true)})
        from src.eval.cross_validate import _aggregate
        return {"folds": folds, "aggregate": _aggregate(
            [f["metrics"] for f in folds], None,
            fold_seasons=[f["test_season"] for f in folds])}

    def test_summary_table_returns_dataframe(self):
        from src.eval.report import summary_table
        result = self._make_cv_result()
        df = summary_table(result)
        import pandas as pd
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_fold_table_one_row_per_fold(self):
        from src.eval.report import fold_table
        result = self._make_cv_result()
        df = fold_table(result)
        assert len(df) == len(result["folds"])

    def test_region_table_correct_length(self):
        from src.eval.report import region_table
        df = region_table(self._metrics)
        assert len(df) == self._metrics["n_regions"]

    def test_region_table_with_names(self):
        from src.eval.report import region_table
        names = [f"R{i}" for i in range(self._metrics["n_regions"])]
        df = region_table(self._metrics, region_names=names)
        assert list(df["Region"]) == names

    def test_to_latex_returns_string(self):
        from src.eval.report import summary_table, to_latex
        result = self._make_cv_result()
        df = summary_table(result)
        latex = to_latex(df, caption="Test table", label="test")
        assert isinstance(latex, str)
        assert r"\begin{table}" in latex or r"\begin{tabular}" in latex


class TestFluSightOverlapRelWIS:
    def test_baseline_denominator_differs_by_overlap(self, tmp_path):
        """Regression: baseline WIS must be recomputed on each model's overlap.

        Synthetic hub where baseline covers 6 weeks but the model covers only
        the last 3.  Ground truth differs between halves so baseline WIS on the
        full window != baseline WIS on the overlap.  With flat quantiles
        (all levels = point prediction) WIS = |y - pred|, giving exact expected
        values.
        """
        import pandas as pd
        from src.eval.flusight_hub import evaluate_all_hub_models
        from src.eval.quantile import FLUSIGHT_LEVELS

        states = ["CA", "NY"]
        fips = {"CA": "06", "NY": "36"}
        target_dates = pd.date_range("2024-01-06", periods=6, freq="7D")

        gt = np.empty((6, 2))
        gt[:3] = 10.0
        gt[3:] = 100.0

        hub = tmp_path / "hub"
        bl_dir = hub / "model-output" / "FluSight-baseline"
        model_dir = hub / "model-output" / "TestModel"
        bl_dir.mkdir(parents=True)
        model_dir.mkdir(parents=True)

        def _write_flat(directory, name, ref_str, pred_val):
            rows = []
            for fip in fips.values():
                for lvl in FLUSIGHT_LEVELS:
                    rows.append({
                        "horizon": 1,
                        "location": fip,
                        "output_type": "quantile",
                        "output_type_id": str(lvl),
                        "value": pred_val,
                    })
            pd.DataFrame(rows).to_csv(
                directory / f"{ref_str}-{name}.csv", index=False,
            )

        for td in target_dates:
            ref = (td - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
            _write_flat(bl_dir, "FluSight-baseline", ref, 50.0)

        for td in target_dates[3:]:
            ref = (td - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
            _write_flat(model_dir, "TestModel", ref, 80.0)

        df = evaluate_all_hub_models(
            hub_dir=hub, target_dates=target_dates, gt=gt,
            states=states, horizons=[1], min_coverage=0.0,
        )

        model_rel = df.loc[df["Model"] == "TestModel", "Rel. WIS"].values[0]
        bl_rel = df.loc[df["Model"] == "FluSight-baseline", "Rel. WIS"].values[0]

        assert bl_rel == pytest.approx(1.0, abs=0.001)

        # flat quantiles => WIS = |y - pred|
        # model (pred=80) weeks 4-6 (gt=100): wis_per_region = 20
        # baseline (pred=50) weeks 4-6 only:  wis_per_region = 50
        # correct overlap Rel.WIS  = 20/50 = 0.400
        # old-bug full-window      = 20/45 = 0.444
        assert model_rel == pytest.approx(0.400, abs=0.002)
        assert model_rel != pytest.approx(0.444, abs=0.01)
