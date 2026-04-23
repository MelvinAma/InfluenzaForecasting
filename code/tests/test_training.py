import sys
import pytest
import torch
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT))

US_EPIGNN_CFG  = str(PROJ_ROOT / "src" / "configs" / "us_epignn.json")
SE_EPIGNN_CFG  = str(PROJ_ROOT / "src" / "configs" / "sweden_epignn.json")
US_COLAGNN_CFG = str(PROJ_ROOT / "src" / "configs" / "us_colagnn.json")
SE_COLAGNN_CFG = str(PROJ_ROOT / "src" / "configs" / "sweden_colagnn.json")

FAST_CFG = {"lr": 1e-3, "k": 4, "hidP": 1, "hidA": 32, "hidR": 64, "n_hidden": 16,
            "n_layer": 1, "n": 1, "dropout": 0.2, "weight_decay": 5e-4,
            "batch_size": 32, "res": 0, "s": 2, "hw": 0}

FAST_COLA_CFG = {"lr": 1e-3, "k": 5, "hidsp": 8, "n_hidden": 16,
                 "n_layer": 1, "dropout": 0.2, "weight_decay": 5e-4,
                 "batch_size": 32, "bi": False, "rnn_model": "RNN"}


class TestEpiGNNRunner:
    def test_train_one_epoch_us_returns_dict(self):
        from src.tuning.runner import train_epignn
        result = train_epignn(FAST_CFG, US_EPIGNN_CFG, num_epochs=1, patience=999)
        assert isinstance(result, dict)
        assert "train_loss" in result and "val_loss" in result

    def test_train_one_epoch_sweden_finite_loss(self):
        from src.tuning.runner import train_epignn
        result = train_epignn(FAST_CFG, SE_EPIGNN_CFG, num_epochs=1, patience=999)
        assert len(result["train_loss"]) == 1
        assert torch.isfinite(torch.tensor(result["train_loss"][0]))

    def test_train_tracks_minimum_val_loss(self):
        from src.tuning.runner import train_epignn
        result = train_epignn(FAST_CFG, US_EPIGNN_CFG, num_epochs=3, patience=999)
        assert abs(result["best_val_loss"] - min(result["val_loss"])) < 1e-6

    def test_train_returns_best_state(self):
        from src.tuning.runner import train_epignn
        result = train_epignn(FAST_CFG, US_EPIGNN_CFG, num_epochs=2, patience=999)
        assert result["best_state"] is not None
        assert isinstance(result["best_state"], dict)

    def test_patience_stops_early(self):
        from src.tuning.runner import train_epignn
        result = train_epignn(FAST_CFG, US_EPIGNN_CFG, num_epochs=100, patience=1)
        assert len(result["train_loss"]) < 100


class TestColaGNNRunner:
    def test_train_one_epoch_us_returns_dict(self):
        from src.tuning.runner import train_colagnn
        result = train_colagnn(FAST_COLA_CFG, US_COLAGNN_CFG, num_epochs=1, patience=999)
        assert isinstance(result, dict)
        assert "train_loss" in result and "val_loss" in result

    def test_train_one_epoch_sweden_finite_loss(self):
        from src.tuning.runner import train_colagnn
        result = train_colagnn(FAST_COLA_CFG, SE_COLAGNN_CFG, num_epochs=1, patience=999)
        assert len(result["train_loss"]) == 1
        assert torch.isfinite(torch.tensor(result["train_loss"][0]))

    def test_train_returns_best_state(self):
        from src.tuning.runner import train_colagnn
        result = train_colagnn(FAST_COLA_CFG, US_COLAGNN_CFG, num_epochs=2, patience=999)
        assert result["best_state"] is not None
        assert isinstance(result["best_state"], dict)


class TestSearchSpaces:
    def test_epignn_space_has_required_keys(self):
        from src.tuning.hpo import EPIGNN_SPACE
        required = {"lr", "k", "hidP", "hidA", "n_hidden", "dropout", "weight_decay", "batch_size"}
        assert required.issubset(set(EPIGNN_SPACE.keys()))

    def test_colagnn_space_has_required_keys(self):
        from src.tuning.hpo import COLAGNN_SPACE
        required = {"lr", "k", "hidsp", "n_hidden", "dropout", "weight_decay", "batch_size"}
        assert required.issubset(set(COLAGNN_SPACE.keys()))
