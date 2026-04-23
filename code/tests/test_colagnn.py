import sys
import argparse
import json
import torch
import pytest
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]
EPIGNN_SRC = PROJ_ROOT / "Models" / "EpiGNN" / "src"
COLAGNN_SRC = PROJ_ROOT / "Models" / "ColaGNN" / "src"

US_CONFIG = str(PROJ_ROOT / "src" / "configs" / "us_colagnn.json")
SE_CONFIG = str(PROJ_ROOT / "src" / "configs" / "sweden_colagnn.json")

_EPIGNN_MODULES = {"models", "data", "utils", "layers", "ablation"}
_COLAGNN_MODULES = {"models", "data", "utils", "layers", "dcrnn_model", "dcrnn_cell", "dcrnn_utils"}


def _activate_colagnn():
    """Ensure ColaGNN's src is active and conflicting modules are cleared."""
    epignn = str(EPIGNN_SRC)
    while epignn in sys.path:
        sys.path.remove(epignn)
    for name in _COLAGNN_MODULES | _EPIGNN_MODULES:
        sys.modules.pop(name, None)
    colagnn = str(COLAGNN_SRC)
    if colagnn in sys.path:
        sys.path.remove(colagnn)
    sys.path.insert(0, colagnn)


def make_args(**overrides):
    defaults = dict(
        window=20, horizon=1, n_layer=1, n_hidden=20, seed=42,
        dropout=0.2, k=10, hidsp=10, cuda=False,
        rnn_model="RNN", bi=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestMExDataLoaderUS:
    def setup_method(self):
        _activate_colagnn()
        from data import MExDataLoader
        self.loader = MExDataLoader(US_CONFIG, cuda=False)

    def test_node_count(self):
        assert self.loader.m == 51

    def test_d_attribute(self):
        assert self.loader.d == 0

    def test_orig_adj_shape(self):
        assert self.loader.orig_adj.shape == (51, 51)

    def test_train_split_nonempty(self):
        X, Y = self.loader.train
        assert X.shape[0] > 0

    def test_test_season_nonempty(self):
        X, Y = self.loader.test
        assert X.shape[0] > 0

    def test_window_shape(self):
        X, Y = self.loader.train
        assert X.shape[1] == 20
        assert X.shape[2] == 51

    def test_get_batches_yields_two(self):
        # ColaGNN train loop unpacks only inputs[0] and inputs[1]
        batch = next(iter(self.loader.get_batches(self.loader.train, 32)))
        assert len(batch) == 2

    def test_normalization_range(self):
        X, Y = self.loader.train
        assert float(Y.max()) <= 1.0 + 1e-5
        assert float(Y.min()) >= -1e-5


class TestMExDataLoaderSweden:
    def setup_method(self):
        _activate_colagnn()
        from data import MExDataLoader
        self.loader = MExDataLoader(SE_CONFIG, cuda=False)

    def test_node_count(self):
        assert self.loader.m == 21

    def test_orig_adj_shape(self):
        assert self.loader.orig_adj.shape == (21, 21)

    def test_covid_season_excluded(self):
        import json
        from loader import FluDataLoader
        with open(SE_CONFIG) as f:
            cfg = json.load(f)
        fl = FluDataLoader(cfg)
        excluded = {"2020/21"}
        all_idx = fl.train_set + fl.val_set + fl.test_set
        for i in all_idx:
            assert fl.seasons[i] not in excluded


class TestColaGNNForwardUS:
    def setup_method(self):
        _activate_colagnn()
        from data import MExDataLoader
        from models import cola_gnn
        self.MExDataLoader = MExDataLoader
        self.cola_gnn = cola_gnn
        self.loader = MExDataLoader(US_CONFIG, cuda=False)
        self.args = make_args()

    def test_forward_output_shape(self):
        model = self.cola_gnn(self.args, self.loader)
        X, Y = self.loader.train
        with torch.no_grad():
            out, _ = model(X[:4])
        assert out.shape == (4, 51)

    def test_forward_no_nan(self):
        model = self.cola_gnn(self.args, self.loader)
        X, Y = self.loader.train
        with torch.no_grad():
            out, _ = model(X[:8])
        assert not torch.isnan(out).any()

    def test_hidsp_controls_spatial_width(self):
        args = make_args(hidsp=7)
        model = self.cola_gnn(args, self.loader)
        assert model.n_spatial == 7
        assert model.out.in_features == model.n_hidden + 7

    def test_wili_baseline_config_pins_reference_hidsp(self):
        config_path = PROJ_ROOT / "src" / "configs" / "us_wili_colagnn.json"
        with open(config_path, "r", encoding="utf-8") as handle:
            cfg = json.load(handle)
        assert cfg["hidsp"] == 10


class TestColaGNNForwardSweden:
    def setup_method(self):
        _activate_colagnn()
        from data import MExDataLoader
        from models import cola_gnn
        self.MExDataLoader = MExDataLoader
        self.cola_gnn = cola_gnn
        self.loader = MExDataLoader(SE_CONFIG, cuda=False)
        self.args = make_args()

    def test_forward_output_shape(self):
        model = self.cola_gnn(self.args, self.loader)
        X, Y = self.loader.train
        with torch.no_grad():
            out, _ = model(X[:4])
        assert out.shape == (4, 21)

    def test_forward_no_nan(self):
        model = self.cola_gnn(self.args, self.loader)
        X, Y = self.loader.train
        with torch.no_grad():
            out, _ = model(X[:8])
        assert not torch.isnan(out).any()
