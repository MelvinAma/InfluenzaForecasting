import sys
import argparse
import torch
import pytest
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]
EPIGNN_SRC = PROJ_ROOT / "Models" / "EpiGNN" / "src"
COLAGNN_SRC = PROJ_ROOT / "Models" / "ColaGNN" / "src"

US_CONFIG = str(PROJ_ROOT / "src" / "configs" / "us_epignn.json")
SE_CONFIG = str(PROJ_ROOT / "src" / "configs" / "sweden_epignn.json")

_EPIGNN_MODULES = {"models", "data", "utils", "layers", "ablation"}
_COLAGNN_MODULES = {"models", "data", "utils", "layers", "dcrnn_model", "dcrnn_cell", "dcrnn_utils"}


def _activate_epignn():
    """Ensure EpiGNN's src is active and conflicting modules are cleared."""
    colagnn = str(COLAGNN_SRC)
    while colagnn in sys.path:
        sys.path.remove(colagnn)
    for name in _COLAGNN_MODULES | _EPIGNN_MODULES:
        sys.modules.pop(name, None)
    epignn = str(EPIGNN_SRC)
    if epignn in sys.path:
        sys.path.remove(epignn)
    sys.path.insert(0, epignn)


def make_args(**overrides):
    defaults = dict(
        window=20, horizon=1, n_layer=1, n_hidden=20, seed=42,
        dropout=0.2, k=8, hidR=64, hidA=64, hidP=1, hw=0,
        extra="", label="", pcc="", n=2, res=0, s=2, cuda=False,
        ablation=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestMExDataLoaderUS:
    def setup_method(self):
        _activate_epignn()
        from data import MExDataLoader
        self.loader = MExDataLoader(US_CONFIG, cuda=False)

    def test_node_count(self):
        assert self.loader.m == 51

    def test_adj_shape(self):
        assert self.loader.adj.shape == (51, 51)

    def test_degree_adj_shape(self):
        assert self.loader.degree_adj.shape == (51,)

    def test_train_split_nonempty(self):
        X, Y = self.loader.train
        assert X.shape[0] > 0

    def test_test_season_nonempty(self):
        X, Y = self.loader.test
        assert X.shape[0] > 0

    def test_no_train_test_overlap(self):
        overlap = set(self.loader.train_set) & set(self.loader.test_set)
        assert len(overlap) == 0

    def test_window_shape(self):
        X, Y = self.loader.train
        assert X.shape[1] == 20
        assert X.shape[2] == 51

    def test_get_batches_yields_three(self):
        batch = next(iter(self.loader.get_batches(self.loader.train, 32)))
        assert len(batch) == 3

    def test_normalization_range(self):
        X, Y = self.loader.train
        assert float(Y.max()) <= 1.0 + 1e-5
        assert float(Y.min()) >= -1e-5


class TestMExDataLoaderSweden:
    def setup_method(self):
        _activate_epignn()
        from data import MExDataLoader
        self.loader = MExDataLoader(SE_CONFIG, cuda=False)

    def test_node_count(self):
        assert self.loader.m == 21

    def test_adj_shape(self):
        assert self.loader.adj.shape == (21, 21)

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


class TestEpiGNNForwardUS:
    def setup_method(self):
        _activate_epignn()
        from data import MExDataLoader
        from models import EpiGNN
        self.MExDataLoader = MExDataLoader
        self.EpiGNN = EpiGNN
        self.loader = MExDataLoader(US_CONFIG, cuda=False)
        self.args = make_args()

    def test_forward_output_shape(self):
        model = self.EpiGNN(self.args, self.loader)
        X, Y = self.loader.train
        with torch.no_grad():
            out, _ = model(X[:4], None)
        assert out.shape == (4, 51)

    def test_forward_no_nan(self):
        model = self.EpiGNN(self.args, self.loader)
        X, Y = self.loader.train
        with torch.no_grad():
            out, _ = model(X[:8], None)
        assert not torch.isnan(out).any()


class TestEpiGNNForwardSweden:
    def setup_method(self):
        _activate_epignn()
        from data import MExDataLoader
        from models import EpiGNN
        self.MExDataLoader = MExDataLoader
        self.EpiGNN = EpiGNN
        self.loader = MExDataLoader(SE_CONFIG, cuda=False)
        self.args = make_args()

    def test_forward_output_shape(self):
        model = self.EpiGNN(self.args, self.loader)
        X, Y = self.loader.train
        with torch.no_grad():
            out, _ = model(X[:4], None)
        assert out.shape == (4, 21)

    def test_forward_no_nan(self):
        model = self.EpiGNN(self.args, self.loader)
        X, Y = self.loader.train
        with torch.no_grad():
            out, _ = model(X[:8], None)
        assert not torch.isnan(out).any()
