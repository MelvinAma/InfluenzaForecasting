import sys
import argparse
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]

FLUSIGHT_LEVELS: list[float] = [
    0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99,
]


def _pinball_loss(pred, target, quantile_levels_tensor):
    """Pinball (quantile) loss averaged over all quantiles, regions, and batch.

    pred:    (batch, m, Q)
    target:  (batch, m)
    """
    errors = target.unsqueeze(-1) - pred
    loss = torch.max(quantile_levels_tensor * errors,
                     (quantile_levels_tensor - 1) * errors)
    return loss.mean()


def _replace_output_layer(model, attr_name, n_quantiles):
    """Replace the final linear layer to output n_quantiles instead of 1."""
    old_layer = getattr(model, attr_name)
    new_layer = nn.Linear(old_layer.in_features, n_quantiles)
    nn.init.xavier_uniform_(new_layer.weight)
    stdv = 1.0 / math.sqrt(n_quantiles)
    new_layer.bias.data.uniform_(-stdv, stdv)
    setattr(model, attr_name, new_layer)
EPIGNN_SRC = PROJ_ROOT / "Models" / "EpiGNN" / "src"
COLAGNN_SRC = PROJ_ROOT / "Models" / "ColaGNN" / "src"

_EPIGNN_MODULES = {"models", "data", "utils", "layers", "ablation"}
_COLAGNN_MODULES = {"models", "data", "utils", "layers", "dcrnn_model", "dcrnn_cell", "dcrnn_utils"}


def _activate_epignn():
    colagnn = str(COLAGNN_SRC)
    while colagnn in sys.path:
        sys.path.remove(colagnn)
    for name in _COLAGNN_MODULES | _EPIGNN_MODULES:
        sys.modules.pop(name, None)
    epignn = str(EPIGNN_SRC)
    while epignn in sys.path:
        sys.path.remove(epignn)
    sys.path.insert(0, epignn)


def _activate_colagnn():
    epignn = str(EPIGNN_SRC)
    while epignn in sys.path:
        sys.path.remove(epignn)
    for name in _COLAGNN_MODULES | _EPIGNN_MODULES:
        sys.modules.pop(name, None)
    colagnn = str(COLAGNN_SRC)
    while colagnn in sys.path:
        sys.path.remove(colagnn)
    sys.path.insert(0, colagnn)


def _seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epignn(config: dict, config_path: str, num_epochs: int = 200,
                 patience: int = 20, seed: int = 42, device: str = "cpu",
                 quantile_levels: Optional[list[float]] = None) -> dict:
    _activate_epignn()
    from data import MExDataLoader
    from models import EpiGNN

    _seed_everything(seed)

    cuda = device.startswith("cuda") and torch.cuda.is_available()
    loader = MExDataLoader(config_path, cuda=cuda)

    args = argparse.Namespace(
        window=loader.P,
        horizon=loader.h,
        n_layer=config.get("n_layer", 1),
        n_hidden=config.get("n_hidden", 20),
        dropout=config.get("dropout", 0.2),
        k=config.get("k", 8),
        hidR=config.get("hidR", 64),
        hidA=config.get("hidA", 64),
        hidP=config.get("hidP", 1),
        hw=config.get("hw", 0),
        extra="", label="", pcc="",
        n=config.get("n", 2),
        res=config.get("res", 0),
        s=config.get("s", 2),
        cuda=cuda,
        ablation=None,
    )

    model = EpiGNN(args, loader)

    q_mode = quantile_levels is not None
    q_tensor = None
    if q_mode:
        n_q = len(quantile_levels)
        _replace_output_layer(model, "output", n_q)
        dev = torch.device(device if cuda else "cpu")
        q_tensor = torch.tensor(quantile_levels, dtype=torch.float32, device=dev)

    if cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("lr", 1e-3),
        weight_decay=config.get("weight_decay", 5e-4),
    )

    batch_size = config.get("batch_size", 128)
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    bad_counter = 0

    for epoch in range(1, num_epochs + 1):
        model.train(True)
        total_loss, n_samples = 0.0, 0
        for inputs in loader.get_batches(loader.train, batch_size, shuffle=True):
            X, Y, index = inputs[0], inputs[1], inputs[2]
            optimizer.zero_grad()
            out, _ = model(X, index)
            if q_mode:
                loss = _pinball_loss(out, Y, q_tensor)
            else:
                if Y.size(0) == 1:
                    Y = Y.view(-1)
                    out = out.view(-1)
                loss = F.mse_loss(out, Y)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            n_samples += 1
        if n_samples == 0:
            raise RuntimeError(f"Train split empty for config {config_path}")
        train_loss = total_loss / n_samples

        model.train(False)
        val_total, n_val = 0.0, 0
        with torch.no_grad():
            for inputs in loader.get_batches(loader.val, batch_size, shuffle=False):
                X, Y, index = inputs[0], inputs[1], inputs[2]
                out, _ = model(X, index)
                if q_mode:
                    val_total += _pinball_loss(out, Y, q_tensor).item()
                else:
                    if Y.size(0) == 1:
                        Y = Y.view(-1)
                        out = out.view(-1)
                    val_total += F.mse_loss(out, Y).item()
                n_val += 1
        if n_val == 0:
            raise RuntimeError(f"Val split empty for config {config_path}")
        val_loss = val_total / n_val

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_counter = 0
        else:
            bad_counter += 1
        if bad_counter >= patience:
            break

    history["best_val_loss"] = best_val
    history["best_epoch"] = best_epoch
    history["best_state"] = best_state
    if q_mode:
        history["quantile_levels"] = quantile_levels
    return history


def train_colagnn(config: dict, config_path: str, num_epochs: int = 200,
                  patience: int = 20, seed: int = 42, device: str = "cpu",
                  quantile_levels: Optional[list[float]] = None) -> dict:
    _activate_colagnn()
    from data import MExDataLoader
    from models import cola_gnn

    _seed_everything(seed)

    cuda = device.startswith("cuda") and torch.cuda.is_available()
    loader = MExDataLoader(config_path, cuda=cuda)

    args = argparse.Namespace(
        window=loader.P,
        horizon=loader.h,
        n_layer=config.get("n_layer", 1),
        n_hidden=config.get("n_hidden", 20),
        dropout=config.get("dropout", 0.2),
        k=config.get("k", 10),
        hidsp=config.get("hidsp", 10),
        bi=config.get("bi", False),
        rnn_model=config.get("rnn_model", "RNN"),
        cuda=cuda,
    )

    model = cola_gnn(args, loader)

    q_mode = quantile_levels is not None
    q_tensor = None
    if q_mode:
        n_q = len(quantile_levels)
        _replace_output_layer(model, "out", n_q)
        dev = torch.device(device if cuda else "cpu")
        q_tensor = torch.tensor(quantile_levels, dtype=torch.float32, device=dev)

    if cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("lr", 1e-3),
        weight_decay=config.get("weight_decay", 5e-4),
    )

    batch_size = config.get("batch_size", 32)
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    bad_counter = 0

    for epoch in range(1, num_epochs + 1):
        model.train(True)
        total_loss, n_samples = 0.0, 0
        for inputs in loader.get_batches(loader.train, batch_size, shuffle=True):
            X, Y = inputs[0], inputs[1]
            optimizer.zero_grad()
            out, _ = model(X)
            if q_mode:
                loss = _pinball_loss(out, Y, q_tensor)
            else:
                if Y.size(0) == 1:
                    Y = Y.view(-1)
                loss = F.l1_loss(out, Y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            n_samples += 1
        if n_samples == 0:
            raise RuntimeError(f"Train split empty for config {config_path}")
        train_loss = total_loss / n_samples

        model.train(False)
        val_total, n_val = 0.0, 0
        with torch.no_grad():
            for inputs in loader.get_batches(loader.val, batch_size, shuffle=False):
                X, Y = inputs[0], inputs[1]
                out, _ = model(X)
                if q_mode:
                    val_total += _pinball_loss(out, Y, q_tensor).item()
                else:
                    if Y.size(0) == 1:
                        Y = Y.view(-1)
                    val_total += F.l1_loss(out, Y).item()
                n_val += 1
        if n_val == 0:
            raise RuntimeError(f"Val split empty for config {config_path}")
        val_loss = val_total / n_val

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_counter = 0
        else:
            bad_counter += 1
        if bad_counter >= patience:
            break

    history["best_val_loss"] = best_val
    history["best_epoch"] = best_epoch
    history["best_state"] = best_state
    if q_mode:
        history["quantile_levels"] = quantile_levels
    return history
