from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]

US_CONFIG = {
    "data_path": "src/data/state_flu_admissions.txt",
    "adj_path": "src/data/state-adj-51.txt",
    "date_index_path": "src/data/date_index.csv",
    "date_format": "date",
    "window": 20,
    "horizon": 1,
    "split_mode": "season",
    "test_seasons": ["2024/25"],
    "val_seasons": ["2023/24"],
    "exclude_seasons": ["2020/21", "2025/26"],
    "train_ratio": 0.6,
    "val_ratio": 0.2,
}

SWEDEN_CONFIG = {
    "data_path": "src/data/sweden_flu_cases.txt",
    "adj_path": "src/data/sweden-adj-21.txt",
    "date_index_path": "src/data/sweden_week_index.csv",
    "date_format": "week",
    "window": 20,
    "horizon": 1,
    "split_mode": "season",
    "test_seasons": ["2024/25"],
    "val_seasons": ["2023/24"],
    "exclude_seasons": ["2020/21", "2025/26"],
    "train_ratio": 0.6,
    "val_ratio": 0.2,
}


def assign_seasons(iso_years, iso_weeks):
    seasons = []
    for year, week in zip(iso_years, iso_weeks):
        if week >= 40:
            seasons.append(f"{year}/{str(year + 1)[-2:]}")
        elif week <= 20:
            seasons.append(f"{year - 1}/{str(year)[-2:]}")
        else:
            seasons.append(None)
    return seasons


def _season_start_year(season):
    if season is None:
        return None
    return int(str(season).split("/")[0])


def _load_date_index(path, date_format):
    if date_format == "date":
        df = pd.read_csv(
            path, header=None, names=["idx", "date"], parse_dates=["date"]
        )
        iso = df["date"].dt.isocalendar()
        return iso.year.astype(int).values, iso.week.astype(int).values
    df = pd.read_csv(path, header=None, names=["idx", "year", "week"])
    weeks = df["week"].str.replace("w", "").astype(int).values
    return df["year"].values.astype(int), weeks


class FluDataLoader:

    def __init__(self, config):
        self.P = config["window"]
        self.h = config["horizon"]

        self.rawdat = np.loadtxt(ROOT / config["data_path"], delimiter=",")
        if self.rawdat.ndim == 1:
            self.rawdat = self.rawdat.reshape(-1, 1)
        self.n, self.m = self.rawdat.shape
        self.dat = np.zeros_like(self.rawdat)

        adj_np = np.loadtxt(ROOT / config["adj_path"], delimiter=",")
        self.adj = torch.Tensor(adj_np)
        self.orig_adj = self.adj.clone()
        self.degree_adj = torch.sum(self.orig_adj, dim=-1)

        if "date_index_path" in config:
            iso_years, iso_weeks = _load_date_index(
                ROOT / config["date_index_path"], config["date_format"]
            )
            self.seasons = assign_seasons(iso_years, iso_weeks)
        else:
            self.seasons = [None] * self.n

        if config["split_mode"] == "season":
            train_idx, val_idx, test_idx = self._split_by_season(config)
        else:
            train_idx, val_idx, test_idx = self._split_by_ratio(config)

        self.train_set = train_idx
        self.val_set = val_idx
        self.test_set = test_idx

        self._compute_normalization(train_idx)

        self.train = self._batchify(train_idx)
        self.val = self._batchify(val_idx)
        self.test = self._batchify(test_idx)

        if len(self.val[0]) == 0:
            self.val = self.test

    def _split_by_season(self, config):
        test_seasons = set(config.get("test_seasons", []))
        val_seasons = set(config.get("val_seasons", []))
        exclude_seasons = set(config.get("exclude_seasons", []))
        cutoff = config.get("train_cutoff_row", None)
        min_idx = self.P + self.h - 1
        boundary_seasons = val_seasons or test_seasons
        boundary_years = [
            _season_start_year(season)
            for season in boundary_seasons
            if _season_start_year(season) is not None
        ]
        train_boundary_year = min(boundary_years) if boundary_years else None

        train_idx, val_idx, test_idx = [], [], []
        for i in range(min_idx, self.n):
            season = self.seasons[i]
            if season is None:
                continue
            if season in exclude_seasons:
                continue
            if season in test_seasons:
                test_idx.append(i)
            elif season in val_seasons:
                val_idx.append(i)
            elif cutoff is not None and i >= cutoff:
                continue
            elif (
                train_boundary_year is not None
                and _season_start_year(season) >= train_boundary_year
            ):
                continue
            else:
                train_idx.append(i)
        return sorted(train_idx), sorted(val_idx), sorted(test_idx)

    def _split_by_ratio(self, config):
        min_idx = self.P + self.h - 1
        train_end = int(config["train_ratio"] * self.n)
        val_end = int((config["train_ratio"] + config["val_ratio"]) * self.n)
        return (
            list(range(min_idx, train_end)),
            list(range(train_end, val_end)),
            list(range(val_end, self.n)),
        )

    def _compute_normalization(self, train_idx):
        # MEx Thesis Adaptation: normalization quirk preserved from original code.
        # tmp[0][0] is the FIRST training window only (shape P x m), not all
        # windows.  Combined with all targets tmp[1] (shape n x m) this gives
        # a (P+n, m) matrix.  Using all windows (tmp[0] reshaped to n*P x m)
        # would be more correct but is intentionally left unchanged to match
        # the behaviour the model was trained and evaluated under.
        tmp = self._batchify(train_idx, use_raw=True)
        train_mx = torch.cat((tmp[0][0], tmp[1]), 0).numpy()
        self.max = np.max(train_mx, 0)
        self.min = np.min(train_mx, 0)
        self.peak_thold = np.mean(train_mx, 0)
        self.dat = (self.rawdat - self.min) / (self.max - self.min + 1e-12)

    def _batchify(self, idx_set, use_raw=False):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.m))
        source = self.rawdat if use_raw else self.dat
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :self.P, :] = torch.from_numpy(source[start:end, :])
            Y[i, :] = torch.from_numpy(source[idx_set[i], :])
        return [X, Y]

    def get_batches(self, data, batch_size, shuffle=True):
        inputs, targets = data[0], data[1]
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while start_idx < length:
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            yield [X, Y, excerpt]
            start_idx += batch_size
