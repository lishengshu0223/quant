from typing import Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SectionalDataset(Dataset):
    def __init__(self, data: pd.DataFrame, cat_col_names: Union[str, list[str]] = None):
        self._len = len(data)
        self.cat_idxs = []
        self.cat_dims = []

        if "label" in data.columns:
            self.label = torch.from_numpy(data["label"].values.astype(np.int8)).to(
                torch.uint8
            )
            data = data.drop(columns="label")
        else:
            self.label = None

        if cat_col_names is not None:
            if isinstance(cat_col_names, str):
                cat_col_names = [cat_col_names]
            for cat_col_name in cat_col_names:
                if cat_col_name not in data.columns:
                    msg = f"'cat_col_names'中的{cat_col_name}不在data的columns当中"
                    raise ValueError(msg)
            for cat_col_name in cat_col_names:
                unique_cat = data[cat_col_name].unique()
                cat_dim = len(unique_cat)
                replaced_num = np.arange(0, cat_dim)
                data[cat_col_name] = data[cat_col_name].replace(
                    unique_cat, replaced_num
                )
                cat_idx = np.where(data.columns==cat_col_name)[0][0]
                self.cat_idxs.append(cat_idx)
                self.cat_dims.append(cat_dim)

        self.data = torch.from_numpy(data.values.astype(np.float16)).to(torch.float16)

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        if self.label is None:
            return self.data[item, :]
        else:
            return self.data[item, :], self.label[item]


class SeriesDataset(Dataset):
    def __init__(self, X, y, period):
        self.X = X.iloc[:, :-1].values.tolist()
        self.y = y.values.tolist()
        self.period = period

    def __len__(self):
        return int(len(self.y) / 8) - self.period + 1

    def __getitem__(self, item):
        x_idx = int(item * 8)
        X = torch.tensor(self.X[x_idx : x_idx + self.period * 8]).float()
        y = torch.tensor(self.y[x_idx + self.period * 8 - 1]).long()
        return X, y


class SectionalPredictionDataset(Dataset):
    def __init__(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.X = torch.tensor(X).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx, :]