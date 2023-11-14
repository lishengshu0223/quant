import pandas as pd
import torch
from torch.utils.data import Dataset


class SectionalDataset(Dataset):
    def __init__(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.values
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]


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
