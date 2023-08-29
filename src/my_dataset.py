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
        self.y = torch.tensor(y).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]


class SeriesDataset(Dataset):
    def __init__(self, X, n):
        # if isinstance(X, pd.DataFrame):
        #     X = X.values
        # if isinstance(y, pd.DataFrame):
        #     y = y.values
        # self.X = torch.tensor(X).float()
        # self.y = torch.tensor(y).float()
        self.X = X.values.tolist()
        self.n = n

    def __len__(self):
        return len(self.X) - self.n + 1

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx : idx + self.n]).float()
