import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out)
        return out.squeeze(2)


class MyDataset(Dataset):
    def __init__(
        self,
        factor_value,
        stock_return,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ):
        factor_value = factor_value.loc[(slice(None), slice(start_date, end_date)), :]
        stock_return = stock_return.loc[start_date:end_date, :]

        period = len(stock_return)
        threshold = int(period * 0.05)
        selected_stock = (
            stock_return.isna().sum()[stock_return.isna().sum() < threshold].index
        )
        selected_stock = np.intersect1d(
            factor_value.index.get_level_values(0).unique(), selected_stock
        )

        na_factor_value = factor_value.groupby("code").apply(lambda x: x.isna().sum())
        na_factor_value = na_factor_value[na_factor_value > threshold].dropna(how="all")
        inc_factor_value = factor_value.groupby("code").apply(lambda x: period - len(x))
        inc_factor_value = inc_factor_value[inc_factor_value > threshold].dropna(
            how="all"
        )

        selected_stock = np.setdiff1d(selected_stock, na_factor_value.index)
        selected_stock = np.setdiff1d(selected_stock, inc_factor_value.index)
        factor_value = factor_value.loc[(selected_stock, slice(None)), :]
        stock_return = stock_return[selected_stock]
        stock_return = stock_return.apply(lambda x: x.fillna(x.mean()), axis=1)

        stock_return_stack = stock_return.stack().swaplevel(0, 1).sort_index()
        return_index = stock_return_stack.index
        factor_value = factor_value.reindex(return_index)
        factor_value = factor_value.groupby("dt").apply(lambda x: x.fillna(x.mean()))

        return_quantile = stock_return.apply(
            lambda x: pd.qcut(x, np.arange(3) / 2, np.arange(2)), axis=1
        )

        self.len = len(stock_return.columns)
        self.factor_value = factor_value
        self.return_quantile = return_quantile
        self.valid_stock = return_quantile.columns

    def __getitem__(self, idx):
        code = self.valid_stock[idx]
        single_stock_factor_value = self.factor_value.loc[pd.IndexSlice[code, :], :]
        single_stock_factor_value = single_stock_factor_value.reset_index(
            "code", drop=True
        )
        single_stock_factor_value = single_stock_factor_value.reindex(
            self.return_quantile.index
        ).ffill()
        single_stock_return_quantile = self.return_quantile.loc[:, code]

        single_stock_factor_value = single_stock_factor_value.values
        single_stock_return_quantile = np.array(single_stock_return_quantile)
        return single_stock_factor_value, single_stock_return_quantile

    def __len__(self):
        return self.len


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    factor_value = pd.read_pickle("F:\\factor_concat.pkl")
    stock_return = (
        pd.read_pickle("F:\\Trade_data\\adjopen.pkl")
        .pct_change(5)
        .shift(-6)
        .dropna(how="all", axis=0)
    )

    input_size = 12
    hidden_size = 20
    num_layers = 5

    dataset = MyDataset(
        factor_value, stock_return, pd.Timestamp(2018, 1, 1), pd.Timestamp(2018, 12, 31)
    )
    bath_size = 256
    dataloader = DataLoader(dataset, batch_size=bath_size, shuffle=True)

    model = MyModel(input_size, hidden_size, num_layers)
    model.to(device)

    epochs = 500
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    pbar = tqdm(range(1, epochs + 1), desc="正在训练模型...")
    for epoch in pbar:
        loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            x = x.float().to(device)
            y = y.float().to(device)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        pbar.set_description(f"epoch:{epoch}, loss:{loss.item():.2f}\t")
        # print()

    model = model.to("cpu")

    pd.to_pickle(model, "F:\\model.pkl")
