import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        out = self.linear(out)
        return out


class MyDataset(Dataset):
    def __init__(self, factor_value, stock_return):
        factor_index = factor_value.index
        stock_return_stack = stock_return.stack().swaplevel(0, 1)
        return_index = stock_return_stack.index
        common_index = np.intersect1d(factor_index, return_index)

        factor_value = factor_value.loc[common_index, :]
        stock_return = stock_return_stack.loc[common_index].unstack().T
        return_quantile = stock_return.apply(lambda x: pd.qcut(x, np.arange(3) / 2, np.arange(2)),
                                             axis=1)

        self.len = len(stock_return.columns)
        self.factor_value = factor_value
        self.return_quantile = return_quantile

    def __getitem__(self, idx):
        code = self.factor_value.index.get_level_values('code').unique()[idx]
        single_stock_factor_value = self.factor_value.loc[pd.IndexSlice[code, :], :]
        single_stock_factor_value = single_stock_factor_value.reset_index('code', drop=True)
        single_stock_factor_value = single_stock_factor_value.reindex(self.return_quantile.index).ffill()
        single_stock_return_quantile = self.return_quantile.loc[:, code]

        single_stock_factor_value = single_stock_factor_value.values
        single_stock_return_quantile = np.array(single_stock_return_quantile)
        return single_stock_factor_value, single_stock_return_quantile

    def __len__(self):
        return self.len


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    factor_value = pd.read_pickle("F:\\factor_concat.pkl")
    stock_return = pd.read_pickle("F:\\Trade_data\\adjopen.pkl").pct_change(5).shift(-6).dropna(how='all', axis=0)

    train_x = factor_value.loc[(slice(None), slice("2018-01-01", "2018-01-31")), :]
    train_y = stock_return.loc["2018-01-01":"2018-01-31", :]

    input_size = 12
    hidden_size = 20
    num_layers = 5

    dataset = MyDataset(train_x, train_y)
    bath_size = 1
    dataloader = DataLoader(dataset, batch_size=bath_size, shuffle=True)

    model = MyModel(input_size, hidden_size, num_layers)
    model.to(device)

    epochs = 500
    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            x = x.float().to(device)
            y = y.float().unsqueeze(2).to(device)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        print(f"epoch:{epoch}, loss:{loss.item()}")

    model = model.to('cpu')
    pd.to_pickle(model, "F:\\model.pkl")
