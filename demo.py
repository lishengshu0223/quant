from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from src.my_network import GruNetworkWithAttention as Network


data = pd.read_pickle("E:/Factor_Download/temp/large_small_cap.pkl")
data = data.loc[("000852.XSHG", slice(None)), :]
data.index = data.index.droplevel(0)
data_idx = data.index
daily_price = data.resample('D').first().dropna(how="all")["open"]
n = 1
daily_rtn = daily_price.pct_change(n).shift(-n-1).to_frame("rtn")
daily_rtn[daily_rtn > 0] = 1
daily_rtn[daily_rtn < 0] = 0
data = pd.concat([data, daily_rtn], axis=1)
data = data.ffill().reindex(data_idx)

class MyDataset(Dataset):
    def __init__(self, X, period):
        self.X = X.iloc[:, :-1].values.tolist()
        self.y = X['rtn'].values.tolist()
        self.period = period

    def __len__(self):
        return int(len(self.y) / 8) - self.period + 1

    def __getitem__(self, item):
        x_idx = int(item * 8)
        X = torch.tensor(self.X[x_idx : x_idx + self.period * 8]).float()
        y = torch.tensor(self.y[x_idx + self.period * 8 - 1]).long()
        return X, y

n = 5
test_dataset = MyDataset(data, n)
test_dataloader = DataLoader(test_dataset, 64, shuffle=True)

network = Network(6 ,n)

criterion = CrossEntropyLoss()
optimizer = SGD(network.parameters(), 1e-2)

epochs = 100
for epoch in range(epochs):
    mean_loss = []
    for x, y in test_dataloader:
        pred = network(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        mean_loss.append(loss.item() / len(x))
    mean_loss = np.sum(mean_loss) / len(mean_loss)
    print(mean_loss)
