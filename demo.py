from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from src.my_network import MyNetwork as Network
from src.my_dataset import SectionalDataset as MyDataset


data = pd.read_pickle("./data/large_small_cap.pkl")
data = data.loc[("000852.XSHG", slice(None)), :]
data.index = data.index.droplevel(0)
data_idx = data.index
daily_price = data.resample('D').first().dropna(how="all")["open"]
p = 1
daily_rtn = daily_price.pct_change(p).shift(-p-1)
daily_rtn[daily_rtn > 0] = 1
daily_rtn[daily_rtn < 0] = 0
daily_rtn = daily_rtn.reindex(data.index, method='ffill').ffill()
# data = pd.concat([data, daily_rtn], axis=1)
# data = data.ffill().reindex(data_idx)


batch_size = 128
test_dataset = MyDataset(data, daily_rtn)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)

n = 2
network = Network(6, n)

criterion = CrossEntropyLoss()
optimizer = Adam(network.parameters(), 1e-4)

epochs = 100
for epoch in range(epochs):
    mean_loss = 0
    for x, y in test_dataloader:
        pred = network(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        mean_loss += loss.item()
    mean_loss = mean_loss / len(test_dataloader)
    print(mean_loss)
