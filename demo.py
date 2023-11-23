import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from src.my_network import MyNetwork as Network
from src.my_dataset import SectionalDataset as MyDataset


# data = pd.read_pickle("./data/large_small_cap.pkl")
# data = data.loc[("000852.XSHG", slice(None)), :]
# data.index = data.index.droplevel(0)
# data_idx = data.index
# daily_price = data.resample('D').first().dropna(how="all")["open"]
# p = 1
# daily_rtn = daily_price.pct_change(p).shift(-p-1)
# daily_rtn[daily_rtn > 0] = 1
# daily_rtn[daily_rtn < 0] = 0
# daily_rtn = daily_rtn.reindex(data.index, method='ffill').ffill()
# # data = pd.concat([data, daily_rtn], axis=1)
# # data = data.ffill().reindex(data_idx)
#
# batch_size = 128
# test_dataset = MyDataset(data, daily_rtn)
# test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)
#
# n = 2
# network = Network(6, n)
#
# criterion = CrossEntropyLoss()
# optimizer = Adam(network.parameters(), 1e-4)
#
# epochs = 100
# for epoch in range(epochs):
#     mean_loss = 0
#     for x, y in test_dataloader:
#         pred = network(x)
#         loss = criterion(pred, y)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         mean_loss += loss.item()
#     mean_loss = mean_loss / len(test_dataloader)
#     print(mean_loss)
#
import math
from torch import nn


class FullyConnect(nn.Module):
    def __init__(
        self, feature_num: int, output_dim: int, p: float = 0.1, shrink: int = 4
    ):
        super(FullyConnect, self).__init__()
        self.feature_num = feature_num
        self.fc_block = nn.ModuleList()
        while True:
            if int(feature_num / shrink) < output_dim:
                break
            fc_block = nn.Sequential(
                nn.Linear(feature_num, int(feature_num / shrink)),
                nn.ReLU(),
                nn.BatchNorm1d(int(feature_num / shrink)),
                nn.Dropout(p),
            )
            self.fc_block.append(fc_block)
            feature_num = int(feature_num / shrink)
        if feature_num != output_dim:
            fc_block = nn.Sequential(
                nn.Linear(feature_num, output_dim),
                nn.ReLU(),
                nn.BatchNorm1d(output_dim),
                nn.Dropout(p),
            )
            self.fc_block.append(fc_block)

    def forward(self, x):
        for nn in self.fc_block:
            x = nn(x)
        return x


class MyTabTransformer(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, p: float = 0.1, shrink: int = 16
    ):
        super(MyTabTransformer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = 2 ** (int(math.log(input_dim, 2)) + 1)
        self.linear = nn.Linear(self.input_dim, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=8, dropout=p
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fullyconnect = FullyConnect(self.d_model, self.output_dim, p, shrink)

    def forward(self, x):
        x = self.linear(x)
        x = self.transformer(x)
        out = self.fullyconnect(x)
        return out


path = "E:\\Factor_Download\\concat"
data = pd.read_pickle(os.path.join(path, "factor", "2012-01-01.pkl"))
label = pd.read_pickle(os.path.join(path, "label.pkl"))
label = label.loc[data.index].astype(int)
batch_size = 128
test_dataset = MyDataset(data, label)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)
network = MyTabTransformer(data.shape[1], 2)
print(network)
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
