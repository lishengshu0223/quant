import os
import math

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torch.utils.data import Dataset, DataLoader
from src.my_dataset import SectionalDataset as MyDataset
from src.my_model import MyModel
from sklearn.model_selection import train_test_split

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


if __name__ == '__main__':
    path = 'F:/Factor/concat'
    label = pd.read_pickle(os.path.join(path, 'label.pkl'))
    date_range = pd.date_range('2018-01-01', '2023-08-01', freq='6M').strftime('%Y-%m-01')

    for date in date_range:
        X_train = []
        print(date)
        period_range = pd.date_range(end=date, periods=36, freq='M').strftime('%Y-%m-01')
        for period in period_range:
            X_train.append(pd.read_pickle(os.path.join(path, 'factor', f'{period}.pkl')))

        X_train = pd.concat(X_train).fillna(0)
        y_train = label.loc[X_train.index].fillna(0)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=0)
        network = FullyConnect(X_train.shape[1], 2)
        network_kwargs = {"input_dim": X_train.shape[1], "output_dim": 2, "p": 0.5, "shrink": 16}
        model = MyModel(
            batch_size=1024,
            max_epoch=100,
            early_stop_epoch=10,
            loss_fn=CrossEntropyLoss,
            optimizer=Adam,
            network=MyTabTransformer,
            **network_kwargs
        )
        # model.fit(X_train.values, y_train.values, eval_set=[(X_test.values, y_test.values)])
        # model.save('F:/Temp/model', f'{date}')
        model.load('F:/Temp/model/2018-01-01.pth')
        torch.cuda.empty_cache()
        X_test = []
        for month in range(1, 13):
            X_test.append(pd.read_pickle(os.path.join(path, 'factor', f'2018-{str(month).zfill(2)}-01.pkl')))
        X_test = pd.concat(X_test)
        score = model.predict_proba(X_test.values)[:, 1]
        score = pd.Series(score, index=X_test)
        print(score)
        score.to_pickle('score.pkl')
        break

