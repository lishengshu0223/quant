import math

import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import MyDataset
from .model import MyModel
from .config import EPOCHS, PERIOD

epochs = EPOCHS
period = PERIOD


def train(factor_value,
          stock_return,
          start_date: pd.Timestamp,
          end_date: pd.Timestamp
          ):
    global epochs, period

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_size = len(factor_value.columns.get_level_values(0).unique())
    hidden_size = input_size * 2
    num_layers = int(math.log(input_size) / 2)

    dataset = MyDataset(
        factor_value, stock_return, start_date, end_date, period=period
    )
    batch_size = 256
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MyModel(input_size, hidden_size, num_layers, period)
    model.to(device)

    learning_rate = 1e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    test_loss_list = [0]
    for epoch in (range(1, epochs + 1)):
        pbar = tqdm(dataloader)
        test_loss = 0
        last_loss = test_loss_list[-1]
        for _, _, x, y in pbar:
            optimizer.zero_grad()
            x = x.float().to(device)
            y = y.float().to(device)
            y_pred = model(x).squeeze()
            loss = criterion(y_pred, y)
            test_loss = test_loss + loss.item()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"epoch:{epoch}, loss:{last_loss:.4e}\t")
        test_loss_list.append(test_loss / len(dataset))

    model = model.to("cpu")
    return model
