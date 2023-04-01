import pandas as pd

from torch.utils.data import DataLoader
from .dataset import MyDataset
from .config import EPOCHS, PERIOD

period = PERIOD


def train(factor_value,
          stock_return,
          start_date: pd.Timestamp,
          end_date: pd.Timestamp
          ):
    global epochs, period

    model = pd.read_pickle(".\mode\model_2017.pkl")
    dataset = MyDataset(factor_value, stock_return, start_date, end_date, period=period)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for code, date, x, y in dataloader:
        model(x.float())
        break

    return