import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.utils.data import Dataset


def get_valid_dataset_location(factor_concat, stock_return, period=10):
    factor_concat.columns.names = ['factor', 'code']
    print("getting valid location")
    x_valid_batch = (
        factor_concat.groupby("code", axis=1)
        .apply(lambda x: x.isna().any(axis=1))
        .rolling(period)
        .sum()
        .shift(-period + 1)
        .dropna(how="all")
    )
    y_valid_batch = stock_return.isna().rolling(period).sum().shift(-period + 1)

    valid_location_list = []
    for date in tqdm(x_valid_batch.index):
        for code in x_valid_batch.columns:
            if (x_valid_batch.loc[date, code] == 0) and (
                y_valid_batch.loc[date, code] == 0
            ):
                valid_location_list.append([date, code])
    print(f"Dataset length: {len(valid_location_list)}")
    return valid_location_list


class MyDataset(Dataset):
    def __init__(
        self,
        factor_concat,
        stock_return,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        quantile: int = 2,
        period: int = 10,
    ):
        factor_concat = factor_concat.loc[start_date:end_date, :]
        stock_return = stock_return.loc[start_date:end_date, :]
        common_date = np.intersect1d(factor_concat.index, stock_return.index)
        factor_concat = factor_concat.loc[common_date]
        stock_return = stock_return.loc[common_date]

        return_quantile = stock_return.apply(
            lambda x: pd.qcut(
                x, np.arange(quantile + 1) / quantile, np.arange(quantile)
            ),
            axis=1,
        )
        np_return_quantile = np.where(
            return_quantile >= (quantile - 1),
            1,
            np.where(return_quantile < (quantile - 1), 0, np.nan),
        )
        return_quantile = pd.DataFrame(
            np_return_quantile,
            columns=return_quantile.columns,
            index=return_quantile.index,
        )
        del np_return_quantile
        self.valid_location_list = get_valid_dataset_location(
            factor_concat, stock_return, period
        )
        self.factor_concat = factor_concat
        self.return_quantile = return_quantile
        self.date_list = common_date
        self.period = period

    def __getitem__(self, idx):
        [date, code] = self.valid_location_list[idx]
        date_idx = np.where(self.date_list == date)[0][0]
        temp_period = self.date_list[date_idx : date_idx + self.period]
        last_date = self.date_list[date_idx + self.period-1]
        last_date = pd.to_datetime(str(last_date)).strftime("%Y-%m-%d")
        x_single_batch = self.factor_concat.loc[temp_period, (slice(None), code)].values
        y_single_batch = self.return_quantile.loc[last_date, code]
        return code, last_date, x_single_batch, y_single_batch

    def __len__(self):
        return len(self.valid_location_list)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    factor_concat = pd.read_pickle("./data/factor_concat.pkl")
    stock_return = pd.read_pickle("./data/stock_return.pkl")
    dataset = MyDataset(
        factor_concat, stock_return, pd.Timestamp(2018, 1, 1), pd.Timestamp(2019, 1, 1)
    )
    print(len(dataset))
    dataloader = DataLoader(dataset)
    for code, x, y in dataloader:
        print(code)
        print(x.shape)
        print(y.shape)
        break
