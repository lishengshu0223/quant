import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset


def get_valid_dataset_location(factor_concat, stock_return, period=10):
    factor_concat.columns.names = ["factor", "code"]
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
        temp_period = self.date_list[date_idx: date_idx + self.period]
        last_date = self.date_list[date_idx + self.period - 1]
        last_date = pd.to_datetime(str(last_date)).strftime("%Y-%m-%d")
        x_single_batch = self.factor_concat.loc[temp_period, (slice(None), code)].values
        y_single_batch = self.return_quantile.loc[last_date, code]
        return code, last_date, x_single_batch, y_single_batch

    def __len__(self):
        return len(self.valid_location_list)


class MyFastDataset(Dataset):
    def __init__(
            self,
            value_dict: dict,
            quantile_return: pd.DataFrame,
            date_num_dict: dict,
            code_num_dict: dict,
            num_date_dict: dict,
            num_code_dict: dict,
    ):
        fake_multi_index = pd.MultiIndex.from_tuples(list(value_dict.keys()))
        common_index = np.intersect1d(fake_multi_index, quantile_return.index.swaplevel(0, 1))
        keys_to_delete = np.setdiff1d(fake_multi_index, common_index)
        for key in keys_to_delete:
            value_dict.pop(key, None)
        quantile_return = quantile_return.loc[pd.MultiIndex.from_tuples(common_index).swaplevel(0,1)]
        self.sample = value_dict
        self.label = quantile_return.to_dict()
        self.keys_list = list(quantile_return.to_dict().keys())
        self.date_num_dict = date_num_dict
        self.code_num_dict = code_num_dict
        self.num_date_dict = num_date_dict
        self.num_code_dict = num_code_dict

    def __len__(self):
        return len(self.keys_list)

    def __getitem__(self, idx):
        key = self.keys_list[idx]
        inverse_key = (key[1], key[0])
        date_num = torch.Tensor([self.date_num_dict[key[0]]])
        code_num = torch.Tensor([self.code_num_dict[key[1]]])
        sample = torch.Tensor(self.sample[inverse_key].flatten())
        label = torch.Tensor([self.label[key]])
        return date_num, code_num, sample, label
