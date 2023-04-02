import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.config import QUANTILE

quantile = QUANTILE


def make_idx_num_dict(idx_array):
    idx_num_dict = {}
    num_idx_dict = {}
    for i in range(len(idx_array)):
        idx = idx_array[i]
        idx_num_dict[idx] = i
        num_idx_dict[i] = idx
    return idx_num_dict, num_idx_dict


class MyDataset(Dataset):
    def __init__(self, factor_concat: pd.DataFrame, stock_return: pd.DataFrame):
        factor_stack = factor_concat.stack().dropna(how="all", axis=1).dropna()
        return_stack = stock_return.stack().dropna()
        common_index = np.intersect1d(factor_stack.index, return_stack.index)
        factor_stack = factor_stack.loc[common_index, :]
        return_stack = return_stack.loc[common_index]
        return_stack.index.names = ["date", "code"]
        unique_date = factor_stack.index.get_level_values(0).unique()
        unique_code = factor_stack.index.get_level_values(1).unique()
        date_num_dict, num_date_dict = make_idx_num_dict(unique_date)
        code_num_dict, num_code_dict = make_idx_num_dict(unique_code)

        quantile_return = return_stack.groupby("date").apply(
            lambda x: pd.qcut(
                x, np.arange(quantile + 1) / quantile, np.arange(quantile)
            )
        )
        quantile_return[quantile_return < (quantile - 1)] = 0
        quantile_return[quantile_return == (quantile - 1)] = 1

        self.sample = factor_stack.T.to_dict(orient="list")
        self.label = quantile_return.to_dict()
        self.keys_list = list(quantile_return.to_dict().keys())
        self.date_num_dict = date_num_dict
        self.code_num_dict = code_num_dict
        self.num_date_dict = num_date_dict
        self.num_code_dict = num_code_dict
        return

    def __len__(self):
        return len(self.keys_list)

    def __getitem__(self, idx):
        key = self.keys_list[idx]
        date_num = torch.Tensor([self.date_num_dict[key[0]]])
        code_num = torch.Tensor([self.code_num_dict[key[1]]])
        sample = torch.Tensor(self.sample[key])
        label = torch.Tensor([self.label[key]])
        return date_num, code_num, sample, label


class MyFastDataset(Dataset):
    def __init__(
        self,
        factor_stack: pd.DataFrame,
        quantile_return: pd.DataFrame,
        date_num_dict: dict,
        code_num_dict: dict,
        num_date_dict: dict,
        num_code_dict: dict,
    ):
        common_index = np.intersect1d(factor_stack.index, quantile_return.index)
        factor_stack = factor_stack.loc[common_index, :]
        quantile_return = quantile_return.loc[common_index]
        self.sample = factor_stack.T.to_dict(orient="list")
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
        date_num = torch.Tensor([self.date_num_dict[key[0]]])
        code_num = torch.Tensor([self.code_num_dict[key[1]]])
        sample = torch.Tensor(self.sample[key])
        label = torch.Tensor([self.label[key]])
        return date_num, code_num, sample, label
