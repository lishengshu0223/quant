import os
import h5py
import numpy as np
import pandas as pd
from .config import (
    get_data_path,
    BUNDLE_DIR,
    BUNDLE_BUNDLE_DIR,
    DIVIDENDS_H5,
    EX_CUM_FACTOR_H5,
    SPLIT_FACTOR_H5,
    DIVIDEND_FIELDS,
)
from ._utils import normalize_codes, normalize_date


def _get_dividends_path():
    return get_data_path(BUNDLE_DIR, BUNDLE_BUNDLE_DIR, DIVIDENDS_H5)


def _get_ex_cum_factor_path():
    return get_data_path(BUNDLE_DIR, BUNDLE_BUNDLE_DIR, EX_CUM_FACTOR_H5)


def _get_split_factor_path():
    return get_data_path(BUNDLE_DIR, BUNDLE_BUNDLE_DIR, SPLIT_FACTOR_H5)


def _factor_int_to_timestamp(dt_int):
    if dt_int == 0:
        return pd.Timestamp("1990-01-01")
    s = str(int(dt_int))
    if len(s) >= 8:
        return pd.Timestamp(year=int(s[:4]), month=int(s[4:6]), day=int(s[6:8]))
    return pd.Timestamp(s)


def _date_int_to_timestamp(date_int):
    s = str(int(date_int))
    return pd.Timestamp(year=int(s[:4]), month=int(s[4:6]), day=int(s[6:8]))


def get_dividend(order_book_ids):
    """
    获取股票分红送股信息

    Parameters
    ----------
    order_book_ids : str or list[str]
        股票代码

    Returns
    -------
    pd.DataFrame
        MultiIndex: [order_book_id, ...], columns包含分红相关字段
    """
    codes = normalize_codes(order_book_ids)
    if not codes:
        return pd.DataFrame()

    h5_path = _get_dividends_path()
    if not os.path.exists(h5_path):
        return pd.DataFrame()

    all_dfs = []
    with h5py.File(h5_path, "r") as f:
        for code in codes:
            if code not in f:
                continue
            data = f[code][:]
            if len(data) == 0:
                continue
            df = pd.DataFrame(data)
            df["order_book_id"] = code
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)

    # 日期字段转换
    for col in ["book_closure_date", "announcement_date", "ex_dividend_date", "payable_date"]:
        if col in result.columns:
            result[col] = result[col].apply(_date_int_to_timestamp)

    result = result.set_index("order_book_id").sort_index()

    valid_fields = [f for f in DIVIDEND_FIELDS if f in result.columns]
    if valid_fields:
        result = result[valid_fields]

    for col in result.columns:
        if result[col].dtype == np.float64:
            result[col] = result[col].astype("float32")

    return result


def get_ex_cum_factor(order_book_ids):
    """
    获取累计复权因子

    Parameters
    ----------
    order_book_ids : str or list[str]
        股票代码

    Returns
    -------
    pd.DataFrame
        index: start_date, columns: [ex_cum_factor], 每只股票一组
    """
    codes = normalize_codes(order_book_ids)
    if not codes:
        return pd.DataFrame()

    h5_path = _get_ex_cum_factor_path()
    if not os.path.exists(h5_path):
        return pd.DataFrame()

    all_dfs = []
    with h5py.File(h5_path, "r") as f:
        for code in codes:
            if code not in f:
                continue
            data = f[code][:]
            if len(data) == 0:
                continue
            df = pd.DataFrame(data)
            df["start_date"] = df["start_date"].apply(_factor_int_to_timestamp)
            df["order_book_id"] = code
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    result = result.set_index(["order_book_id", "start_date"]).sort_index()

    if "ex_cum_factor" in result.columns:
        result["ex_cum_factor"] = result["ex_cum_factor"].astype("float32")

    return result


def get_split_factor(order_book_ids):
    """
    获取拆分因子（送股/转增）

    Parameters
    ----------
    order_book_ids : str or list[str]
        股票代码

    Returns
    -------
    pd.DataFrame
        MultiIndex: [order_book_id, ex_date], columns: [split_factor, split_coefficient_to, split_coefficient_from]
    """
    codes = normalize_codes(order_book_ids)
    if not codes:
        return pd.DataFrame()

    h5_path = _get_split_factor_path()
    if not os.path.exists(h5_path):
        return pd.DataFrame()

    all_dfs = []
    with h5py.File(h5_path, "r") as f:
        for code in codes:
            if code not in f:
                continue
            data = f[code][:]
            if len(data) == 0:
                continue
            df = pd.DataFrame(data)
            df["ex_date"] = df["ex_date"].apply(_factor_int_to_timestamp)
            df["order_book_id"] = code
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    result = result.set_index(["order_book_id", "ex_date"]).sort_index()

    for col in ["split_factor", "split_coefficient_to", "split_coefficient_from"]:
        if col in result.columns:
            result[col] = result[col].astype("float32")

    return result
