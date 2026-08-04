import os
import h5py
import numpy as np
import pandas as pd
from .config import (
    get_data_path,
    BUNDLE_DIR,
    BUNDLE_BUNDLE_DIR,
    ST_STOCK_DAYS_H5,
    SUSPENDED_DAYS_H5,
    TRADABLE_STATUS_DIR,
)
from ._utils import normalize_codes, normalize_date, filter_dates_by_range


def _get_st_stock_days_path():
    return get_data_path(BUNDLE_DIR, BUNDLE_BUNDLE_DIR, ST_STOCK_DAYS_H5)


def _get_suspended_days_path():
    return get_data_path(BUNDLE_DIR, BUNDLE_BUNDLE_DIR, SUSPENDED_DAYS_H5)


def is_st_stock(order_book_ids, start_date=None, end_date=None, market="cn"):
    """
    判断股票在指定日期范围内是否为ST股

    Parameters
    ----------
    order_book_ids : str or list[str]
        股票代码
    start_date : str or pd.Timestamp, optional
        开始日期
    end_date : str or pd.Timestamp, optional
        结束日期

    Returns
    -------
    pd.DataFrame
        MultiIndex: [order_book_id, date], columns: [is_st]
    """
    codes = normalize_codes(order_book_ids)
    if not codes:
        return pd.DataFrame()

    start_dt = normalize_date(start_date)
    end_dt = normalize_date(end_date)
    start_int = int(start_dt.strftime("%Y%m%d")) if start_dt else -1
    end_int = int(end_dt.strftime("%Y%m%d")) if end_dt else 99999999

    h5_path = _get_st_stock_days_path()
    if not os.path.exists(h5_path):
        return pd.DataFrame()

    all_dfs = []
    with h5py.File(h5_path, "r") as f:
        for code in codes:
            if code not in f:
                # 没有ST记录的股票，说明从未被ST
                continue
            data = f[code][:]
            if len(data) == 0:
                continue

            dates = data.astype(int)
            mask = (dates >= start_int) & (dates <= end_int)
            filtered = dates[mask]
            if len(filtered) == 0:
                continue

            df = pd.DataFrame({
                "date": [pd.Timestamp(str(d)) for d in filtered],
                "is_st": True,
                "order_book_id": code,
            })
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame(columns=["is_st"])

    result = pd.concat(all_dfs, ignore_index=True)
    result = result.set_index(["order_book_id", "date"]).sort_index()
    return result


def is_suspended(order_book_ids, start_date=None, end_date=None, market="cn"):
    """
    判断股票在指定日期范围内是否停牌

    Parameters
    ----------
    order_book_ids : str or list[str]
        股票代码
    start_date : str or pd.Timestamp, optional
        开始日期
    end_date : str or pd.Timestamp, optional
        结束日期

    Returns
    -------
    pd.DataFrame
        MultiIndex: [order_book_id, date], columns: [is_suspended]
    """
    codes = normalize_codes(order_book_ids)
    if not codes:
        return pd.DataFrame()

    start_dt = normalize_date(start_date)
    end_dt = normalize_date(end_date)
    start_int = int(start_dt.strftime("%Y%m%d")) if start_dt else -1
    end_int = int(end_dt.strftime("%Y%m%d")) if end_dt else 99999999

    h5_path = _get_suspended_days_path()
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

            dates = data.astype(int)
            mask = (dates >= start_int) & (dates <= end_int)
            filtered = dates[mask]
            if len(filtered) == 0:
                continue

            df = pd.DataFrame({
                "date": [pd.Timestamp(str(d)) for d in filtered],
                "is_suspended": True,
                "order_book_id": code,
            })
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame(columns=["is_suspended"])

    result = pd.concat(all_dfs, ignore_index=True)
    result = result.set_index(["order_book_id", "date"]).sort_index()
    return result


def get_tradable_matrix(start_date, end_date):
    """
    获取指定日期范围内全市场的每日可交易状态矩阵

    数据来源: F:\\Trade_data\\tradable_status\\ 下按交易日保存的 parquet 文件
    （由 update/tradable_status 更新模块生成）

    Parameters
    ----------
    start_date : str or pd.Timestamp
        开始日期
    end_date : str or pd.Timestamp
        结束日期

    Returns
    -------
    pd.DataFrame
        index: 日期（已保存数据的交易日）
        columns: 全市场股票 order_book_id
        值:
          - True   当天可交易
          - False  当天不可交易（ST / *ST、停牌、上市未满一年、涨跌停含一字板）
          - NaN    当天该股票尚未上市或已退市
    """
    start_dt = normalize_date(start_date)
    end_dt = normalize_date(end_date)

    base_dir = get_data_path(TRADABLE_STATUS_DIR)
    if not os.path.exists(base_dir):
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date"), columns=[])

    date_files = []
    for fname in os.listdir(base_dir):
        if fname.endswith(".parquet"):
            date_files.append((fname.replace(".parquet", ""), os.path.join(base_dir, fname)))
    date_files.sort(key=lambda x: x[0])
    date_files = filter_dates_by_range(date_files, start_dt, end_dt)
    if not date_files:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date"), columns=[])

    dates = [pd.Timestamp(d) for d, _ in date_files]
    dfs = [pd.read_parquet(path) for _, path in date_files]

    # 全市场股票 = 范围内所有出现过（上市过）的股票并集
    all_codes = sorted(set().union(*[set(df["code"]) for df in dfs]))

    matrix = pd.DataFrame(index=pd.DatetimeIndex(dates, name="date"), columns=all_codes, dtype=object)
    for d, df in zip(dates, dfs):
        tradable = df.set_index("code")["tradable"]
        matrix.loc[d, tradable.index] = tradable.values
    return matrix
