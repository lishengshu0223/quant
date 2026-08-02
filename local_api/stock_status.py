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
)
from ._utils import normalize_codes, normalize_date


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
