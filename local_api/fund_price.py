import os
import h5py
import numpy as np
import pandas as pd
from .config import (
    get_data_path,
    BUNDLE_DIR,
    BUNDLE_BUNDLE_DIR,
    FUNDS_H5,
    FUNDS_DAILY_FIELDS,
)
from ._utils import normalize_date


def _get_funds_h5_path():
    return get_data_path(BUNDLE_DIR, BUNDLE_BUNDLE_DIR, FUNDS_H5)


def _datetime_int_to_timestamp(dt_int):
    dt_str = str(int(dt_int))
    return pd.Timestamp(
        year=int(dt_str[:4]),
        month=int(dt_str[4:6]),
        day=int(dt_str[6:8]),
    )


def get_fund_price(
    order_book_ids,
    start_date=None,
    end_date=None,
    frequency="1d",
    fields=None,
    expect_df=True,
):
    """
    获取基金日线K线数据

    Parameters
    ----------
    order_book_ids : str or list[str]
        基金合约代码，如 '510050.XSHG', '159919.XSHE'
    start_date : str or pd.Timestamp, optional
        开始日期
    end_date : str or pd.Timestamp, optional
        结束日期
    frequency : str, default "1d"
        频率（仅支持日线 "1d"）
    fields : str or list[str], optional
        需要的字段
    expect_df : bool, default True
        是否返回DataFrame

    Returns
    -------
    pd.DataFrame
        MultiIndex: [order_book_id, date]
    """
    if isinstance(order_book_ids, str):
        codes = [order_book_ids]
    else:
        codes = list(order_book_ids)

    start_dt = normalize_date(start_date)
    end_dt = normalize_date(end_date)
    start_date_int = int(start_dt.strftime("%Y%m%d")) if start_dt else -1
    end_date_int = int(end_dt.strftime("%Y%m%d")) if end_dt else 99999999

    if fields is None:
        fields = FUNDS_DAILY_FIELDS.copy()

    h5_path = _get_funds_h5_path()
    if not os.path.exists(h5_path):
        return pd.DataFrame()

    all_dfs = []
    with h5py.File(h5_path, "r") as f:
        for code in codes:
            if code not in f:
                continue
            data = f[code][:]
            df = pd.DataFrame(data)
            df["datetime"] = df["datetime"].apply(_datetime_int_to_timestamp)

            dt_ints = df["datetime"].apply(lambda x: int(x.strftime("%Y%m%d")))
            mask = (dt_ints >= start_date_int) & (dt_ints <= end_date_int)
            df = df[mask]

            df["order_book_id"] = code
            df = df.rename(columns={"datetime": "date"})
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    result = result.set_index(["order_book_id", "date"]).sort_index()

    valid_fields = [f for f in fields if f in result.columns]
    if valid_fields:
        result = result[valid_fields]

    for col in result.columns:
        if result[col].dtype == np.float64:
            result[col] = result[col].astype("float32")

    if expect_df:
        return result
    return result.unstack("order_book_id")
