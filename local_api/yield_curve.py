import os
import h5py
import numpy as np
import pandas as pd
from .config import (
    get_data_path,
    BUNDLE_DIR,
    BUNDLE_BUNDLE_DIR,
    YIELD_CURVE_H5,
    YIELD_CURVE_TENORS,
)
from ._utils import normalize_date


def _get_yield_curve_path():
    return get_data_path(BUNDLE_DIR, BUNDLE_BUNDLE_DIR, YIELD_CURVE_H5)


def get_yield_curve(start_date=None, end_date=None, tenor=None):
    """
    获取国债收益率曲线

    Parameters
    ----------
    start_date : str or pd.Timestamp, optional
        开始日期
    end_date : str or pd.Timestamp, optional
        结束日期
    tenor : str or list[str], optional
        期限，如 '1Y', '10Y'，默认返回全部

    Returns
    -------
    pd.DataFrame
        index: date, columns: 各期限收益率
    """
    start_dt = normalize_date(start_date)
    end_dt = normalize_date(end_date)
    start_int = int(start_dt.strftime("%Y%m%d")) if start_dt else -1
    end_int = int(end_dt.strftime("%Y%m%d")) if end_dt else 99999999

    h5_path = _get_yield_curve_path()
    if not os.path.exists(h5_path):
        return pd.DataFrame()

    with h5py.File(h5_path, "r") as f:
        if "data" not in f:
            return pd.DataFrame()
        data = f["data"][:]
        df = pd.DataFrame(data)

    # 日期转换
    df["date"] = df["date"].apply(lambda x: pd.Timestamp(str(int(x))))

    # 日期过滤
    dt_ints = df["date"].apply(lambda x: int(x.strftime("%Y%m%d")))
    mask = (dt_ints >= start_int) & (dt_ints <= end_int)
    df = df[mask]

    df = df.set_index("date").sort_index()

    # 选择期限
    if tenor is not None:
        if isinstance(tenor, str):
            tenor = [tenor]
        valid_tenors = [t for t in tenor if t in df.columns]
        if valid_tenors:
            df = df[valid_tenors]
    else:
        valid_tenors = [t for t in YIELD_CURVE_TENORS if t in df.columns]
        if valid_tenors:
            df = df[valid_tenors]

    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype("float32")

    return df
