import os
import json
import pandas as pd
from .config import (
    get_data_path,
    BUNDLE_DIR,
    BUNDLE_BUNDLE_DIR,
    FUTURE_INFO_JSON,
)

_FUTURE_INFO_CACHE = None


def _get_future_info_path():
    return get_data_path(BUNDLE_DIR, BUNDLE_BUNDLE_DIR, FUTURE_INFO_JSON)


def _load_future_info():
    global _FUTURE_INFO_CACHE
    if _FUTURE_INFO_CACHE is not None:
        return _FUTURE_INFO_CACHE

    filepath = _get_future_info_path()
    if not os.path.exists(filepath):
        _FUTURE_INFO_CACHE = []
        return _FUTURE_INFO_CACHE

    with open(filepath, "r") as f:
        _FUTURE_INFO_CACHE = json.load(f)

    return _FUTURE_INFO_CACHE


def get_future_info(underlying_symbol=None):
    """
    获取期货合约信息（手续费率、保证金率、最小变动价位等）

    Parameters
    ----------
    underlying_symbol : str or list[str], optional
        期货品种代码，如 'IF', 'AU', 'RB'。默认返回全部

    Returns
    -------
    pd.DataFrame
        index为underlying_symbol, columns包含:
        commission_type, open_commission_ratio, close_commission_ratio,
        close_commission_today_ratio, margin_rate, tick_size
    """
    data = _load_future_info()
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    if underlying_symbol is not None:
        if isinstance(underlying_symbol, str):
            underlying_symbol = [underlying_symbol]
        df = df[df["underlying_symbol"].isin(underlying_symbol)]

    df = df.set_index("underlying_symbol").sort_index()

    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")

    return df
