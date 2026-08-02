import os
import h5py
import numpy as np
import pandas as pd
from .config import (
    get_data_path,
    STOCK_MINUTE_DIR,
    BUNDLE_H5_DIR,
    EQUITIES_H5_DIR,
    BUNDLE_DIR,
    BUNDLE_BUNDLE_DIR,
    FUTURES_H5,
    FUTURES_DAILY_FIELDS,
    FUTURES_MINUTE_FIELDS,
    MINUTE_FREQUENCIES,
)
from ._utils import normalize_date


def _get_futures_h5_path():
    return get_data_path(BUNDLE_DIR, BUNDLE_BUNDLE_DIR, FUTURES_H5)


def _get_future_minute_dir():
    return get_data_path(STOCK_MINUTE_DIR, BUNDLE_H5_DIR, EQUITIES_H5_DIR, "future")


def _datetime_int_to_timestamp(dt_int):
    dt_str = str(int(dt_int))
    return pd.Timestamp(
        year=int(dt_str[:4]),
        month=int(dt_str[4:6]),
        day=int(dt_str[6:8]),
        hour=int(dt_str[8:10]) if len(dt_str) > 8 else 0,
        minute=int(dt_str[10:12]) if len(dt_str) > 10 else 0,
        second=int(dt_str[12:14]) if len(dt_str) > 12 else 0,
    )


def _date_int_to_timestamp(date_int):
    s = str(int(date_int))
    return pd.Timestamp(year=int(s[:4]), month=int(s[4:6]), day=int(s[6:8]))


def get_future_price(
    order_book_ids,
    start_date=None,
    end_date=None,
    frequency="1d",
    fields=None,
    expect_df=True,
):
    """
    获取期货K线数据（日线或分钟线）

    Parameters
    ----------
    order_book_ids : str or list[str]
        期货合约代码，如 'IF2401', 'IF88'（主力连续）
    start_date : str or pd.Timestamp, optional
        开始日期
    end_date : str or pd.Timestamp, optional
        结束日期
    frequency : str, default "1d"
        频率: "1d"（日线）, "1m", "5m", "15m", "30m", "60m"（分钟线）
    fields : str or list[str], optional
        需要的字段
    expect_df : bool, default True
        是否返回DataFrame

    Returns
    -------
    pd.DataFrame
        MultiIndex: [order_book_id, datetime]
    """
    if isinstance(order_book_ids, str):
        codes = [order_book_ids]
    else:
        codes = list(order_book_ids)

    start_dt = normalize_date(start_date)
    end_dt = normalize_date(end_date)
    start_date_int = int(start_dt.strftime("%Y%m%d")) if start_dt else -1
    end_date_int = int(end_dt.strftime("%Y%m%d")) if end_dt else 99999999

    if frequency == "1d":
        return _get_future_daily(codes, start_date_int, end_date_int, fields, expect_df)
    elif frequency in MINUTE_FREQUENCIES:
        return _get_future_minute(codes, start_date_int, end_date_int, frequency, fields, expect_df)
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")


def _get_future_daily(codes, start_date_int, end_date_int, fields, expect_df):
    h5_path = _get_futures_h5_path()
    if not os.path.exists(h5_path):
        return pd.DataFrame()

    if fields is None:
        fields = FUTURES_DAILY_FIELDS.copy()

    all_dfs = []
    with h5py.File(h5_path, "r") as f:
        for code in codes:
            if code not in f:
                continue
            data = f[code][:]
            df = pd.DataFrame(data)
            df["datetime"] = df["datetime"].apply(_datetime_int_to_timestamp)

            mask = (df["datetime"] >= pd.Timestamp("1900-01-01")) & (df["datetime"] <= pd.Timestamp("2100-01-01"))
            if start_date_int > 0:
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


def _get_future_minute(codes, start_date_int, end_date_int, frequency, fields, expect_df):
    minute_dir = _get_future_minute_dir()
    if not os.path.exists(minute_dir):
        return pd.DataFrame()

    if fields is None:
        fields = FUTURES_MINUTE_FIELDS.copy()

    all_dfs = []
    for code in codes:
        filepath = os.path.join(minute_dir, f"{code}.h5")
        if not os.path.exists(filepath):
            continue

        with h5py.File(filepath, "r") as f:
            if "data" not in f or "index" not in f:
                continue

            index_data = f["index"][:]
            data_array = f["data"]

            dates = index_data["date"]
            line_nos = index_data["line_no"]

            mask = (dates >= start_date_int) & (dates <= end_date_int)
            if not np.any(mask):
                continue

            selected = np.where(mask)[0]
            start_line = int(line_nos[selected[0]])
            if selected[-1] + 1 < len(line_nos):
                end_line = int(line_nos[selected[-1] + 1])
            else:
                end_line = data_array.shape[0]

            raw = data_array[start_line:end_line]
            if len(raw) == 0:
                continue

            df = pd.DataFrame(raw)
            df["datetime"] = df["datetime"].apply(_datetime_int_to_timestamp)
            df["order_book_id"] = code
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    if frequency != "1m":
        combined = _aggregate_future_frequency(combined, frequency)

    if combined.empty:
        return pd.DataFrame()

    combined = combined.set_index(["order_book_id", "datetime"]).sort_index()

    valid_fields = [f for f in fields if f in combined.columns]
    if valid_fields:
        combined = combined[valid_fields]

    for col in combined.columns:
        if combined[col].dtype == np.float64:
            combined[col] = combined[col].astype("float32")

    if expect_df:
        return combined
    return combined.unstack("order_book_id")


def _aggregate_future_frequency(df, frequency):
    """将1分钟期货数据聚合为更高频率，使用结束时间标签(label=right)"""
    window_map = {"5m": 5, "15m": 15, "30m": 30, "60m": 60}
    window = window_map.get(frequency)
    if window is None:
        return df

    agg_funcs = {
        "open": lambda x: x.iloc[0],
        "high": lambda x: x.max(),
        "low": lambda x: x.min(),
        "close": lambda x: x.iloc[-1],
        "volume": lambda x: x.sum(),
        "total_turnover": lambda x: x.sum(),
        "open_interest": lambda x: x.iloc[-1],
    }
    agg_funcs = {k: v for k, v in agg_funcs.items() if k in df.columns}

    result_list = []
    for (code, date), group in df.groupby(["order_book_id", df["datetime"].dt.date]):
        if group.empty:
            continue
        group = group.sort_values("datetime").reset_index(drop=True)

        n = len(group)
        start_indices = list(range(0, n, window))
        for start in start_indices:
            end = min(start + window, n)
            chunk = group.iloc[start:end]
            if chunk.empty:
                continue
            row = {"datetime": chunk["datetime"].iloc[-1], "order_book_id": code}
            for col, func in agg_funcs.items():
                row[col] = func(chunk[col])
            result_list.append(row)

    if not result_list:
        return pd.DataFrame(columns=["order_book_id", "datetime"] + list(agg_funcs.keys()))
    return pd.DataFrame(result_list)
