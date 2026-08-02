import os
import h5py
import numpy as np
import pandas as pd
from .config import (
    get_data_path,
    STOCK_MINUTE_DIR,
    BUNDLE_H5_DIR,
    EQUITIES_H5_DIR,
    STOCK_MINUTE_FIELDS,
    MINUTE_FREQUENCIES,
)
from ._utils import normalize_date, normalize_codes


def _get_equities_minute_dir():
    return get_data_path(STOCK_MINUTE_DIR, BUNDLE_H5_DIR, EQUITIES_H5_DIR, "equities")


def _get_ex_cum_factor_path():
    return get_data_path(STOCK_MINUTE_DIR, "ex_cum_factor.h5")


def _get_split_factor_path():
    return get_data_path(STOCK_MINUTE_DIR, "bundle", "split_factor.h5")


def _datetime_int_to_timestamp(dt_int):
    dt_str = str(int(dt_int))
    return pd.Timestamp(
        year=int(dt_str[:4]),
        month=int(dt_str[4:6]),
        day=int(dt_str[6:8]),
        hour=int(dt_str[8:10]),
        minute=int(dt_str[10:12]),
        second=int(dt_str[12:14]) if len(dt_str) > 12 else 0,
    )


def _date_int_to_timestamp(date_int):
    date_str = str(int(date_int))
    return pd.Timestamp(
        year=int(date_str[:4]),
        month=int(date_str[4:6]),
        day=int(date_str[6:8]),
    )


def _load_single_stock_minute(code, start_date, end_date):
    """加载单只股票的指定日期范围的分钟数据"""
    filepath = os.path.join(_get_equities_minute_dir(), f"{code}.h5")
    if not os.path.exists(filepath):
        return pd.DataFrame()

    start_dt = normalize_date(start_date)
    end_dt = normalize_date(end_date)

    with h5py.File(filepath, "r") as f:
        index_data = f["index"][:]
        data_array = f["data"]

        dates = index_data["date"]
        line_nos = index_data["line_no"]

        # 转换为datetime便于比较
        start_date_int = int(start_dt.strftime("%Y%m%d")) if start_dt else -1
        end_date_int = int(end_dt.strftime("%Y%m%d")) if end_dt else 99999999

        # 找到起始和结束日期在index中的位置
        mask = (dates >= start_date_int) & (dates <= end_date_int)
        if not np.any(mask):
            return pd.DataFrame()

        selected_indices = np.where(mask)[0]
        start_idx = selected_indices[0]
        end_idx = selected_indices[-1]

        start_line = int(line_nos[start_idx])
        if end_idx + 1 < len(line_nos):
            end_line = int(line_nos[end_idx + 1])
        else:
            end_line = data_array.shape[0]

        # 读取数据
        raw_data = data_array[start_line:end_line]

        if len(raw_data) == 0:
            return pd.DataFrame()

        # 转换为DataFrame
        df = pd.DataFrame(raw_data)

        # 转换datetime
        df["datetime"] = df["datetime"].apply(_datetime_int_to_timestamp)

        # 设置order_book_id
        df["order_book_id"] = code

        return df


def _aggregate_frequency(df, frequency):
    """将1分钟数据聚合为更高频率的K线，按A股交易时段对齐"""
    if frequency == "1m":
        return df

    window_map = {
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "60m": 60,
    }
    window = window_map.get(frequency)
    if window is None:
        raise ValueError(f"Unsupported frequency: {frequency}")

    agg_funcs = {
        "open": lambda x: x.iloc[0],
        "high": lambda x: x.max(),
        "low": lambda x: x.min(),
        "close": lambda x: x.iloc[-1],
        "volume": lambda x: x.sum(),
        "total_turnover": lambda x: x.sum(),
        "num_trades": lambda x: x.sum(),
    }
    # 只保留存在的字段
    agg_funcs = {k: v for k, v in agg_funcs.items() if k in df.columns}

    result_list = []

    for (order_book_id, date), group in df.groupby(["order_book_id", df["datetime"].dt.date]):
        if group.empty:
            continue

        group = group.sort_values("datetime").reset_index(drop=True)

        # 区分上午和下午时段
        # 上午: 09:31 ~ 11:30 (120分钟)
        # 下午: 13:01 ~ 15:00 (120分钟)
        morning_mask = group["datetime"].dt.hour < 12
        afternoon_mask = group["datetime"].dt.hour >= 13

        for session_mask in [morning_mask, afternoon_mask]:
            session = group[session_mask].copy()
            if session.empty:
                continue

            session = session.reset_index(drop=True)
            n = len(session)

            # 按window大小分组聚合
            start_indices = list(range(0, n, window))
            for start in start_indices:
                end = min(start + window, n)
                chunk = session.iloc[start:end]
                if chunk.empty:
                    continue

                # rqdatac对>1m频率使用结束时间作为标签(label=right)
                # 例如5m第一根覆盖09:31-09:35, 标签为09:35
                row = {"datetime": chunk["datetime"].iloc[-1]}
                for col, func in agg_funcs.items():
                    row[col] = func(chunk[col])
                row["order_book_id"] = order_book_id
                result_list.append(row)

    if not result_list:
        return pd.DataFrame(columns=["order_book_id", "datetime"] + list(agg_funcs.keys()))

    result = pd.DataFrame(result_list)
    return result


def _factor_int_to_timestamp(dt_int):
    """转换复权因子的日期整数，处理0（表示最早有效日期）的特殊情况"""
    if dt_int == 0:
        return pd.Timestamp("1990-01-01")
    dt_str = str(int(dt_int))
    # ex_cum_factor的start_date格式可能是YYYYMMDD000000或纯日期
    if len(dt_str) >= 8:
        return pd.Timestamp(
            year=int(dt_str[:4]),
            month=int(dt_str[4:6]),
            day=int(dt_str[6:8]),
        )
    return pd.Timestamp(str(dt_int))


def _load_ex_cum_factors(code):
    """加载单只股票的累计复权因子"""
    filepath = _get_ex_cum_factor_path()
    if not os.path.exists(filepath):
        return pd.DataFrame()

    with h5py.File(filepath, "r") as f:
        if code not in f:
            return pd.DataFrame()
        data = f[code][:]
        df = pd.DataFrame(data)
        df["start_date"] = df["start_date"].apply(_factor_int_to_timestamp)
        return df


def _load_split_factors(code):
    """加载单只股票的拆分因子"""
    filepath = _get_split_factor_path()
    if not os.path.exists(filepath):
        return pd.DataFrame()

    with h5py.File(filepath, "r") as f:
        if code not in f:
            return pd.DataFrame()
        data = f[code][:]
        df = pd.DataFrame(data)
        df["ex_date"] = df["ex_date"].apply(_factor_int_to_timestamp)
        return df


def _get_ex_cum_factor_at(series, timestamp):
    """获取某个时间点的累计复权因子"""
    if series.empty:
        return 1.0
    factors = series.sort_values("start_date")
    valid = factors[factors["start_date"] <= timestamp]
    if valid.empty:
        return 1.0
    return float(valid.iloc[-1]["ex_cum_factor"])


def _get_split_factor_at(series, timestamp):
    """获取某个时间点的累计拆分因子（乘积）"""
    if series.empty:
        return 1.0
    factors = series.sort_values("ex_date")
    valid = factors[factors["ex_date"] <= timestamp]
    if valid.empty:
        return 1.0
    return float(valid["split_factor"].prod())


def _apply_adjust(df, adjust_type):
    """对分钟数据应用复权"""
    if adjust_type == "none":
        return df

    if df.empty:
        return df

    result_df = df.copy()
    codes = result_df["order_book_id"].unique()

    price_fields = ["open", "high", "low", "close"]

    for code in codes:
        mask = result_df["order_book_id"] == code
        code_rows = result_df.loc[mask]

        # 加载复权因子
        ex_cum_factors = _load_ex_cum_factors(code)
        split_factors = _load_split_factors(code)

        # pre复权以全局最新因子为基准（不是请求范围内的最新）
        latest_ex_cum = _get_ex_cum_factor_at(ex_cum_factors, pd.Timestamp.max)
        latest_split = _get_split_factor_at(split_factors, pd.Timestamp.max)

        # post复权以全局最早因子为基准（不是请求范围内的最早）
        earliest_ex_cum = _get_ex_cum_factor_at(ex_cum_factors, pd.Timestamp.min)
        earliest_split = _get_split_factor_at(split_factors, pd.Timestamp.min)

        # 逐行计算复权
        datetime_series = code_rows["datetime"]
        ex_cum_series = datetime_series.apply(
            lambda dt: _get_ex_cum_factor_at(ex_cum_factors, dt)
        )
        split_series = datetime_series.apply(
            lambda dt: _get_split_factor_at(split_factors, dt)
        )

        if adjust_type in ("pre", "pre_volume"):
            # 前复权: 价格 = 原始价 * ex_cum / latest_ex_cum
            factor_series = ex_cum_series.values / latest_ex_cum

            for field in price_fields:
                if field in result_df.columns:
                    result_df.loc[mask, field] = (
                        result_df.loc[mask, field].values * factor_series
                    ).astype("float32")

            if adjust_type == "pre":
                # pre模式: volume用拆分因子调整
                volume_factor = earliest_split / split_series.values
            else:
                # pre_volume模式: volume用复权因子调整
                volume_factor = earliest_ex_cum / ex_cum_series.values

            if "volume" in result_df.columns:
                result_df.loc[mask, "volume"] = (
                    result_df.loc[mask, "volume"].values * volume_factor
                ).astype("float32")

        elif adjust_type in ("post", "post_volume"):
            # 后复权: 价格 = 原始价 * ex_cum / earliest_ex_cum
            factor_series = ex_cum_series.values / earliest_ex_cum

            for field in price_fields:
                if field in result_df.columns:
                    result_df.loc[mask, field] = (
                        result_df.loc[mask, field].values * factor_series
                    ).astype("float32")

            if adjust_type == "post":
                # post模式: volume用拆分因子调整
                volume_factor = latest_split / split_series.values
            else:
                # post_volume模式: volume用复权因子调整
                volume_factor = latest_ex_cum / ex_cum_series.values

            if "volume" in result_df.columns:
                result_df.loc[mask, "volume"] = (
                    result_df.loc[mask, "volume"].values * volume_factor
                ).astype("float32")

    # 转换float64为float32节省内存
    for col in ["open", "high", "low", "close", "volume", "total_turnover", "num_trades"]:
        if col in result_df.columns and result_df[col].dtype == np.float64:
            result_df[col] = result_df[col].astype("float32")

    return result_df


def _apply_time_slice(df, time_slice):
    """按时间段切分数据"""
    if time_slice is None or df.empty:
        return df

    start_time_str, end_time_str = time_slice
    start_h, start_m = map(int, start_time_str.split(":"))
    end_h, end_m = map(int, end_time_str.split(":"))

    start_total = start_h * 60 + start_m
    end_total = end_h * 60 + end_m

    datetime_col = df["datetime"]
    time_minutes = datetime_col.dt.hour * 60 + datetime_col.dt.minute

    mask = (time_minutes >= start_total) & (time_minutes <= end_total)
    return df[mask].copy()


def get_minute_price(
    order_book_ids,
    start_date=None,
    end_date=None,
    frequency="1m",
    fields=None,
    adjust_type="none",
    skip_suspended=True,
    market="cn",
    expect_df=True,
    time_slice=None,
):
    """
    获取股票分钟级K线数据

    Parameters
    ----------
    order_book_ids : str or list[str]
        合约代码或代码列表
    start_date : str or pd.Timestamp, optional
        开始日期
    end_date : str or pd.Timestamp, optional
        结束日期
    frequency : str, default "1m"
        频率: "1m", "5m", "15m", "30m", "60m"
    fields : str or list[str], optional
        需要的字段，默认全部: ["open", "high", "low", "close", "volume", "total_turnover", "num_trades"]
    adjust_type : str, default "none"
        复权方式: "none", "pre", "post", "pre_volume", "post_volume"
    skip_suspended : bool, default True
        是否跳过停牌数据（停牌期间为NaN）
    market : str, default "cn"
        市场
    expect_df : bool, default True
        是否返回DataFrame
    time_slice : tuple(str, str), optional
        时间段切分，如 ("09:31", "10:00")

    Returns
    -------
    pd.DataFrame
        MultiIndex: [order_book_id, datetime], columns为指定fields
    """
    # 验证frequency
    if frequency not in MINUTE_FREQUENCIES:
        raise ValueError(
            f"Unsupported frequency '{frequency}'. Supported: {MINUTE_FREQUENCIES}"
        )

    # 验证adjust_type
    valid_adjust = ["none", "pre", "post", "pre_volume", "post_volume"]
    if adjust_type not in valid_adjust:
        raise ValueError(
            f"Unsupported adjust_type '{adjust_type}'. Supported: {valid_adjust}"
        )

    # 标准化代码
    codes = normalize_codes(order_book_ids)
    if not codes:
        return pd.DataFrame()

    # 标准化日期
    start_dt = normalize_date(start_date)
    end_dt = normalize_date(end_date)

    # 默认fields
    if fields is None:
        fields = STOCK_MINUTE_FIELDS.copy()
    elif isinstance(fields, str):
        fields = [fields]

    # 检查目录
    base_dir = _get_equities_minute_dir()
    if not os.path.exists(base_dir):
        return pd.DataFrame()

    # 逐只股票加载数据
    all_dfs = []
    for code in codes:
        df = _load_single_stock_minute(code, start_dt, end_dt)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # 先复权（在1分钟级别上）
    combined_df = _apply_adjust(combined_df, adjust_type)

    # 应用时间段切分（在聚合之前）
    combined_df = _apply_time_slice(combined_df, time_slice)

    if combined_df.empty:
        return pd.DataFrame()

    # 频率聚合
    combined_df = _aggregate_frequency(combined_df, frequency)

    if combined_df.empty:
        return pd.DataFrame()

    # 处理停牌数据: 如果volume和total_turnover都为0则视为停牌
    if skip_suspended:
        suspended_mask = (combined_df["volume"] == 0) & (combined_df["total_turnover"] == 0)
        combined_df = combined_df[~suspended_mask].copy()

    # 设置MultiIndex
    combined_df = combined_df.set_index(["order_book_id", "datetime"]).sort_index()

    # 只保留指定fields
    valid_fields = [f for f in fields if f in combined_df.columns]
    if valid_fields:
        combined_df = combined_df[valid_fields]

    if expect_df:
        return combined_df
    return combined_df.unstack("order_book_id")
