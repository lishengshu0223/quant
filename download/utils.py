import os
import pandas as pd
from .config import get_data_path, TRADING_DATES_DIR, START_DATE
from .logger import logger


def format_date(date):
    if isinstance(date, pd.Timestamp):
        return date.strftime("%Y%m%d")
    return str(date).replace("-", "")


def get_existing_dates(base_dir):
    existing = []
    if os.path.exists(base_dir):
        for filename in os.listdir(base_dir):
            if filename.endswith(".parquet"):
                date_str = filename.replace(".parquet", "")
                if len(date_str) == 8:
                    existing.append(date_str)
    return sorted(existing)


def get_missing_dates(trading_dates, existing_dates):
    trading_dates_str = [format_date(d) for d in trading_dates]
    missing = sorted(set(trading_dates_str) - set(existing_dates))
    return missing


def load_trading_dates():
    filepath = get_data_path(TRADING_DATES_DIR, "trading_dates.parquet")
    if os.path.exists(filepath):
        df = pd.read_parquet(filepath)
        return df["date"].tolist()
    return []


def save_parquet(df, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_parquet(filepath, engine="pyarrow")


def save_snapshot(df, filepath):
    """按日快照统一落盘：float64 压为 float32 后写 parquet（各下载模块沿用同一规格）"""
    df = df.astype({col: 'float32' for col in df.select_dtypes(include=['float64']).columns})
    return save_parquet(df, filepath)


def get_stock_universe():
    """全市场 A 股 order_book_id 列表（rqdatac all_instruments 快照，剔除 B 股/北交所等）"""
    from rqdatac import all_instruments
    df = all_instruments(type="CS", market="cn")
    return df[df["type"] == "CS"]["order_book_id"].tolist()


def resolve_target_dates(base_dir, download_dates=None, force=False, end_date=None):
    """统一解析待下载日期列表（YYYYMMDD 字符串，顺序与输入/交易日历一致）

    - download_dates 为空：起始 START_DATE 至 end_date（默认今天）的全部交易日，
      force=True 时全部重下，否则减去已存在文件对应的日期（只补缺口）；
    - download_dates 非空：按给定列表（force=True）或过滤已存在日期。

    base_dir 为 None 时不参与已存在日期过滤（视为无历史）。
    """
    from rqdatac import get_trading_dates

    existing = set(get_existing_dates(base_dir)) if base_dir else set()
    if download_dates is None:
        end = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp.now().date()
        trading_dates = get_trading_dates(pd.Timestamp(START_DATE), end, market="cn")
        target = [format_date(d) for d in trading_dates]
        if force:
            return target
        return [d for d in target if d not in existing]
    target = [format_date(d) for d in download_dates]
    if force:
        return target
    return [d for d in target if d not in existing]


def run_with_exception_handling(func, *args, **kwargs):
    func_name = func.__name__
    try:
        result = func(*args, **kwargs)
        logger.info(f"SUCCESS: {func_name} completed")
        return result
    except Exception as e:
        logger.error(f"FAILED: {func_name} - {str(e)}")
        return None