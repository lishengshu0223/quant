import os
import pandas as pd
from .config import get_data_path, TRADING_DATES_DIR
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


def run_with_exception_handling(func, *args, **kwargs):
    func_name = func.__name__
    try:
        result = func(*args, **kwargs)
        logger.info(f"SUCCESS: {func_name} completed")
        return result
    except Exception as e:
        logger.error(f"FAILED: {func_name} - {str(e)}")
        return None