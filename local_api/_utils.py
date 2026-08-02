import os
import pandas as pd
from .config import INDEX_FULL_CODES, get_data_path, TRADING_DATES_DIR


def format_date(date):
    if isinstance(date, pd.Timestamp):
        return date.strftime("%Y%m%d")
    if isinstance(date, str):
        return date.replace("-", "")
    return str(date)


def normalize_date(date):
    if date is None:
        return None
    if isinstance(date, pd.Timestamp):
        return date
    if isinstance(date, str):
        if len(date) == 8:
            return pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:8]}")
        return pd.Timestamp(date)
    return pd.Timestamp(date)


def normalize_code(code):
    if code is None:
        return None
    if isinstance(code, str):
        if "." in code:
            return code
        if code in INDEX_FULL_CODES:
            return INDEX_FULL_CODES[code]
        if len(code) == 6:
            if code.startswith("6"):
                return f"{code}.XSHG"
            else:
                return f"{code}.XSHE"
    return code


def normalize_codes(codes):
    if codes is None:
        return None
    if isinstance(codes, str):
        return [normalize_code(codes)]
    return [normalize_code(c) for c in codes]


def short_code(code):
    if code is None:
        return None
    if isinstance(code, str) and "." in code:
        return code.split(".")[0]
    return code


def short_codes(codes):
    if codes is None:
        return None
    if isinstance(codes, str):
        return [short_code(codes)]
    return [short_code(c) for c in codes]


def get_existing_date_files(base_dir):
    if not os.path.exists(base_dir):
        return []
    files = []
    for filename in os.listdir(base_dir):
        if filename.endswith(".parquet"):
            date_str = filename.replace(".parquet", "")
            if len(date_str) == 8:
                files.append((date_str, os.path.join(base_dir, filename)))
    files.sort(key=lambda x: x[0])
    return files


def filter_dates_by_range(date_files, start_date, end_date):
    if start_date is None and end_date is None:
        return date_files
    start_str = format_date(start_date) if start_date else "00000000"
    end_str = format_date(end_date) if end_date else "99999999"
    return [(d, f) for d, f in date_files if start_str <= d <= end_str]


def load_trading_dates_cache():
    filepath = get_data_path(TRADING_DATES_DIR, "trading_dates.parquet")
    if os.path.exists(filepath):
        df = pd.read_parquet(filepath)
        return df["date"].tolist()
    return []


_TRADING_DATES_CACHE = None


def get_trading_dates_cached():
    global _TRADING_DATES_CACHE
    if _TRADING_DATES_CACHE is None:
        _TRADING_DATES_CACHE = load_trading_dates_cache()
    return _TRADING_DATES_CACHE
