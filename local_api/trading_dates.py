import pandas as pd
from .config import get_data_path, TRADING_DATES_DIR
from ._utils import normalize_date, get_trading_dates_cached


def get_trading_dates(start_date=None, end_date=None, market="cn"):
    all_dates = get_trading_dates_cached()
    
    start_dt = normalize_date(start_date)
    end_dt = normalize_date(end_date)
    
    if start_dt is None and end_dt is None:
        return all_dates
    
    dates_series = pd.Series(all_dates)
    
    mask = pd.Series(True, index=dates_series.index)
    if start_dt is not None:
        mask &= dates_series >= start_dt.date()
    if end_dt is not None:
        mask &= dates_series <= end_dt.date()
    
    return dates_series[mask].tolist()


def get_previous_trading_date(date=None, market="cn"):
    all_dates = get_trading_dates_cached()
    
    if date is None:
        date = pd.Timestamp.now()
    else:
        date = normalize_date(date)
    
    dates_series = pd.Series(all_dates)
    mask = dates_series < date.date()
    if mask.any():
        return dates_series[mask].iloc[-1]
    return None


def get_next_trading_date(date=None, market="cn"):
    all_dates = get_trading_dates_cached()
    
    if date is None:
        date = pd.Timestamp.now()
    else:
        date = normalize_date(date)
    
    dates_series = pd.Series(all_dates)
    mask = dates_series > date.date()
    if mask.any():
        return dates_series[mask].iloc[0]
    return None


def get_latest_trading_date(market="cn"):
    all_dates = get_trading_dates_cached()
    if all_dates:
        return max(all_dates)
    return None
