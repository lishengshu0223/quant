import pandas as pd
from .config import get_data_path, INSTRUMENTS_DIR

_INSTRUMENTS_CACHE = None


def all_instruments(type="CS", market="cn"):
    global _INSTRUMENTS_CACHE
    
    if _INSTRUMENTS_CACHE is None:
        filepath = get_data_path(INSTRUMENTS_DIR, "all_stocks.parquet")
        if not filepath or not __import__('os').path.exists(filepath):
            return pd.DataFrame()
        _INSTRUMENTS_CACHE = pd.read_parquet(filepath)
    
    df = _INSTRUMENTS_CACHE.copy()
    
    if type is not None:
        df = df[df["type"] == type]
    
    return df


def instrument(code, market="cn"):
    df = all_instruments(type=None, market=market)
    if isinstance(code, str):
        return df[df["order_book_id"] == code]
    return df[df["order_book_id"].isin(code)]


def get_stock_codes(type="CS", market="cn"):
    df = all_instruments(type=type, market=market)
    return df["order_book_id"].tolist()
