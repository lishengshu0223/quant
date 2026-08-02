import os
import pandas as pd
import pyarrow.dataset as ds
from .config import get_data_path, INDEX_WEIGHTS_DIR, INDEX_FULL_CODES, INDEX_WEIGHT_CODES
from ._utils import normalize_date, short_code, format_date, get_existing_date_files, filter_dates_by_range


def index_weights(order_book_id, date=None, market="cn"):
    
    short_code_value = short_code(order_book_id)
    if short_code_value not in INDEX_WEIGHT_CODES:
        return pd.DataFrame()
    
    index_dir = get_data_path(INDEX_WEIGHTS_DIR, short_code_value)
    if not os.path.exists(index_dir):
        return pd.DataFrame()
    
    date_dt = normalize_date(date)
    date_str = format_date(date_dt) if date_dt else None
    
    if date_str:
        filepath = os.path.join(index_dir, f"{date_str}.parquet")
        if not os.path.exists(filepath):
            return pd.DataFrame()
        df = pd.read_parquet(filepath)
    else:
        date_files = get_existing_date_files(index_dir)
        if not date_files:
            return pd.DataFrame()
        latest_date_str, latest_file = date_files[-1]
        df = pd.read_parquet(latest_file)
    
    df = df.reset_index()
    df["code"] = df["code"].astype(str)
    df = df.set_index("code")
    
    return df[short_code_value]


def index_weights_ex(order_book_id, date=None, market="cn"):
    
    short_code_value = short_code(order_book_id)
    if short_code_value not in INDEX_WEIGHT_CODES:
        return pd.DataFrame()
    
    index_dir = get_data_path(INDEX_WEIGHTS_DIR, short_code_value)
    if not os.path.exists(index_dir):
        return pd.DataFrame()
    
    date_dt = normalize_date(date)
    date_str = format_date(date_dt) if date_dt else None
    
    if date_str:
        filepath = os.path.join(index_dir, f"{date_str}.parquet")
        if not os.path.exists(filepath):
            return pd.DataFrame()
        df = pd.read_parquet(filepath)
    else:
        date_files = get_existing_date_files(index_dir)
        if not date_files:
            return pd.DataFrame()
        latest_date_str, latest_file = date_files[-1]
        df = pd.read_parquet(latest_file)
    
    df = df.reset_index()
    df["code"] = df["code"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index(["date", "code"]).sort_index()
    
    return df


def get_index_weights(order_book_ids, start_date=None, end_date=None, market="cn"):
    
    if isinstance(order_book_ids, str):
        order_book_ids = [order_book_ids]
    
    result_dfs = []
    for order_book_id in order_book_ids:
        short_code_value = short_code(order_book_id)
        if short_code_value not in INDEX_WEIGHT_CODES:
            continue
        
        index_dir = get_data_path(INDEX_WEIGHTS_DIR, short_code_value)
        if not os.path.exists(index_dir):
            continue
        
        start_dt = normalize_date(start_date)
        end_dt = normalize_date(end_date)
        
        date_files = get_existing_date_files(index_dir)
        date_files = filter_dates_by_range(date_files, start_dt, end_dt)
        
        if not date_files:
            continue
        
        dfs = []
        for _, filepath in date_files:
            df = pd.read_parquet(filepath)
            dfs.append(df)
        
        if dfs:
            df = pd.concat(dfs)
            df = df.reset_index()
            df["code"] = df["code"].astype(str)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index(["date", "code"]).sort_index()
            result_dfs.append(df)
    
    if result_dfs:
        return pd.concat(result_dfs)
    return pd.DataFrame()
