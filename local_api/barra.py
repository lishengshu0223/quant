import os
import pandas as pd
import pyarrow.dataset as ds
from .config import get_data_path, BARRA_DIR, FACTOR_MODELS, INDUSTRY_MAPPING
from ._utils import normalize_date, normalize_codes, short_codes, format_date, get_existing_date_files, filter_dates_by_range


def get_factor_exposure(order_book_ids, start_date=None, end_date=None, factors=None, 
                        industry_mapping=INDUSTRY_MAPPING, model="v1", market="cn"):
    
    base_dir = get_data_path(BARRA_DIR, model, "exposure")
    if not os.path.exists(base_dir):
        return pd.DataFrame()
    
    start_dt = normalize_date(start_date)
    end_dt = normalize_date(end_date)
    
    date_files = get_existing_date_files(base_dir)
    date_files = filter_dates_by_range(date_files, start_dt, end_dt)
    
    if not date_files:
        return pd.DataFrame()
    
    dataset = ds.dataset(base_dir, format="parquet")
    table = dataset.to_table()
    df = table.to_pandas()
    
    if not df.empty:
        df = df.reset_index()
        
        if order_book_ids:
            codes = normalize_codes(order_book_ids)
            df = df[df["code"].isin(codes)]
        
        df["code"] = df["code"].astype(str)
        
        if factors is not None:
            columns_to_keep = ["date", "code"] + list(factors)
            columns_to_keep = [c for c in columns_to_keep if c in df.columns]
            df = df[columns_to_keep]
        
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index(["date", "code"]).sort_index()
    
    return df


def get_factor_return(start_date=None, end_date=None, factors=None, 
                      industry_mapping=INDUSTRY_MAPPING, model="v1", market="cn"):
    
    base_dir = get_data_path(BARRA_DIR, model, "return")
    if not os.path.exists(base_dir):
        return pd.DataFrame()
    
    start_dt = normalize_date(start_date)
    end_dt = normalize_date(end_date)
    
    date_files = get_existing_date_files(base_dir)
    date_files = filter_dates_by_range(date_files, start_dt, end_dt)
    
    if not date_files:
        return pd.DataFrame()
    
    dataset = ds.dataset(base_dir, format="parquet")
    table = dataset.to_table()
    df = table.to_pandas()
    
    if not df.empty:
        if factors is not None:
            columns_to_keep = list(factors)
            columns_to_keep = [c for c in columns_to_keep if c in df.columns]
            df = df[columns_to_keep]
        
        df = df.sort_index()
    
    return df


def get_specific_risk(order_book_ids, start_date=None, end_date=None, 
                      horizon="daily", model="v1", industry_mapping=INDUSTRY_MAPPING):
    
    base_dir = get_data_path(BARRA_DIR, model, "exposure")
    if not os.path.exists(base_dir):
        return pd.DataFrame()
    
    start_dt = normalize_date(start_date)
    end_dt = normalize_date(end_date)
    
    date_files = get_existing_date_files(base_dir)
    date_files = filter_dates_by_range(date_files, start_dt, end_dt)
    
    if not date_files:
        return pd.DataFrame()
    
    dataset = ds.dataset(base_dir, format="parquet")
    table = dataset.to_table()
    df = table.to_pandas()
    
    if not df.empty:
        df = df.reset_index()
        
        if order_book_ids:
            codes = normalize_codes(order_book_ids)
            df = df[df["code"].isin(codes)]
        
        df["code"] = df["code"].astype(str)
        df["date"] = pd.to_datetime(df["date"])
        df = df[["date", "code", "specific_risk"]]
        df = df.set_index(["date", "code"]).sort_index()
    
    return df


def get_specific_return(order_book_ids, start_date=None, end_date=None, 
                        model="v1"):
    
    base_dir = get_data_path(BARRA_DIR, model, "exposure")
    if not os.path.exists(base_dir):
        return pd.DataFrame()
    
    start_dt = normalize_date(start_date)
    end_dt = normalize_date(end_date)
    
    date_files = get_existing_date_files(base_dir)
    date_files = filter_dates_by_range(date_files, start_dt, end_dt)
    
    if not date_files:
        return pd.DataFrame()
    
    dataset = ds.dataset(base_dir, format="parquet")
    table = dataset.to_table()
    df = table.to_pandas()
    
    if not df.empty:
        df = df.reset_index()
        
        if order_book_ids:
            codes = normalize_codes(order_book_ids)
            df = df[df["code"].isin(codes)]
        
        df["code"] = df["code"].astype(str)
        df["date"] = pd.to_datetime(df["date"])
        df = df[["date", "code", "specific_return"]]
        df = df.set_index(["date", "code"]).sort_index()
    
    return df
