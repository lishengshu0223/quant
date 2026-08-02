import os
import pandas as pd
import pyarrow.dataset as ds
from .config import get_data_path, INDEX_PRICE_DIR, INDEX_FULL_CODES
from ._utils import normalize_date, normalize_codes, short_codes, get_existing_date_files, filter_dates_by_range


def get_index_price(order_book_ids, start_date=None, end_date=None, frequency="1d", 
                    fields=None, skip_suspended=True, market="cn", expect_df=True):
    
    base_dir = get_data_path(INDEX_PRICE_DIR)
    if not os.path.exists(base_dir):
        return pd.DataFrame()
    
    start_dt = normalize_date(start_date)
    end_dt = normalize_date(end_date)
    
    date_files = get_existing_date_files(base_dir)
    date_files = filter_dates_by_range(date_files, start_dt, end_dt)
    
    if not date_files:
        return pd.DataFrame()
    
    if fields is None:
        fields = ["open", "close", "high", "low", "total_turnover", "volume"]

    codes = normalize_codes(order_book_ids)
    short_codes_list = short_codes(codes) if codes else None

    # 只读取日期范围内的文件
    filepaths = [f for _, f in date_files]
    dataset = ds.dataset(filepaths, format="parquet")
    table = dataset.to_table()
    df = table.to_pandas()
    
    if not df.empty:
        df = df.reset_index()
        
        if short_codes_list:
            df = df[df["code"].isin(short_codes_list)]
        
        df["code"] = df["code"].astype(str)
        
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index(["date", "code"]).sort_index()
        
        df = df[fields]
    
    if expect_df:
        return df
    return df.unstack("code")
