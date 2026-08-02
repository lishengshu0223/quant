import os
import pandas as pd
from rqdatac import all_instruments
from .config import get_data_path, INSTRUMENTS_DIR, ensure_dir
from .logger import logger


def download_all_instruments(data_root=None):
    if data_root is None:
        from .config import DEFAULT_DATA_ROOT
        data_root = DEFAULT_DATA_ROOT
    
    try:
        logger.info("Starting download_all_instruments")
        df = all_instruments(type="CS", market="cn")
        df = df[df["type"] == "CS"]
        
        save_path = get_data_path(INSTRUMENTS_DIR, "all_stocks.parquet")
        ensure_dir(os.path.dirname(save_path))
        df.to_parquet(save_path, engine="pyarrow")
        
        logger.info(f"download_all_instruments completed, saved {len(df)} records")
        return df
    except Exception as e:
        logger.error(f"download_all_instruments failed: {str(e)}")
        raise