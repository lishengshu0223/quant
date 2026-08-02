import os
import pandas as pd
from rqdatac import get_trading_dates, get_latest_trading_date
from .config import get_data_path, TRADING_DATES_DIR, ensure_dir
from .logger import logger


def download_trading_dates(data_root=None):
    if data_root is None:
        from .config import DEFAULT_DATA_ROOT
        data_root = DEFAULT_DATA_ROOT
    
    try:
        logger.info("Starting download_trading_dates")
        start_date = "2000-01-01"
        end_date = "2030-12-31"
        dates = get_trading_dates(start_date, end_date, market="cn")
        
        df = pd.DataFrame({"date": dates})
        save_path = get_data_path(TRADING_DATES_DIR, "trading_dates.parquet")
        ensure_dir(os.path.dirname(save_path))
        df.to_parquet(save_path, engine="pyarrow")
        
        logger.info(f"download_trading_dates completed, saved {len(df)} dates")
        return dates
    except Exception as e:
        logger.error(f"download_trading_dates failed: {str(e)}")
        raise