import os
import pandas as pd
from rqdatac import index_weights, index_weights_ex, get_trading_dates
from .config import get_data_path, INDEX_WEIGHTS_DIR, INDEX_WEIGHT_CODES, START_DATE, ensure_dir
from .utils import format_date, get_existing_dates, get_missing_dates
from .logger import logger


def download_index_weights(data_root=None, download_dates=None, force=False, end_date=None):
    if data_root is None:
        from .config import DEFAULT_DATA_ROOT
        data_root = DEFAULT_DATA_ROOT
    
    try:
        logger.info("Starting download_index_weights")
        
        short_codes = list(INDEX_WEIGHT_CODES.keys())
        full_codes = list(INDEX_WEIGHT_CODES.values())
        
        base_dir = get_data_path(INDEX_WEIGHTS_DIR)
        ensure_dir(base_dir)
        
        for short_code, full_code in INDEX_WEIGHT_CODES.items():
            try:
                index_dir = os.path.join(base_dir, short_code)
                ensure_dir(index_dir)
                
                existing_dates = get_existing_dates(index_dir)
                
                if download_dates is None:
                    if end_date is None:
                        end_date = pd.Timestamp.now().date()
                    else:
                        end_date = pd.Timestamp(end_date).date()
                    start_dt = pd.Timestamp(START_DATE)
                    end_dt = pd.Timestamp(end_date)
                    trading_dates = get_trading_dates(start_dt, end_dt, market="cn")
                    if force:
                        missing_dates = [format_date(d) for d in trading_dates]
                    else:
                        missing_dates = get_missing_dates(trading_dates, existing_dates)
                else:
                    if force:
                        missing_dates = download_dates
                    else:
                        missing_dates = [d for d in download_dates if d not in existing_dates]
                
                if not missing_dates:
                    logger.info(f"No missing dates for index {short_code}")
                    continue
                
                logger.info(f"Downloading {len(missing_dates)} missing dates for index {short_code}")
                
                success_count = 0
                for date_str in missing_dates:
                    try:
                        date = pd.Timestamp(date_str)
                        
                        if full_code.endswith(".RI"):
                            weights = index_weights(full_code, date=date)
                        else:
                            weights = index_weights_ex(full_code, date=date)
                        
                        if weights is None or weights.empty:
                            logger.warning(f"No weight data for {short_code} on {date_str}")
                            continue
                        
                        df = weights.reset_index()
                        df.columns = ["code", short_code]
                        df["date"] = date
                        df = df.set_index(["date", "code"])
                        
                        df = df.astype({col: 'float32' for col in df.select_dtypes(include=['float64']).columns})
                        
                        save_path = os.path.join(index_dir, f"{date_str}.parquet")
                        df.to_parquet(save_path, engine="pyarrow")
                        
                        success_count += 1
                    
                    except Exception as e:
                        logger.error(f"Failed to download weights for {short_code} on {date_str}: {str(e)}")
                
                logger.info(f"Downloaded {success_count}/{len(missing_dates)} dates for index {short_code}")
            
            except Exception as e:
                logger.error(f"Failed to process index {short_code}: {str(e)}")
        
        logger.info("download_index_weights completed")
        return True
    
    except Exception as e:
        logger.error(f"download_index_weights failed: {str(e)}")
        raise