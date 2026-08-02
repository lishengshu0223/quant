import os
import pandas as pd
from rqdatac import get_price, all_instruments, get_trading_dates
from .config import get_data_path, STOCK_PRICE_DIR, STOCK_PRICE_FIELDS, START_DATE, ensure_dir
from .utils import format_date, get_existing_dates, get_missing_dates
from .logger import logger


def download_stock_daily_price(data_root=None, download_dates=None, force=False, end_date=None):
    if data_root is None:
        from .config import DEFAULT_DATA_ROOT
        data_root = DEFAULT_DATA_ROOT
    
    try:
        logger.info("Starting download_stock_daily_price")
        
        stock_codes = all_instruments(type="CS", market="cn")
        stock_codes = stock_codes[stock_codes["type"] == "CS"]["order_book_id"].tolist()
        logger.info(f"Found {len(stock_codes)} stocks")
        
        base_dir = get_data_path(STOCK_PRICE_DIR)
        ensure_dir(base_dir)
        
        existing_dates = get_existing_dates(base_dir)
        
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
            logger.info("No missing dates to download")
            return []
        
        logger.info(f"Downloading {len(missing_dates)} missing dates")
        
        success_dates = []
        for date_str in missing_dates:
            try:
                date = pd.Timestamp(date_str)
                
                df_unadj = get_price(
                    stock_codes,
                    start_date=date,
                    end_date=date,
                    frequency="1d",
                    fields=STOCK_PRICE_FIELDS,
                    adjust_type="none",
                    skip_suspended=True,
                    market="cn",
                    expect_df=True,
                )
                
                df_post = get_price(
                    stock_codes,
                    start_date=date,
                    end_date=date,
                    frequency="1d",
                    fields=[f for f in STOCK_PRICE_FIELDS if f != "total_turnover"],
                    adjust_type="post",
                    skip_suspended=True,
                    market="cn",
                    expect_df=True,
                )
                
                if df_unadj.empty:
                    logger.warning(f"No data for {date_str}")
                    continue
                
                df_unadj = df_unadj.reset_index()
                df_unadj.columns = ["code", "date", "open", "close", "high", "low", "total_turnover", "volume"]
                
                if not df_post.empty:
                    df_post = df_post.reset_index()
                    df_post.columns = ["code", "date", "adjopen", "adjclose", "adjhigh", "adjlow", "adjvolume"]
                    df_unadj = pd.merge(df_unadj, df_post, on=["date", "code"], how="left")
                else:
                    for field in ["open", "close", "high", "low", "volume"]:
                        df_unadj[f"adj{field}"] = df_unadj[field]
                
                df_unadj["cum_exfactor"] = df_unadj["adjopen"] / df_unadj["open"]
                
                df_unadj = df_unadj.set_index(["date", "code"])
                
                df_unadj = df_unadj.astype({col: 'float32' for col in df_unadj.select_dtypes(include=['float64']).columns})
                
                save_path = os.path.join(base_dir, f"{date_str}.parquet")
                df_unadj.to_parquet(save_path, engine="pyarrow")
                
                success_dates.append(date_str)
                logger.info(f"Downloaded stock price for {date_str}, {len(df_unadj)} records")
            
            except Exception as e:
                logger.error(f"Failed to download stock price for {date_str}: {str(e)}")
        
        logger.info(f"download_stock_daily_price completed, {len(success_dates)}/{len(missing_dates)} dates downloaded")
        return success_dates
    
    except Exception as e:
        logger.error(f"download_stock_daily_price failed: {str(e)}")
        raise