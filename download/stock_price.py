import os
import pandas as pd
from rqdatac import get_price
from .config import get_data_path, STOCK_PRICE_DIR, STOCK_PRICE_FIELDS, ensure_dir
from .utils import format_date, get_stock_universe, resolve_target_dates, save_snapshot
from .logger import logger


def download_stock_daily_price(data_root=None, download_dates=None, force=False, end_date=None):
    if data_root is None:
        from .config import DEFAULT_DATA_ROOT
        data_root = DEFAULT_DATA_ROOT

    try:
        logger.info("Starting download_stock_daily_price")

        stock_codes = get_stock_universe()
        logger.info(f"Found {len(stock_codes)} stocks")

        base_dir = get_data_path(STOCK_PRICE_DIR)
        ensure_dir(base_dir)

        missing_dates = resolve_target_dates(base_dir, download_dates, force, end_date)

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

                save_path = os.path.join(base_dir, f"{date_str}.parquet")
                save_snapshot(df_unadj, save_path)

                success_dates.append(date_str)
                logger.info(f"Downloaded stock price for {date_str}, {len(df_unadj)} records")

            except Exception as e:
                logger.error(f"Failed to download stock price for {date_str}: {str(e)}")

        logger.info(f"download_stock_daily_price completed, {len(success_dates)}/{len(missing_dates)} dates downloaded")
        return success_dates

    except Exception as e:
        logger.error(f"download_stock_daily_price failed: {str(e)}")
        raise