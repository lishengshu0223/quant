import os
import pandas as pd
from rqdatac import get_shares
from .config import get_data_path, SHARES_DIR, ensure_dir
from .utils import get_stock_universe, resolve_target_dates, save_snapshot
from .logger import logger


def download_shares(data_root=None, download_dates=None, force=False, end_date=None):
    """
    下载 A 股每日股本结构（rqdatac.get_shares）。

    字段与米筐一致：total（总股本）、circulation_a（A 股流通股本）、
    non_circulation_a（非流通 A 股）、total_a（A 股总股本）、
    preferred_shares（优先股）、free_circulation（自由流通股本）。

    存储结构与 stock_price 一致：F:\\Trade_data\\shares\\YYYYMMDD.parquet，
    索引为 (date, code)，列为上述股本字段（单位：股）。
    """
    if data_root is None:
        from .config import DEFAULT_DATA_ROOT
        data_root = DEFAULT_DATA_ROOT

    try:
        logger.info("Starting download_shares")

        stock_codes = get_stock_universe()
        logger.info(f"Found {len(stock_codes)} stocks")

        base_dir = get_data_path(SHARES_DIR)
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

                df = get_shares(
                    stock_codes,
                    start_date=date,
                    end_date=date,
                    expect_df=True,
                    market="cn",
                )

                if df is None or df.empty:
                    logger.warning(f"No data for {date_str}")
                    continue

                df = df.reset_index()
                df.columns = ["code", "date"] + list(df.columns[2:])
                df = df.set_index(["date", "code"])

                save_path = os.path.join(base_dir, f"{date_str}.parquet")
                save_snapshot(df, save_path)

                success_dates.append(date_str)
                logger.info(f"Downloaded shares for {date_str}, {len(df)} records")

            except Exception as e:
                logger.error(f"Failed to download shares for {date_str}: {str(e)}")

        logger.info(f"download_shares completed, {len(success_dates)}/{len(missing_dates)} dates downloaded")
        return success_dates

    except Exception as e:
        logger.error(f"download_shares failed: {str(e)}")
        raise
