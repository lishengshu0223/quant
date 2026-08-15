import os
import pandas as pd
from rqdatac import get_turnover_rate
from .config import get_data_path, TURNOVER_DIR, ensure_dir
from .utils import get_stock_universe, resolve_target_dates, save_snapshot
from .logger import logger


def download_turnover_rate(data_root=None, download_dates=None, force=False, end_date=None):
    """
    下载 A 股每日换手率（rqdatac.get_turnover_rate 的 today 字段，单位 %）。

    存储结构与 stock_price 一致：F:\\Trade_data\\turnover\\YYYYMMDD.parquet，
    索引为 (date, code)，列为 turnover_rate。
    """
    if data_root is None:
        from .config import DEFAULT_DATA_ROOT
        data_root = DEFAULT_DATA_ROOT

    try:
        logger.info("Starting download_turnover_rate")

        stock_codes = get_stock_universe()
        logger.info(f"Found {len(stock_codes)} stocks")

        base_dir = get_data_path(TURNOVER_DIR)
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

                df = get_turnover_rate(
                    stock_codes,
                    start_date=date,
                    end_date=date,
                    fields=["today"],
                    expect_df=True,
                    market="cn",
                )

                if df is None or df.empty:
                    logger.warning(f"No data for {date_str}")
                    continue

                df = df.reset_index()
                df.columns = ["code", "date", "turnover_rate"]
                df = df.set_index(["date", "code"])

                save_path = os.path.join(base_dir, f"{date_str}.parquet")
                save_snapshot(df, save_path)

                success_dates.append(date_str)
                logger.info(f"Downloaded turnover rate for {date_str}, {len(df)} records")

            except Exception as e:
                logger.error(f"Failed to download turnover rate for {date_str}: {str(e)}")

        logger.info(f"download_turnover_rate completed, {len(success_dates)}/{len(missing_dates)} dates downloaded")
        return success_dates

    except Exception as e:
        logger.error(f"download_turnover_rate failed: {str(e)}")
        raise
