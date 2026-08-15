import os
import pandas as pd
from rqdatac import get_price
from .config import get_data_path, INDEX_PRICE_DIR, INDEX_FULL_CODES, ensure_dir
from .utils import format_date, resolve_target_dates, save_snapshot
from .logger import logger


def download_index_daily_price(data_root=None, download_dates=None, force=False, end_date=None):
    if data_root is None:
        from .config import DEFAULT_DATA_ROOT
        data_root = DEFAULT_DATA_ROOT

    try:
        logger.info("Starting download_index_daily_price")

        index_codes = list(INDEX_FULL_CODES.values())
        logger.info(f"Downloading {len(index_codes)} indices")

        base_dir = get_data_path(INDEX_PRICE_DIR)
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

                df = get_price(
                    index_codes,
                    start_date=date,
                    end_date=date,
                    frequency="1d",
                    fields=["open", "close", "high", "low", "total_turnover", "volume"],
                    skip_suspended=True,
                    market="cn",
                    expect_df=True,
                )

                if df.empty:
                    logger.warning(f"No data for {date_str}")
                    continue

                if isinstance(df.index, pd.MultiIndex):
                    df = df.reset_index()
                    df.columns = ["code", "date", "open", "close", "high", "low", "total_turnover", "volume"]
                    df["code"] = df["code"].astype(str).apply(lambda x: x.split(".")[0])
                else:
                    df = df.reset_index()
                    df.columns = ["date", "open", "close", "high", "low", "total_turnover", "volume"]
                    df["code"] = index_codes[0].split(".")[0]

                df = df.set_index(["date", "code"])

                save_path = os.path.join(base_dir, f"{date_str}.parquet")
                save_snapshot(df, save_path)

                success_dates.append(date_str)
                logger.info(f"Downloaded index price for {date_str}, {len(df)} records")

            except Exception as e:
                logger.error(f"Failed to download index price for {date_str}: {str(e)}")

        logger.info(f"download_index_daily_price completed, {len(success_dates)}/{len(missing_dates)} dates downloaded")
        return success_dates

    except Exception as e:
        logger.error(f"download_index_daily_price failed: {str(e)}")
        raise