import os
import pandas as pd
from rqdatac import get_factor_exposure, get_factor_return, get_specific_risk, get_specific_return
from .config import get_data_path, BARRA_DIR, FACTOR_MODELS, INDUSTRY_MAPPING, ensure_dir
from .utils import get_stock_universe, resolve_target_dates, save_snapshot
from .logger import logger


def download_barra_exposure(model="v1", data_root=None, download_dates=None, force=False, end_date=None):
    if data_root is None:
        from .config import DEFAULT_DATA_ROOT
        data_root = DEFAULT_DATA_ROOT

    try:
        logger.info(f"Starting download_barra_exposure for model {model}")

        stock_codes = get_stock_universe()
        logger.info(f"Found {len(stock_codes)} stocks")

        base_dir = get_data_path(BARRA_DIR, model, "exposure")
        ensure_dir(base_dir)

        missing_dates = resolve_target_dates(base_dir, download_dates, force, end_date)

        if not missing_dates:
            logger.info(f"No missing dates for barra exposure {model}")
            return []

        logger.info(f"Downloading {len(missing_dates)} missing dates for barra exposure {model}")

        success_dates = []
        for date_str in missing_dates:
            try:
                date = pd.Timestamp(date_str)

                df_exp = get_factor_exposure(
                    stock_codes,
                    start_date=date,
                    end_date=date,
                    factors=None,
                    industry_mapping=INDUSTRY_MAPPING,
                    model=model,
                    market="cn",
                )

                if df_exp is None or df_exp.empty:
                    logger.warning(f"No exposure data for {date_str}")
                    continue

                df_exp = df_exp.reset_index()
                df_exp.columns = ["date", "code"] + list(df_exp.columns[2:])

                if "comovement" in df_exp.columns:
                    df_exp = df_exp.drop(columns=["comovement"])

                df_specific_risk = get_specific_risk(
                    stock_codes,
                    start_date=date,
                    end_date=date,
                    horizon="daily",
                    model=model,
                    industry_mapping=INDUSTRY_MAPPING,
                )

                df_specific_return = get_specific_return(
                    stock_codes,
                    start_date=date,
                    end_date=date,
                    model=model,
                )

                if not df_specific_risk.empty:
                    df_specific_risk = df_specific_risk.reset_index().melt(id_vars=["date"], var_name="code", value_name="specific_risk")
                    df_exp = pd.merge(df_exp, df_specific_risk, on=["date", "code"], how="left")

                if not df_specific_return.empty:
                    df_specific_return = df_specific_return.reset_index().melt(id_vars=["date"], var_name="code", value_name="specific_return")
                    df_exp = pd.merge(df_exp, df_specific_return, on=["date", "code"], how="left")

                df_exp = df_exp.set_index(["date", "code"])
                df_exp = df_exp.sort_index(axis=1)

                save_path = os.path.join(base_dir, f"{date_str}.parquet")
                save_snapshot(df_exp, save_path)

                success_dates.append(date_str)
                logger.info(f"Downloaded barra exposure {model} for {date_str}, {len(df_exp)} records")

            except Exception as e:
                logger.error(f"Failed to download barra exposure {model} for {date_str}: {str(e)}")

        logger.info(f"download_barra_exposure {model} completed, {len(success_dates)}/{len(missing_dates)} dates downloaded")
        return success_dates

    except Exception as e:
        logger.error(f"download_barra_exposure {model} failed: {str(e)}")
        raise


def download_barra_return(model="v1", data_root=None, download_dates=None, force=False, end_date=None):
    if data_root is None:
        from .config import DEFAULT_DATA_ROOT
        data_root = DEFAULT_DATA_ROOT

    try:
        logger.info(f"Starting download_barra_return for model {model}")

        base_dir = get_data_path(BARRA_DIR, model, "return")
        ensure_dir(base_dir)

        missing_dates = resolve_target_dates(base_dir, download_dates, force, end_date)

        if not missing_dates:
            logger.info(f"No missing dates for barra return {model}")
            return []

        logger.info(f"Downloading {len(missing_dates)} missing dates for barra return {model}")

        start_date = pd.Timestamp(min(missing_dates))
        end_date = pd.Timestamp(max(missing_dates))

        df = get_factor_return(
            start_date=start_date,
            end_date=end_date,
            factors=None,
            industry_mapping=INDUSTRY_MAPPING,
            model=model,
            market="cn",
        )

        if df.empty:
            logger.warning(f"No barra return data for {model}")
            return []

        if "comovement" in df.columns:
            df = df.drop(columns=["comovement"])

        df = df.sort_index(axis=1)

        success_dates = []
        for date_str in missing_dates:
            date = pd.Timestamp(date_str)
            if date in df.index:
                day_df = df.loc[[date]]
                save_path = os.path.join(base_dir, f"{date_str}.parquet")
                save_snapshot(day_df, save_path)
                success_dates.append(date_str)

        logger.info(f"download_barra_return {model} completed, {len(success_dates)}/{len(missing_dates)} dates downloaded")
        return success_dates

    except Exception as e:
        logger.error(f"download_barra_return {model} failed: {str(e)}")
        raise


def download_all_barra(data_root=None, download_dates=None, force=False, end_date=None):
    download_barra_exposure("v1", data_root, download_dates, force, end_date)
    download_barra_exposure("v2", data_root, download_dates, force, end_date)
    download_barra_return("v1", data_root, download_dates, force, end_date)
    download_barra_return("v2", data_root, download_dates, force, end_date)