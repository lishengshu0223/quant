# -*- coding: utf-8 -*-
"""
每日可交易股票状态更新

数据源: rqdatac 在线 API
存储: F:\\Trade_data\\tradable_status\\YYYYMMDD.parquet
      每个文件为当日全市场（当日已上市且未退市的股票）的可交易状态，
      列为: code, is_st, is_suspended, is_new_listed, is_limit, tradable

可交易判断（tradable = False 表示当日不可交易）:
  - is_st:         当日为 ST / *ST 股票
  - is_suspended:  当日停牌
  - is_new_listed: 上市未满一年（上市日期距今不足 MIN_LIST_DAYS 个自然日）
  - is_limit:      当日收盘价触及涨跌停（含一字涨跌停）
  以上任一为 True 则该股当日不可交易

未上市或已退市的股票不会出现在当日文件中，由 local_api 读取时统一表示为 NaN。
"""
import os

import pandas as pd
from rqdatac import all_instruments, is_st_stock, is_suspended, get_price, get_trading_dates

from download.config import DEFAULT_DATA_ROOT, START_DATE
from download.utils import format_date, get_existing_dates, get_missing_dates
from download.logger import logger

# 存储目录名（相对 DEFAULT_DATA_ROOT）
TRADABLE_STATUS_DIR = "tradable_status"
# 上市未满一年（自然日）
MIN_LIST_DAYS = 365
# 涨跌停判断的浮点容差
LIMIT_EPS = 1e-4
# 整段日期分块查询的交易日数量，避免单次 API 请求过大
BLOCK_DAYS = 60


def _get_universe():
    """获取全市场 A 股合约（含已退市），返回 order_book_id / listed_date / de_listed_date"""
    ai = all_instruments(type="CS", market="cn")
    ai = ai[["order_book_id", "listed_date", "de_listed_date"]].copy()
    # '0000-00-00' 表示未退市，转为 NaT 便于日期比较
    ai["listed_date"] = pd.to_datetime(ai["listed_date"], errors="coerce", format="%Y-%m-%d")
    ai["de_listed_date"] = pd.to_datetime(ai["de_listed_date"], errors="coerce", format="%Y-%m-%d")
    return ai


def _get_base_dir():
    base_dir = os.path.join(DEFAULT_DATA_ROOT, TRADABLE_STATUS_DIR)
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def _resolve_target_dates(download_dates, force, end_date):
    """确定待更新的日期列表（YYYYMMDD 字符串，已排序）"""
    base_dir = _get_base_dir()
    existing_dates = get_existing_dates(base_dir)

    if download_dates is None:
        if end_date is None:
            end_date = pd.Timestamp.now().date()
        else:
            end_date = pd.Timestamp(end_date).date()
        trading_dates = get_trading_dates(pd.Timestamp(START_DATE), pd.Timestamp(end_date), market="cn")
        if force:
            target_dates = [format_date(d) for d in trading_dates]
        else:
            target_dates = get_missing_dates(trading_dates, existing_dates)
    else:
        if force:
            target_dates = [format_date(d) for d in download_dates]
        else:
            target_dates = [d for d in download_dates if d not in existing_dates]
    return sorted(target_dates)


def _compute_day(universe, st_matrix, susp_matrix, price, date_str):
    """计算单个交易日的可交易状态并保存 parquet，返回该日 DataFrame"""
    date = pd.Timestamp(date_str)

    # 当日已上市（含当日上市）且未退市的股票
    listed_mask = (
        (universe["listed_date"] <= date)
        & (universe["de_listed_date"].isna() | (universe["de_listed_date"] > date))
    )
    universe_day = universe.loc[listed_mask].set_index("order_book_id")
    codes_today = universe_day.index.tolist()
    if not codes_today:
        logger.warning(f"{date_str} 当日无已上市股票")
        return None

    # ST / *ST 状态
    st = st_matrix.loc[date] if date in st_matrix.index else pd.Series(False, index=codes_today)
    # 停牌状态
    susp = susp_matrix.loc[date] if date in susp_matrix.index else pd.Series(False, index=codes_today)

    # 上市未满一年
    age_days = (date - universe_day["listed_date"]).dt.days
    is_new_listed = age_days < MIN_LIST_DAYS

    # 涨跌停（含一字涨跌停）：收盘价触及涨停价或跌停价
    day_price = None
    try:
        day_price = price.xs(date, level="date")
    except KeyError:
        pass
    if day_price is None:
        is_limit = pd.Series(False, index=codes_today)
    else:
        close = day_price["close"].reindex(codes_today)
        limit_up = day_price["limit_up"].reindex(codes_today)
        limit_down = day_price["limit_down"].reindex(codes_today)
        is_limit = ((close >= limit_up - LIMIT_EPS) | (close <= limit_down + LIMIT_EPS)).fillna(False)

    # 汇总
    day_df = pd.DataFrame(index=codes_today)
    day_df["is_st"] = st.reindex(codes_today).fillna(False).values
    day_df["is_suspended"] = susp.reindex(codes_today).fillna(False).values
    day_df["is_new_listed"] = is_new_listed.reindex(codes_today).fillna(False).values
    day_df["is_limit"] = is_limit.reindex(codes_today).fillna(False).values
    day_df["tradable"] = ~(
        day_df["is_st"] | day_df["is_suspended"] | day_df["is_new_listed"] | day_df["is_limit"]
    )
    return day_df.reset_index().rename(columns={"index": "code"})


def download_tradable_status(download_dates=None, force=False, end_date=None):
    """
    更新全市场每日可交易股票状态

    Parameters
    ----------
    download_dates : list[str], optional
        指定要更新的日期（YYYYMMDD）。为空时自动补全缺失日期
    force : bool
        True 时即使文件已存在也强制更新
    end_date : str or datetime.date, optional
        download_dates 为空时，更新的截止日期，默认到最新交易日

    Returns
    -------
    list[str]
        成功保存的日期列表
    """
    try:
        logger.info("Starting download_tradable_status")

        universe = _get_universe()
        codes = universe["order_book_id"].tolist()
        logger.info(f"全市场股票数: {len(codes)}（含退市 {int(universe['de_listed_date'].notna().sum())} 只）")

        target_dates = _resolve_target_dates(download_dates, force, end_date)
        if not target_dates:
            logger.info("无可更新的日期")
            return []

        logger.info(f"待更新 {len(target_dates)} 个交易日: {target_dates[0]} ~ {target_dates[-1]}")

        # 按块查询 ST / 停牌 / 涨跌停，减少 API 调用次数
        blocks = [target_dates[i:i + BLOCK_DAYS] for i in range(0, len(target_dates), BLOCK_DAYS)]
        success_dates = []
        for block in blocks:
            block_start = pd.Timestamp(block[0])
            block_end = pd.Timestamp(block[-1])
            try:
                st_matrix = is_st_stock(codes, block_start, block_end, market="cn")
                susp_matrix = is_suspended(codes, block_start, block_end, market="cn")
                price = get_price(
                    codes,
                    start_date=block_start,
                    end_date=block_end,
                    frequency="1d",
                    fields=["close", "limit_up", "limit_down"],
                    adjust_type="none",
                    skip_suspended=False,
                    market="cn",
                    expect_df=True,
                )
            except Exception as e:
                logger.error(f"块查询失败 {block[0]}~{block[-1]}: {e}")
                continue

            for date_str in block:
                try:
                    day_df = _compute_day(universe, st_matrix, susp_matrix, price, date_str)
                    if day_df is None:
                        continue
                    save_path = os.path.join(_get_base_dir(), f"{date_str}.parquet")
                    day_df.to_parquet(save_path, engine="pyarrow")
                    success_dates.append(date_str)
                    logger.info(
                        f"已保存 {date_str} 可交易状态, 股票 {len(day_df)} 只, "
                        f"可交易 {int(day_df['tradable'].sum())} 只"
                    )
                except Exception as e:
                    logger.error(f"计算失败 {date_str}: {e}")

        logger.info(f"download_tradable_status 完成, 成功 {len(success_dates)}/{len(target_dates)} 个交易日")
        return success_dates
    except Exception as e:
        logger.error(f"download_tradable_status 失败: {e}")
        raise


if __name__ == "__main__":
    import sys
    import time

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import rqdatac

    rqdatac.init()
    start = time.time()
    download_tradable_status()
    print(f"总耗时: {time.time() - start:.1f} 秒")
