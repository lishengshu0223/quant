import os
import pandas as pd
import pyarrow.dataset as ds
from .config import (
    get_data_path,
    STOCK_PRICE_DIR,
    STOCK_PRICE_FIELDS,
    ADJUSTED_FIELDS,
    MINUTE_FREQUENCIES,
)
from ._utils import normalize_date, normalize_codes, short_codes, format_date, get_existing_date_files, filter_dates_by_range


def _get_daily_price(order_book_ids, start_date=None, end_date=None,
                     fields=None, adjust_type="none", skip_suspended=True,
                     market="cn", expect_df=True):
    base_dir = get_data_path(STOCK_PRICE_DIR)
    if not os.path.exists(base_dir):
        return pd.DataFrame()

    start_dt = normalize_date(start_date)
    end_dt = normalize_date(end_date)

    date_files = get_existing_date_files(base_dir)
    date_files = filter_dates_by_range(date_files, start_dt, end_dt)

    if not date_files:
        return pd.DataFrame()

    if fields is None:
        fields = STOCK_PRICE_FIELDS.copy()

    codes = normalize_codes(order_book_ids)

    # 只读取日期范围内的文件，避免读取全量数据
    filepaths = [f for _, f in date_files]
    dataset = ds.dataset(filepaths, format="parquet")
    table = dataset.to_table()
    df = table.to_pandas()

    if not df.empty:
        df = df.reset_index()

        if codes:
            df = df[df["code"].isin(codes)]

        df["code"] = df["code"].astype(str)
        df["date"] = pd.to_datetime(df["date"])

        if adjust_type == "post":
            result_df = df[["date", "code"]].copy()
            for f in fields:
                if f == "total_turnover":
                    result_df[f] = df[f]
                else:
                    adj_field = f"adj{f}"
                    if adj_field in df.columns:
                        result_df[f] = df[adj_field]
                    else:
                        result_df[f] = df[f]
            df = result_df

        df = df.set_index(["date", "code"]).sort_index()
        df = df[fields]

    if expect_df:
        return df
    return df.unstack("code")


def get_price(order_book_ids, start_date=None, end_date=None, frequency="1d",
              fields=None, adjust_type="none", skip_suspended=True, market="cn",
              expect_df=True, time_slice=None):
    """
    获取股票/指数行情数据（日线或分钟线）

    Parameters
    ----------
    order_book_ids : str or list[str]
        合约代码或代码列表
    start_date : str or pd.Timestamp, optional
        开始日期
    end_date : str or pd.Timestamp, optional
        结束日期
    frequency : str, default "1d"
        频率: "1d"（日线）, "1m", "5m", "15m", "30m", "60m"（分钟线）
    fields : str or list[str], optional
        需要的字段
    adjust_type : str, default "none"
        复权方式: "none", "pre", "post", "pre_volume", "post_volume"
        日线仅支持 "none" 和 "post"
    skip_suspended : bool, default True
        是否跳过停牌数据
    market : str, default "cn"
        市场
    expect_df : bool, default True
        是否返回DataFrame
    time_slice : tuple(str, str), optional
        分钟级别的时间段切分，如 ("09:31", "10:00")，仅分钟频率有效

    Returns
    -------
    pd.DataFrame
    """
    if frequency in MINUTE_FREQUENCIES:
        from .stock_minute import get_minute_price
        return get_minute_price(
            order_book_ids=order_book_ids,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            fields=fields,
            adjust_type=adjust_type,
            skip_suspended=skip_suspended,
            market=market,
            expect_df=expect_df,
            time_slice=time_slice,
        )
    else:
        return _get_daily_price(
            order_book_ids=order_book_ids,
            start_date=start_date,
            end_date=end_date,
            fields=fields,
            adjust_type=adjust_type,
            skip_suspended=skip_suspended,
            market=market,
            expect_df=expect_df,
        )
