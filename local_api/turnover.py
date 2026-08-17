import os
import pandas as pd
import pyarrow.dataset as ds
from .config import get_data_path, TURNOVER_DIR
from ._utils import normalize_date, normalize_codes, get_existing_date_files, filter_dates_by_range


def get_turnover_rate(order_book_ids=None, start_date=None, end_date=None, fields=None,
                      market="cn", expect_df=True):
    """
    获取 A 股每日换手率（仿 rqdatac.get_turnover_rate）

    数据来源：本地 F:\\Trade_data\\turnover\\YYYYMMDD.parquet，
    由 rqdatac.get_turnover_rate 的 today 字段（当日换手率，单位 %）生成。

    Parameters
    ----------
    order_book_ids : str or list[str], optional
        合约代码或代码列表，如 "000001.XSHE" 或 "000001"
    start_date : str or pd.Timestamp, optional
        开始日期
    end_date : str or pd.Timestamp, optional
        结束日期
    fields : str or list[str], optional
        需要的字段，仅支持 "today"（与米筐一致，即当日换手率）或 "turnover_rate"
    market : str, default "cn"
        市场
    expect_df : bool, default True
        是否返回 DataFrame

    Returns
    -------
    pd.DataFrame
        MultiIndex (date, code)，列为 turnover_rate（单位 %）
    """
    base_dir = get_data_path(TURNOVER_DIR)
    if not os.path.exists(base_dir):
        return pd.DataFrame()

    start_dt = normalize_date(start_date)
    end_dt = normalize_date(end_date)

    date_files = get_existing_date_files(base_dir)
    date_files = filter_dates_by_range(date_files, start_dt, end_dt)

    if not date_files:
        return pd.DataFrame()

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
        df = df.set_index(["date", "code"]).sort_index()

        if fields is not None:
            # 兼容米筐字段名 today
            field_map = {"today": "turnover_rate", "turnover_rate": "turnover_rate"}
            field_list = fields if isinstance(fields, list) else [fields]
            cols = [field_map[f] for f in field_list if f in field_map]
            cols = [c for c in cols if c in df.columns]
            if cols:
                df = df[cols]

    if expect_df:
        return df
    return df.unstack("code")
