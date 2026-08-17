"""A股公告特征数据本地查询接口（数据源：巨潮资讯网 cninfo.com.cn）

数据由 download/announcements.py 每日下载，存储于
F:\\Trade_data\\announcements\\YYYYMMDD.parquet（每个自然日一个文件）。
"""
import os
from functools import lru_cache

import pandas as pd

from .config import get_data_path, ANNOUNCEMENTS_DIR
from ._utils import normalize_codes, get_existing_date_files, filter_dates_by_range

# 公告大类清单（与 download/announcements.py 保持一致）
ANNOUNCEMENT_CATEGORIES = [
    "年报", "半年报", "一季报", "三季报", "业绩预告", "权益分派",
    "董事会", "监事会", "股东会", "日常经营", "公司治理", "中介报告",
    "首发", "增发", "股权激励", "配股", "解禁", "公司债", "可转债",
    "其他融资", "股权变动", "补充更正", "澄清致歉", "风险提示",
    "特别处理和退市", "退市整理期",
]


@lru_cache(maxsize=1)
def _date_files():
    """已有的公告日度文件列表 [(date_str, filepath)]（进程内缓存）"""
    return tuple(get_existing_date_files(get_data_path(ANNOUNCEMENTS_DIR)))


@lru_cache(maxsize=4096)
def _load_daily(filepath):
    return pd.read_parquet(filepath)


def refresh_announcement_cache():
    """新下载公告文件后调用，使缓存生效"""
    _date_files.cache_clear()
    _load_daily.cache_clear()


def get_announcement_categories():
    """返回可用的公告类别清单"""
    return list(ANNOUNCEMENT_CATEGORIES)


def get_announcements(start_date=None, end_date=None, order_book_ids=None,
                      category=None, searchkey=None):
    """查询 A 股公告特征数据

    Parameters
    ----------
    start_date / end_date : 日期范围（自然日），支持 "YYYY-MM-DD"、"YYYYMMDD"
        或 datetime 对象，缺省为全部已有数据范围
    order_book_ids : str 或 list，股票代码（如 "000001.XSHE"、"600519.XSHG"，
        6 位数字代码自动补后缀）
    category : str，公告类别名（如 "年报"、"业绩预告"），
        匹配公告所属任一类别即命中，清单见 get_announcement_categories()
    searchkey : str，公告标题关键词（子串匹配）

    Returns
    -------
    DataFrame，index 为 (date, order_book_id)，列：
        sec_code   6 位数字代码
        sec_name   股票简称
        title      公告标题
        category   公告类别（多类别用 ";" 连接）
        type_codes 巨潮原始分类代码串
        url        公告 PDF 链接
        ann_id     巨潮公告唯一编号
        ann_time   公告发布时间（北京时间）
    """
    date_files = filter_dates_by_range(_date_files(), start_date, end_date)
    if not date_files:
        return pd.DataFrame()

    frames = []
    for _, filepath in date_files:
        df = _load_daily(filepath)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)

    if order_book_ids is not None:
        codes = normalize_codes(order_book_ids)
        result = result[result["order_book_id"].isin(codes)]
    if category:
        result = result[result["category"].str.split(";").apply(
            lambda cats: category in cats)]
    if searchkey:
        result = result[result["title"].str.contains(searchkey, regex=False,
                                                     na=False)]
    if result.empty:
        return pd.DataFrame()

    result["date"] = pd.to_datetime(result["date"])
    return result.set_index(["date", "order_book_id"]).sort_index()
