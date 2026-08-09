"""
本地化因子挖掘项目 - 本地量价数据加载

完全脱离 qlib / rqagent: 直接从本地 F:\\Trade_data\\stock_price 的 parquet 文件
(通过项目自有的 local_api) 一次性加载全A股后复权量价数据到内存,
构造成 宽表(日期×股票代码) 供因子表达式引擎快速计算。

同时加载每日可交易状态(F:\\Trade_data\\tradable_status, 由 update/tradable_status
模块生成): 剔除 ST / 停牌 / 上市未满一年 / 涨跌停 的股票, 避免因子收益虚高。
"""

import glob
import os
import time

import numpy as np
import pandas as pd

import local_api

from . import console

FIELDS = ["open", "close", "high", "low", "volume", "total_turnover"]

# 每日可交易状态数据目录(update/tradable_status 模块产出, YYYYMMDD.parquet)
TRADABLE_STATUS_DIR = r"F:\Trade_data\tradable_status"


class MarketData:
    """内存中的全市场行情数据(宽表: index=日期, columns=股票代码)"""

    def __init__(self, cfg):
        t0 = time.time()
        console.log(f"    读取本地量价数据: {cfg.data_start_date} 至今, 全A股, 后复权...")
        # order_book_ids 传空列表 => 不做代码过滤, 读取全部股票
        df = local_api.get_price(
            order_book_ids=[],
            start_date=cfg.data_start_date,
            end_date=None,
            fields=FIELDS,
            adjust_type="post",
            expect_df=True,
        )
        if df.empty:
            raise RuntimeError("本地量价数据为空, 请检查 F:\\Trade_data\\stock_price 是否存在")

        console.log(f"    原始数据: {len(df)} 条记录, 转宽表中...")
        self.open = df["open"].unstack("code").astype("float32")
        self.close = df["close"].unstack("code").astype("float32")
        self.high = df["high"].unstack("code").astype("float32")
        self.low = df["low"].unstack("code").astype("float32")
        self.volume = df["volume"].unstack("code").astype("float32")
        self.amount = df["total_turnover"].unstack("code").astype("float32")
        # 日收益率(收盘价涨跌幅)
        self.ret = self.close.pct_change(fill_method=None)

        # 供 factor_analysis 复用的长表收盘价(MultiIndex: date, code)
        self.close_long = df[["close"]].copy()
        self.close_long.index.names = ["date", "code"]
        del df

        # 每日可交易状态宽表(True=可交易), 用于剔除 ST/停牌/次新/涨跌停
        self.tradable = self._load_tradable_status()

        self.n_dates = self.close.shape[0]
        self.n_stocks = self.close.shape[1]
        self.date_range = (str(self.close.index.min().date()), str(self.close.index.max().date()))
        console.log(
            f"    数据加载完成: {self.n_dates} 个交易日 × {self.n_stocks} 只股票, "
            f"区间 {self.date_range[0]} ~ {self.date_range[1]}, "
            f"耗时 {time.time()-t0:.1f} 秒"
        )

    def _load_tradable_status(self) -> pd.DataFrame:
        """从 F:\\Trade_data\\tradable_status 加载每日可交易状态宽表(日期×股票, bool)"""
        t0 = time.time()
        files = sorted(glob.glob(os.path.join(TRADABLE_STATUS_DIR, "*.parquet")))
        if not files:
            console.log("    [警告] tradable_status 数据目录为空, 交易资格遮盖失效。")
            return pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype=bool).fillna(True)
        frames = []
        for fp in files:
            day = pd.read_parquet(fp, columns=["code", "tradable"])
            day["date"] = pd.Timestamp(os.path.basename(fp)[:8])
            frames.append(day)
        df = pd.concat(frames, ignore_index=True)
        wide = df.pivot_table(index="date", columns="code", values="tradable", aggfunc="first")
        wide = wide.astype(bool)
        console.log(
            f"    可交易状态加载完成: {len(wide)} 个交易日, 耗时 {time.time()-t0:.1f} 秒"
            f" (剔除 ST/停牌/次新/涨跌停)")
        return wide

    def var(self, name: str) -> pd.DataFrame:
        """按变量名($open等)返回对应宽表"""
        mapping = {
            "$open": self.open,
            "$close": self.close,
            "$high": self.high,
            "$low": self.low,
            "$volume": self.volume,
            "$amount": self.amount,
            "$return": self.ret,
        }
        key = name.lower()
        if key not in mapping:
            raise KeyError(f"未知变量: {name}")
        return mapping[key]
