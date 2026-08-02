"""
本地化因子挖掘项目 - 本地量价数据加载

完全脱离 qlib / rqagent: 直接从本地 F:\\Trade_data\\stock_price 的 parquet 文件
(通过项目自有的 local_api) 一次性加载全A股后复权量价数据到内存,
构造成 宽表(日期×股票代码) 供因子表达式引擎快速计算。
"""

import time

import numpy as np
import pandas as pd

import local_api

from . import console

FIELDS = ["open", "close", "high", "low", "volume", "total_turnover"]


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

        self.n_dates = self.close.shape[0]
        self.n_stocks = self.close.shape[1]
        self.date_range = (str(self.close.index.min().date()), str(self.close.index.max().date()))
        console.log(
            f"    数据加载完成: {self.n_dates} 个交易日 × {self.n_stocks} 只股票, "
            f"区间 {self.date_range[0]} ~ {self.date_range[1]}, "
            f"耗时 {time.time()-t0:.1f} 秒"
        )

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
