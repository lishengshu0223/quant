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

        # A股流通股本(股)宽表: 由 local_api.get_shares 提供(本地 F:\Trade_data\shares)。
        # 经验证: 米筐 today 换手率口径 = 成交量 / A股流通股本(circulation_a) * 100,
        # 而非总股本 total (total 口径全市场偏差约 48.5%)。
        # 供分钟换手率使用: 分钟换手率(%) = 该分钟成交量 / 当日流通股本 * 100
        self.circulation = self._load_circulation()

        # 日频换手率(%): 公式补全 = 原始成交量 / 当日流通股本 * 100 (计算式优先)。
        # 注意: 主行情 $volume 为后复权口径(adjvolume), 与换手率公式的"原始成交量"量纲不一致,
        # 因此必须用未复权的原始成交量(股)计算(与分钟成交量同为"股"口径)。
        # 股本数据缺失日(如早期停牌/数据未覆盖)回退本地 turnover 字段(米筐预计算, 已验证同口径)。
        raw_vol = self._load_raw_volume()
        with np.errstate(divide="ignore", invalid="ignore"):
            calc_to = raw_vol.to_numpy(dtype=np.float64) / self.circulation.to_numpy(dtype=np.float64) * 100.0
        calc_to = pd.DataFrame(calc_to, index=self.close.index, columns=self.close.columns,
                               dtype="float32")
        self.turnover = calc_to.combine_first(self._load_turnover())

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

    def _load_turnover(self) -> pd.DataFrame:
        """加载日频换手率宽表(日期×股票, 单位%, float32), 与行情宽表按日期/代码对齐。
        换手率数据缺失的股票(停牌/ST等)为 NaN, 因子计算时自然剔除。"""
        t0 = time.time()
        try:
            df = local_api.get_turnover_rate(
                order_book_ids=None,
                start_date=str(self.close.index.min().date()),
                end_date=None,
                expect_df=True,
            )
        except Exception as e:
            console.log(f"    [警告] 换手率数据加载失败({e}), 因子中 $turnover 将恒为 NaN。")
            return pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype="float32")
        if df is None or df.empty:
            console.log("    [警告] 本地换手率数据为空, 因子中 $turnover 将恒为 NaN。")
            return pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype="float32")
        wide = df["turnover_rate"].unstack("code").astype("float32")
        wide = wide.reindex(index=self.close.index, columns=self.close.columns)
        console.log(f"    换手率加载完成: {wide.shape[0]} 个交易日 × {wide.shape[1]} 只股票, "
                    f"非空率 {wide.notna().mean().mean():.1%}, 耗时 {time.time()-t0:.1f} 秒")
        return wide

    def _load_circulation(self) -> pd.DataFrame:
        """加载 A股流通股本宽表(日期×股票, 单位股, float32), 与行情宽表对齐。
        经验证米筐 today 换手率 = 成交量 / circulation_a * 100, 故分钟换手率分母取
        A股流通股本(而非总股本)。股本在无送转/增发等事件时恒定,
        缺失日(停牌等)前向填充取最近值。"""
        t0 = time.time()
        try:
            df = local_api.get_shares(
                order_book_ids=None,
                start_date=str(self.close.index.min().date()),
                end_date=None,
                fields=["circulation_a"],
                expect_df=True,
            )
        except Exception as e:
            console.log(f"    [警告] 股本数据加载失败({e}), $turnover/分钟换手率将不可用。")
            return pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype="float32")
        if df is None or df.empty:
            console.log("    [警告] 本地股本数据为空, $turnover/分钟换手率将不可用。")
            return pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype="float32")
        wide = df["circulation_a"].unstack("code").astype("float32")
        wide = wide.reindex(index=self.close.index, columns=self.close.columns)
        console.log(f"    股本数据加载完成: {wide.shape[0]} 个交易日 × {wide.shape[1]} 只股票, "
                    f"非空率 {wide.notna().mean().mean():.1%}, 耗时 {time.time()-t0:.1f} 秒")
        return wide.ffill(axis=0)

    def _load_raw_volume(self) -> pd.DataFrame:
        """加载未复权原始成交量宽表(日期×股票, 单位股, float32), 与行情宽表对齐。
        供总股本反推(换手率=原始成交量/总股本), 与分钟成交量同为"股"口径。"""
        t0 = time.time()
        try:
            df = local_api.get_price(
                order_book_ids=None,
                start_date=str(self.close.index.min().date()),
                end_date=None,
                fields=["volume"],
                adjust_type="none",
                expect_df=True,
            )
        except Exception as e:
            console.log(f"    [警告] 原始成交量加载失败({e}), 总股本将不可用。")
            return pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype="float32")
        if df is None or df.empty:
            console.log("    [警告] 本地原始成交量数据为空, 总股本将不可用。")
            return pd.DataFrame(index=self.close.index, columns=self.close.columns, dtype="float32")
        wide = df["volume"].unstack("code").astype("float32")
        return wide.reindex(index=self.close.index, columns=self.close.columns)

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
            "$turnover": self.turnover,
        }
        key = name.lower()
        if key not in mapping:
            raise KeyError(f"未知变量: {name}")
        return mapping[key]
