"""
因子分析模组 - 数据清洗与远期收益计算

参考 Alphalens 的 get_clean_factor_and_forward_returns 实现，
对因子数据、价格数据进行对齐清洗，计算多周期远期收益和分位数分组。
"""

import pandas as pd
import numpy as np
from typing import Union, List, Optional
import warnings

from .data import get_factor_prices, get_stock_pool


def _compute_forward_returns(
    prices: pd.DataFrame,
    periods: List[int],
) -> pd.DataFrame:
    """
    计算多周期远期收益

    Parameters
    ----------
    prices : pd.DataFrame
        收盘价数据，MultiIndex (date, code)，单列 close
    periods : list[int]
        远期收益周期列表 (交易日)

    Returns
    -------
    pd.DataFrame
        远期收益，MultiIndex (date, code)，列名形如 "period_1", "period_5"
    """
    # 转为宽格式便于按日期 shift
    close_wide = prices["close"].unstack("code").sort_index()

    # 获取交易日列表 (用于按交易日 shift，而非按自然日)
    trading_dates = close_wide.index

    # 计算每个周期的远期收益
    forward_returns = {}
    for period in periods:
        # 远期收益 = P[t+period] / P[t] - 1
        # 使用 shift(-period) 获取未来第 period 个交易日的价格
        future_price = close_wide.shift(-period)
        fwd_ret = future_price / close_wide - 1.0
        forward_returns[f"period_{period}"] = fwd_ret.stack()

    # 合并为 DataFrame
    result = pd.DataFrame(forward_returns)
    result.index.names = ["date", "code"]

    return result


def _quantize_factor(
    factor: pd.Series,
    quantiles: int = 5,
    min_stocks: int = 10,
) -> pd.Series:
    """
    对因子值按横截面分位数分组

    Parameters
    ----------
    factor : pd.Series
        因子数据，MultiIndex (date, code)
    quantiles : int, default 5
        分组数量
    min_stocks : int, default 10
        每日最少股票数，低于此数则当日不分组

    Returns
    -------
    pd.Series
        分组标签 (1 到 quantiles)，MultiIndex (date, code)
        同名: "factor_quantile"
    """
    def _quantile_group(group):
        # 过滤无效值
        valid = group.dropna()
        if len(valid) < min_stocks:
            return pd.Series(np.nan, index=group.index)
        # 使用 pd.qcut 分组，labels=False 返回 0~quantiles-1
        try:
            bins = pd.qcut(valid, q=quantiles, labels=False, duplicates="drop")
            # 转换为 1-based
            bins = bins + 1
            return pd.Series(bins, index=valid.index).reindex(group.index)
        except ValueError:
            # 当因子值全部相同时 qcut 会失败
            return pd.Series(np.nan, index=group.index)

    result = factor.groupby(level=0, group_keys=False).apply(_quantile_group)
    result.name = "factor_quantile"
    return result


def get_clean_factor_and_forward_returns(
    factor: Union[pd.Series, pd.DataFrame],
    prices: Optional[pd.DataFrame] = None,
    periods: List[int] = (1, 5, 10, 20),
    quantiles: int = 5,
    stock_pool: Optional[Union[str, List[str]]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_stocks_per_day: int = 10,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    数据清洗主函数：对齐因子与价格，计算远期收益和分组

    Parameters
    ----------
    factor : pd.Series or pd.DataFrame
        因子数据，MultiIndex (date, code)
        若为 DataFrame，取第一列
    prices : pd.DataFrame, optional
        价格数据 (含 close 列)，MultiIndex (date, code)
        None 则自动从 local_api 获取
    periods : list[int], default (1, 5, 10, 20)
        远期收益周期 (交易日)
    quantiles : int, default 5
        分组数量
    stock_pool : str or list, optional
        股票池，用于获取价格数据。None 则使用因子中所有股票
    start_date : str, optional
        因子分析开始日期
    end_date : str, optional
        因子分析结束日期
    min_stocks_per_day : int, default 10
        每日最少股票数，低于此数的日期会被剔除
    drop_na : bool, default True
        是否丢弃因子值或收益为 NaN 的样本

    Returns
    -------
    pd.DataFrame
        清洗后的数据，MultiIndex (date, code)，列:
        - "factor": 因子值
        - "factor_quantile": 分组标签 (1~quantiles)
        - "period_1", "period_5", ...: 各周期远期收益
    """
    # 统一因子为 Series
    if isinstance(factor, pd.DataFrame):
        factor = factor.iloc[:, 0]
    factor = factor.copy()
    factor.name = "factor"

    # 统一 index 名称
    if factor.index.names != ["date", "code"]:
        # 尝试重命名
        factor.index.names = ["date", "code"]

    # 标准化股票代码格式 (支持 SH600000 / 600000.XSHG 等多种格式)
    from .data import normalize_factor_codes
    factor = normalize_factor_codes(factor)

    # 确保索引排序
    factor = factor.sort_index()

    # 过滤日期范围
    factor_dates = factor.index.get_level_values(0)
    if start_date is not None:
        factor = factor[factor_dates >= pd.Timestamp(start_date)]
    if end_date is not None:
        factor = factor[factor_dates <= pd.Timestamp(end_date)]

    if factor.empty:
        raise ValueError("因子数据为空，请检查日期范围和股票池")

    # 获取价格数据
    if prices is None:
        max_period = max(periods) if periods else 20
        prices = get_factor_prices(
            factor=factor,
            stock_pool=stock_pool,
            start_date=start_date,
            end_date=end_date,
            fields=["close"],
            max_forward_period=max_period,
        )
    else:
        # 确保 prices 是 MultiIndex 且含 close 列
        if "close" not in prices.columns:
            raise ValueError("prices 必须包含 'close' 列")
        prices = prices[["close"]].copy()

    prices = prices.sort_index()

    # 计算远期收益
    forward_returns = _compute_forward_returns(prices, list(periods))

    # 对齐因子和远期收益
    # 使用 join 保证只保留两者都有的 (date, code)
    combined = factor.to_frame().join(forward_returns, how="inner")

    # 剔除每日股票数过少的日期
    daily_counts = combined.groupby(level=0).size()
    valid_dates = daily_counts[daily_counts >= min_stocks_per_day].index
    combined = combined[combined.index.get_level_values(0).isin(valid_dates)]

    if combined.empty:
        raise ValueError("对齐后数据为空，请检查因子和价格数据的日期/股票代码是否匹配")

    # 分组
    quantile_labels = _quantize_factor(
        combined["factor"],
        quantiles=quantiles,
        min_stocks=min_stocks_per_day,
    )
    combined["factor_quantile"] = quantile_labels

    # 丢弃无效值
    if drop_na:
        # 丢弃因子值为 NaN 的样本
        combined = combined.dropna(subset=["factor"])
        # 丢弃分组为 NaN 的样本
        combined = combined.dropna(subset=["factor_quantile"])
        combined["factor_quantile"] = combined["factor_quantile"].astype(int)

    # 统计信息
    n_dates = combined.index.get_level_values(0).nunique()
    n_stocks = combined.index.get_level_values(1).nunique()
    warnings.warn(
        f"数据清洗完成: {n_dates} 个交易日, {n_stocks} 只股票, "
        f"{len(combined)} 条样本, {quantiles} 分组",
        UserWarning,
    )

    return combined
