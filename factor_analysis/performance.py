"""
因子分析模组 - 性能计算 (向量化优化版)

优化点:
    - IC 计算改为 numpy 矩阵运算 (11.9s → ~0.3s)
    - 换手率计算向量化 (3.7s → ~0.1s)
    - 自相关计算向量化 (2.9s → ~0.1s)
    - 累计收益改用单利
    - 换手率默认双边
    - 新增多头收益计算 (替代多空)
"""

import pandas as pd
import numpy as np
from typing import Optional
from scipy import stats


# =============================================================================
# 向量化辅助函数
# =============================================================================

def _rowwise_corr(a: np.ndarray, b: np.ndarray, min_samples: int = 5) -> np.ndarray:
    """
    逐行计算两个 2D 数组的相关系数 (Pearson)，处理 NaN

    Parameters
    ----------
    a, b : np.ndarray
        形状 (n_dates, n_stocks)
    min_samples : int
        每行最少有效样本数

    Returns
    -------
    np.ndarray
        形状 (n_dates,)，每行的 Pearson 相关系数
    """
    valid = ~(np.isnan(a) | np.isnan(b))
    n = valid.sum(axis=1).astype(np.float64)

    # 填充 NaN 为 0 用于求和
    a_f = np.where(valid, a, 0.0)
    b_f = np.where(valid, b, 0.0)

    n_safe = np.where(n > 0, n, 1.0)
    a_mean = a_f.sum(axis=1) / n_safe
    b_mean = b_f.sum(axis=1) / n_safe

    a_c = np.where(valid, a_f - a_mean[:, None], 0.0)
    b_c = np.where(valid, b_f - b_mean[:, None], 0.0)

    cov = (a_c * b_c).sum(axis=1)
    a_var = np.sqrt((a_c ** 2).sum(axis=1))
    b_var = np.sqrt((b_c ** 2).sum(axis=1))

    denom = a_var * b_var
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(denom > 0, cov / denom, np.nan)
    corr = np.where(n >= min_samples, corr, np.nan)

    return corr


def _to_wide(series: pd.Series) -> pd.DataFrame:
    """将 MultiIndex Series 转为宽格式 DataFrame (date × code)"""
    return series.unstack("code")


def _period_days(period: str) -> int:
    """从列名 'period_5' 解析出前向收益的天数 5"""
    try:
        return int(str(period).split("_")[-1])
    except (ValueError, IndexError):
        return 1


# =============================================================================
# IC (信息系数) 计算 - 向量化
# =============================================================================

def calc_information_coefficient(
    clean_data: pd.DataFrame,
    method: str = "spearman",
) -> pd.DataFrame:
    """
    计算因子 IC 时序 (向量化)

    Parameters
    ----------
    clean_data : pd.DataFrame
        清洗数据，含 "factor" 和 "period_*" 列
    method : str, default "spearman"
        "spearman" (Rank IC) 或 "pearson" (普通 IC)

    Returns
    -------
    pd.DataFrame
        IC 时序，index=date, columns=["period_1", ...]
    """
    if method not in ("spearman", "pearson"):
        raise ValueError("method 必须是 'spearman' 或 'pearson'")

    period_cols = [c for c in clean_data.columns if c.startswith("period_")]
    if not period_cols:
        raise ValueError("clean_data 中未找到 period_* 远期收益列")

    # 转宽格式
    factor_wide = _to_wide(clean_data["factor"])

    # Spearman: 对每行做 rank
    if method == "spearman":
        factor_values = factor_wide.rank(axis=1).values
    else:
        factor_values = factor_wide.values

    ic_results = {}
    for period_col in period_cols:
        return_wide = _to_wide(clean_data[period_col])
        if method == "spearman":
            return_values = return_wide.rank(axis=1).values
        else:
            return_values = return_wide.values

        corr = _rowwise_corr(factor_values, return_values)
        ic_results[period_col] = pd.Series(corr, index=factor_wide.index)

    ic_df = pd.DataFrame(ic_results)
    ic_df.index.name = "date"
    return ic_df


# =============================================================================
# 分组收益计算
# =============================================================================

def calc_group_returns(
    clean_data: pd.DataFrame,
    period: str = "period_1",
    by_quantile: bool = True,
    benchmark_returns: Optional[pd.Series] = None,
    excess: bool = True,
    normalize: bool = False,
) -> pd.DataFrame:
    """
    计算分组收益时序

    Parameters
    ----------
    clean_data : pd.DataFrame
        清洗数据
    period : str, default "period_1"
        使用的远期收益列
    benchmark_returns : pd.Series, optional
        基准收益时序。None 且 excess=True 时使用全市场等权均值
    excess : bool, default True
        是否计算超额收益
    normalize : bool, default False
        是否将 n 日前向收益除以 n 转为日度等效收益。
        n 日前向收益在时间轴上重叠, 直接按年求和/累加会放大 n 倍;
        除以 n 后任意周期的收益量纲统一(日度等效), 可直接跨周期比较,
        用于观察信号衰减对收益的影响。

    Returns
    -------
    pd.DataFrame
        分组收益时序，index=date, columns=[1, 2, ..., quantiles]
    """
    if period not in clean_data.columns:
        raise ValueError(f"clean_data 中未找到 '{period}' 列")

    group_returns = clean_data.groupby(
        [clean_data.index.get_level_values(0), "factor_quantile"]
    )[period].mean()
    group_returns = group_returns.unstack("factor_quantile")
    group_returns.index.name = "date"
    group_returns.columns.name = "quantile"

    if excess:
        if benchmark_returns is not None:
            benchmark = benchmark_returns.reindex(group_returns.index).fillna(0.0)
            if isinstance(benchmark, pd.DataFrame):
                benchmark = benchmark.iloc[:, 0]
        else:
            market_mean = clean_data.groupby(level=0)[period].mean()
            benchmark = market_mean.reindex(group_returns.index).fillna(0.0)
        group_returns = group_returns.subtract(benchmark, axis=0)

    if normalize:
        group_returns = group_returns / _period_days(period)

    return group_returns


def calc_long_returns(
    clean_data: pd.DataFrame,
    period: str = "period_1",
    quantile: Optional[int] = None,
    benchmark_returns: Optional[pd.Series] = None,
    excess: bool = True,
    normalize: bool = False,
) -> pd.Series:
    """
    计算多头组合收益时序 (最高分组)

    Parameters
    ----------
    clean_data : pd.DataFrame
        清洗数据
    period : str, default "period_1"
        使用的远期收益列
    quantile : int, optional
        指定分组。None 则取最高分组
    benchmark_returns : pd.Series, optional
        基准收益时序
    excess : bool, default True
        是否计算超额收益
    normalize : bool, default False
        是否将 n 日前向收益除以 n 转为日度等效收益(使任意周期收益可比)

    Returns
    -------
    pd.Series
        多头组合收益时序，index=date
    """
    group_returns = calc_group_returns(
        clean_data, period=period,
        benchmark_returns=benchmark_returns, excess=excess,
        normalize=normalize,
    )

    if group_returns.empty:
        return pd.Series(dtype=float)

    if quantile is None:
        q = group_returns.columns.max()
    else:
        q = quantile

    long_returns = group_returns[q]
    long_returns.name = "long"
    return long_returns


def calc_long_short_returns(
    clean_data: pd.DataFrame,
    period: str = "period_1",
    benchmark_returns: Optional[pd.Series] = None,
    excess: bool = True,
    normalize: bool = False,
) -> pd.Series:
    """
    计算多空组合收益时序 (最高分组 - 最低分组)

    注意: A股多空意义有限，建议使用 calc_long_returns。

    normalize : bool, default False
        是否将 n 日前向收益除以 n 转为日度等效收益(使任意周期收益可比)

    Returns
    -------
    pd.Series
        多空组合收益时序
    """
    group_returns = calc_group_returns(
        clean_data, period=period,
        benchmark_returns=benchmark_returns, excess=excess,
        normalize=normalize,
    )
    if group_returns.empty:
        return pd.Series(dtype=float)
    max_q = group_returns.columns.max()
    min_q = group_returns.columns.min()
    ls = group_returns[max_q] - group_returns[min_q]
    ls.name = "long_short"
    return ls


def calc_cumulative_returns(
    returns: pd.DataFrame or pd.Series,
    simple_interest: bool = True,
) -> pd.DataFrame or pd.Series:
    """
    计算累计收益

    Parameters
    ----------
    returns : pd.DataFrame or pd.Series
        日频收益
    simple_interest : bool, default True
        True: 单利 (净值 = 1 + cumsum(returns))
        False: 复利 (净值 = cumprod(1 + returns))

    Returns
    -------
    pd.DataFrame or pd.Series
        净值 (起点为 1)
    """
    if simple_interest:
        return 1 + returns.cumsum()
    else:
        return (1 + returns).cumprod()


# =============================================================================
# 分组换手率计算 - 向量化 + 默认双边
# =============================================================================

def calc_group_turnover(
    clean_data: pd.DataFrame,
    double_sided: bool = True,
    period: int = 1,
    normalize: bool = False,
) -> pd.DataFrame:
    """
    计算分组换手率时序 (向量化)

    换手率 = 1 - (相隔 period 个交易日仍留在同组的股票数 / 当期该组总股票数)
    双边换手率 = 2 × 单边换手率

    period 应与因子收益的前向周期一致: 例如因子收益用5日前向收益,
    则模拟5日调仓, 换手率也应比较相隔5个交易日的两期分组。

    Parameters
    ----------
    clean_data : pd.DataFrame
        清洗数据，含 "factor_quantile" 列
    double_sided : bool, default True
        是否计算双边换手率 (默认 True)
    period : int, default 1
        调仓间隔(交易日)。1=逐日调仓; 5=每5个交易日调仓一次
    normalize : bool, default False
        是否将 period 日换手率除以 period 转为日度等效换手率。
        与分组收益的 normalize 口径一致: n日换手率直接×年化天数会放大n倍,
        除以n后任意调仓周期的换手率量纲统一, 可直接跨周期比较。

    Returns
    -------
    pd.DataFrame
        分组换手率时序，index=date, columns=[1, 2, ..., quantiles]
    """
    if "factor_quantile" not in clean_data.columns:
        raise ValueError("clean_data 中未找到 'factor_quantile' 列")
    period = max(1, int(period))

    # 转宽格式
    quantile_wide = _to_wide(clean_data["factor_quantile"])
    prev_q = quantile_wide.shift(period)

    # 所有分组标签
    all_quantiles = sorted(
        q for q in quantile_wide.values.flatten()
        if not np.isnan(q)
    )
    all_quantiles = sorted(set(all_quantiles))

    results = {}
    for q in all_quantiles:
        q = int(q)
        curr_mask = (quantile_wide == q)
        prev_mask = (prev_q == q)
        # 两期都在 q 组的股票数
        held = (curr_mask & prev_mask).sum(axis=1)
        # 当期 q 组总股票数
        total = curr_mask.sum(axis=1)
        # 单边换手率
        total_safe = total.replace(0, np.nan)
        turnover = 1.0 - held / total_safe
        if double_sided:
            turnover = turnover * 2.0
        results[q] = turnover

    turnover_df = pd.DataFrame(results)
    turnover_df.index.name = "date"
    turnover_df.columns.name = "quantile"
    # 前 period 行没有可比对的上一期分组, 置为 NaN 后删除
    no_prev = prev_q.notna().sum(axis=1) == 0
    turnover_df.loc[no_prev, :] = np.nan
    turnover_df = turnover_df.dropna(how="all")

    if normalize and period > 1:
        turnover_df = turnover_df / period

    return turnover_df


def calc_quantile_turnover(
    clean_data: pd.DataFrame,
    quantile: Optional[int] = None,
    double_sided: bool = True,
    period: int = 1,
    normalize: bool = False,
) -> pd.DataFrame or pd.Series:
    """计算指定分组或全部分组的换手率 (period/normalize 见 calc_group_turnover)"""
    turnover = calc_group_turnover(
        clean_data, double_sided=double_sided, period=period, normalize=normalize,
    )
    if quantile is not None:
        if quantile in turnover.columns:
            return turnover[quantile]
        raise ValueError(f"分组 {quantile} 不存在")
    return turnover


# =============================================================================
# 因子自相关系数 - 向量化
# =============================================================================

def calc_factor_autocorrelation(
    clean_data: pd.DataFrame,
    lag: int = 1,
) -> pd.Series:
    """
    计算因子自相关系数时序 (向量化)

    对相邻交易日，计算因子值的横截面 Spearman 相关系数。

    Parameters
    ----------
    clean_data : pd.DataFrame
        清洗数据
    lag : int, default 1
        滞后期 (交易日)

    Returns
    -------
    pd.Series
        自相关系数时序，index=date
    """
    factor_wide = _to_wide(clean_data["factor"])
    factor_rank = factor_wide.rank(axis=1)
    prev_rank = factor_rank.shift(lag)

    corr = _rowwise_corr(factor_rank.values, prev_rank.values)
    result = pd.Series(corr, index=factor_rank.index, name="factor_autocorr")
    result = result.dropna()
    return result
