"""
因子分析模组 - 统计量计算

特性:
    - IC 统计: 只保留 IC 均值和 ICIR，百分号格式
    - 收益统计: 单利回撤 (从最高点亏损的绝对值 / 初始资金)
    - 多头统计: 相对全市场和各宽基指数的超额
    - 分年度统计: 多头收益 + 双边换手率
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional


# =============================================================================
# IC 统计量 (简化: 只保留 IC 均值和 ICIR)
# =============================================================================

def calc_ic_stats(
    ic_data: pd.DataFrame,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    计算 IC 时序的统计量 (简化版)

    只保留 IC 均值和 ICIR。

    Parameters
    ----------
    ic_data : pd.DataFrame
        IC 时序数据
    periods_per_year : int, default 252

    Returns
    -------
    pd.DataFrame
        IC 统计量，index=["IC均值", "ICIR"], columns=周期
        数值以小数形式存储，显示时用百分号格式化
    """
    stats_records = {}

    for period_col in ic_data.columns:
        ic_series = ic_data[period_col].dropna()
        n = len(ic_series)

        if n == 0:
            stats_records[period_col] = {
                "IC均值": np.nan,
                "ICIR": np.nan,
            }
            continue

        mean_ic = ic_series.mean()
        std_ic = ic_series.std(ddof=1) if n > 1 else np.nan
        icir = mean_ic / std_ic if std_ic and std_ic != 0 else np.nan

        stats_records[period_col] = {
            "IC均值": mean_ic,
            "ICIR": icir,
        }

    stats_df = pd.DataFrame(stats_records)
    stats_df.index.name = "统计量"
    return stats_df


def format_ic_stats(ic_stats: pd.DataFrame) -> pd.DataFrame:
    """
    格式化 IC 统计量

    IC均值: 百分号形式 (保留两位小数)
    ICIR: 普通数值 (保留两位小数, ICIR=均值/标准差 为比率, 非百分比)

    Parameters
    ----------
    ic_stats : pd.DataFrame
        calc_ic_stats 返回的统计量

    Returns
    -------
    pd.DataFrame
        格式化后的字符串 DataFrame
    """
    formatted = ic_stats.copy().astype(object)
    for col in formatted.columns:
        for idx in formatted.index:
            val = formatted.loc[idx, col]
            if pd.isna(val):
                formatted.loc[idx, col] = "N/A"
            elif idx == "ICIR":
                # ICIR 是比率 (均值/标准差), 用普通数值显示
                formatted.loc[idx, col] = f"{val:.2f}"
            else:
                # IC均值 用百分号显示
                formatted.loc[idx, col] = f"{val*100:.2f}%"
    return formatted


# =============================================================================
# 收益统计量 (单利回撤)
# =============================================================================

def _calc_return_stats_single(
    returns: pd.Series,
    periods_per_year: int = 252,
    name: str = "",
) -> dict:
    """计算单条收益时序的统计量 (单利模式)"""
    returns = returns.dropna()
    n = len(returns)

    if n == 0:
        return {
            "name": name,
            "年化收益": np.nan,
            "年化波动": np.nan,
            "夏普比率": np.nan,
            "最大回撤": np.nan,
            "胜率": np.nan,
            "n": 0,
        }

    mean_ret = returns.mean()
    std_ret = returns.std(ddof=1) if n > 1 else np.nan

    # 单利年化: 均值 × 252
    annualized_return = mean_ret * periods_per_year
    annualized_vol = std_ret * np.sqrt(periods_per_year) if std_ret == std_ret else np.nan
    sharpe = annualized_return / annualized_vol if annualized_vol and annualized_vol != 0 else np.nan

    # 单利回撤: 净值 = 1 + cumsum(returns)
    # 回撤 = (峰值净值 - 当前净值) / 1.0 = 峰值净值 - 当前净值
    # (因为初始资金为1，所以绝对差值就是回撤比例)
    cum_returns = returns.cumsum()
    running_max = cum_returns.cummax()
    drawdown = running_max - cum_returns  # 正数，表示从高点的回撤
    max_drawdown = drawdown.max()

    win_rate = (returns > 0).mean()

    return {
        "name": name,
        "年化收益": annualized_return,
        "年化波动": annualized_vol,
        "夏普比率": sharpe,
        "最大回撤": max_drawdown,  # 单利回撤: 绝对值比例
        "胜率": win_rate,
        "n": n,
    }


def calc_returns_stats(
    group_returns: pd.DataFrame,
    periods_per_year: int = 252,
    long_returns: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    计算分组收益时序的统计量 (单利模式)

    Parameters
    ----------
    group_returns : pd.DataFrame
        分组收益时序
    periods_per_year : int, default 252
    long_returns : pd.Series, optional
        多头组合收益时序 (最高分组)

    Returns
    -------
    pd.DataFrame
        收益统计量，index=分组名
    """
    stats_list = []

    for col in group_returns.columns:
        stats = _calc_return_stats_single(
            group_returns[col],
            periods_per_year=periods_per_year,
            name=f"Q{int(col)}",
        )
        stats_list.append(stats)

    # 多头组合统计量
    if long_returns is not None:
        long_stats = _calc_return_stats_single(
            long_returns,
            periods_per_year=periods_per_year,
            name="多头",
        )
        stats_list.append(long_stats)

    stats_df = pd.DataFrame(stats_list).set_index("name")
    return stats_df


def format_returns_stats(returns_stats: pd.DataFrame) -> pd.DataFrame:
    """
    格式化收益统计量 (百分号+两位小数)

    Parameters
    ----------
    returns_stats : pd.DataFrame
        calc_returns_stats 返回的统计量

    Returns
    -------
    pd.DataFrame
        格式化后的字符串 DataFrame
    """
    formatted = returns_stats.copy()
    pct_cols = ["年化收益", "年化波动", "最大回撤", "胜率"]
    float_cols = ["夏普比率"]

    for col in pct_cols:
        if col in formatted.columns:
            formatted[col] = formatted[col].apply(
                lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
            )
    for col in float_cols:
        if col in formatted.columns:
            formatted[col] = formatted[col].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
            )
    return formatted


# =============================================================================
# 换手率统计量
# =============================================================================

def calc_turnover_stats(
    turnover_data: pd.DataFrame,
    autocorr_data: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    计算换手率时序的统计量

    Parameters
    ----------
    turnover_data : pd.DataFrame
        换手率时序 (双边)
    autocorr_data : pd.Series, optional
        因子自相关时序

    Returns
    -------
    pd.DataFrame
        换手率统计量，index=分组名
    """
    stats_list = []

    for col in turnover_data.columns:
        turnover_series = turnover_data[col].dropna()
        n = len(turnover_series)

        if n == 0:
            stats_list.append({
                "name": f"Q{int(col)}",
                "平均换手率": np.nan,
                "换手率标准差": np.nan,
                "n": 0,
            })
            continue

        stats_list.append({
            "name": f"Q{int(col)}",
            "平均换手率": turnover_series.mean(),
            "换手率标准差": turnover_series.std(ddof=1) if n > 1 else np.nan,
            "n": n,
        })

    if autocorr_data is not None:
        autocorr = autocorr_data.dropna()
        if len(autocorr) > 0:
            stats_list.append({
                "name": "因子自相关",
                "平均换手率": autocorr.mean(),
                "换手率标准差": autocorr.std(ddof=1) if len(autocorr) > 1 else np.nan,
                "n": len(autocorr),
            })

    stats_df = pd.DataFrame(stats_list).set_index("name")
    return stats_df


def format_turnover_stats(turnover_stats: pd.DataFrame) -> pd.DataFrame:
    """格式化换手率统计量 (百分号+两位小数)"""
    formatted = turnover_stats.copy()
    pct_cols = ["平均换手率", "换手率标准差"]
    for col in pct_cols:
        if col in formatted.columns:
            formatted[col] = formatted[col].apply(
                lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
            )
    return formatted


# =============================================================================
# 分年度统计 (多头收益 + 双边换手率)
# =============================================================================

def calc_yearly_stats(
    group_returns: pd.DataFrame,
    turnover_data: pd.DataFrame,
    long_quantile: Optional[int] = None,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    计算分年度的分组超额收益和多头年化换手率

    每年展示所有分组 (Q1-Q5) 的年化超额收益, 用于观察分组单调性。
    同时展示多头 (最高分组) 的年化双边换手率 (日均双边换手率 × 252)。

    Parameters
    ----------
    group_returns : pd.DataFrame
        分组收益时序
    turnover_data : pd.DataFrame
        分组换手率时序 (双边; 逐日调仓或日度等效口径, 年化时均×periods_per_year)
    long_quantile : int, optional
        多头分组。None 则取最高分组
    periods_per_year : int, default 252

    Returns
    -------
    pd.DataFrame
        分年度统计, index=年份,
        columns=[Q1, Q2, ..., Q5, 多头年化换手率]
        收益为年化超额收益, 换手率为年化双边换手率 (日均×252)
    """
    if long_quantile is None:
        long_quantile = group_returns.columns.max()

    years = group_returns.index.year.unique()

    yearly_records = []
    for year in sorted(years):
        year_mask = group_returns.index.year == year
        year_group_returns = group_returns.loc[year_mask]

        if len(year_group_returns) == 0:
            continue

        n_days = len(year_group_returns)

        record = {"年份": int(year)}
        # 各分组的单利年化超额收益
        for q in group_returns.columns:
            q_int = int(q)
            year_q_returns = year_group_returns[q]
            annual_return = year_q_returns.sum() * (periods_per_year / n_days)
            record[f"Q{q_int}"] = annual_return

        # 多头年化双边换手率 (日均 × 252)
        year_turnover_mask = turnover_data.index.year == year
        if long_quantile in turnover_data.columns:
            year_turnover = turnover_data.loc[year_turnover_mask, long_quantile]
            avg_turnover_daily = year_turnover.mean() if len(year_turnover) > 0 else np.nan
            annualized_turnover = avg_turnover_daily * periods_per_year
        else:
            annualized_turnover = np.nan

        record["多头年化换手率"] = annualized_turnover

        yearly_records.append(record)

    result = pd.DataFrame(yearly_records).set_index("年份")
    return result


def format_yearly_stats(yearly_stats: pd.DataFrame) -> pd.DataFrame:
    """
    格式化分年度统计

    分组收益: 百分号+两位小数
    年化换手率: 浮点数 (年化, 几十到几百量级, 不用百分号)
    """
    formatted = yearly_stats.copy().astype(object)
    for col in formatted.columns:
        for idx in formatted.index:
            val = formatted.loc[idx, col]
            if pd.isna(val):
                formatted.loc[idx, col] = "N/A"
            elif "换手率" in col:
                # 年化换手率用浮点数 (几十到几百量级)
                formatted.loc[idx, col] = f"{val:.2f}"
            else:
                # 分组收益用百分号
                formatted.loc[idx, col] = f"{val*100:.2f}%"
    return formatted


# =============================================================================
# 多头多基准统计
# =============================================================================

def calc_long_stats_multi_benchmark(
    clean_data: pd.DataFrame,
    benchmark_dict: dict,
    period: str = "period_1",
    periods_per_year: int = 252,
    normalize: bool = False,
) -> pd.DataFrame:
    """
    计算多头组合相对多个基准的统计量

    Parameters
    ----------
    clean_data : pd.DataFrame
        清洗数据
    benchmark_dict : dict
        基准字典 {名称: 收益时序}
        如 {"全市场": None, "沪深300": bench300, "中证500": bench500, "中证1000": bench1000}
        None 表示使用全市场等权均值
    period : str, default "period_1"
    periods_per_year : int, default 252
    normalize : bool, default False
        是否将 n 日前向收益除以 n 转为日度等效收益(使任意周期收益可比)

    Returns
    -------
    pd.DataFrame
        多头多基准统计，index=基准名, columns=["年化收益", "夏普比率", "最大回撤", "胜率"]
    """
    from .performance import calc_long_returns

    stats_list = []

    for bench_name, bench_returns in benchmark_dict.items():
        long_ret = calc_long_returns(
            clean_data,
            period=period,
            benchmark_returns=bench_returns,
            excess=(bench_returns is not None or True),  # 始终计算超额
            normalize=normalize,
        )
        s = _calc_return_stats_single(
            long_ret,
            periods_per_year=periods_per_year,
            name=bench_name,
        )
        stats_list.append(s)

    stats_df = pd.DataFrame(stats_list).set_index("name")
    return stats_df


# =============================================================================
# 汇总统计
# =============================================================================

def summary_statistics(
    ic_stats: pd.DataFrame,
    returns_stats: pd.DataFrame,
    turnover_stats: pd.DataFrame,
    yearly_stats: Optional[pd.DataFrame] = None,
    long_multi_benchmark_stats: Optional[pd.DataFrame] = None,
) -> dict:
    """
    汇总因子分析的关键统计量

    Parameters
    ----------
    ic_stats : pd.DataFrame
        IC 统计量
    returns_stats : pd.DataFrame
        收益统计量
    turnover_stats : pd.DataFrame
        换手率统计量
    yearly_stats : pd.DataFrame, optional
        分年度统计
    long_multi_benchmark_stats : pd.DataFrame, optional
        多头多基准统计

    Returns
    -------
    dict
        关键统计量汇总
    """
    summary = {}

    # IC 汇总 (取 period_1)
    if "period_1" in ic_stats.columns:
        ic_col = "period_1"
    elif len(ic_stats.columns) > 0:
        ic_col = ic_stats.columns[0]
    else:
        ic_col = None

    if ic_col:
        summary["ic"] = {
            "IC均值": float(ic_stats[ic_col].get("IC均值", np.nan)),
            "ICIR": float(ic_stats[ic_col].get("ICIR", np.nan)),
        }

    # 多头统计
    if "多头" in returns_stats.index:
        long_s = returns_stats.loc["多头"]
        summary["多头"] = {
            "年化收益": float(long_s.get("年化收益", np.nan)),
            "夏普比率": float(long_s.get("夏普比率", np.nan)),
            "最大回撤": float(long_s.get("最大回撤", np.nan)),
            "胜率": float(long_s.get("胜率", np.nan)),
        }

    # 多头多基准
    if long_multi_benchmark_stats is not None:
        summary["多头多基准"] = {}
        for bench_name in long_multi_benchmark_stats.index:
            row = long_multi_benchmark_stats.loc[bench_name]
            summary["多头多基准"][bench_name] = {
                "年化收益": float(row.get("年化收益", np.nan)),
                "夏普比率": float(row.get("夏普比率", np.nan)),
                "最大回撤": float(row.get("最大回撤", np.nan)),
            }

    # 换手率 (取编号最大的分组, 即多头组; 兼容5组/10组等任意分组数)
    top_q_name, top_q_num = None, -1
    for q_name in turnover_stats.index:
        q_str = str(q_name)
        if q_str.startswith("Q") and q_str[1:].isdigit():
            q_num = int(q_str[1:])
            if q_num > top_q_num:
                top_q_name, top_q_num = q_name, q_num
    if top_q_name is not None:
        t = turnover_stats.loc[top_q_name]
        summary["换手率"] = {
            "平均双边换手率": float(t.get("平均换手率", np.nan)),
        }

    # 分年度
    if yearly_stats is not None and not yearly_stats.empty:
        summary["分年度"] = yearly_stats

    return summary
