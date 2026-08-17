"""
因子分析模组 - 完整分析报告 (Tear Sheet)

特性:
    - 以多头为核心 (A股多空意义有限)
    - 多头相对全市场和各宽基指数(300/500/1000)的超额
    - 单利累计收益和单利回撤
    - 分年度多头收益 + 双边换手率
    - IC 只保留 IC均值和 ICIR
"""

import pandas as pd
import numpy as np
from typing import Optional, List

from .performance import (
    calc_information_coefficient,
    calc_group_returns,
    calc_long_returns,
    calc_cumulative_returns,
    calc_group_turnover,
    calc_factor_autocorrelation,
    _period_days,
)
from .stats import (
    calc_ic_stats,
    format_ic_stats,
    calc_returns_stats,
    format_returns_stats,
    calc_turnover_stats,
    format_turnover_stats,
    calc_yearly_stats,
    format_yearly_stats,
    calc_long_stats_multi_benchmark,
    summary_statistics,
)


# 默认基准配置
_DEFAULT_BENCHMARKS = {
    "000300.XSHG": "沪深300",
    "000905.XSHG": "中证500",
    "000852.XSHG": "中证1000",
}


def create_information_tear_sheet(
    clean_data: pd.DataFrame,
    method: str = "spearman",
    periods_per_year: int = 252,
) -> dict:
    """
    IC 分析报告

    Returns
    -------
    dict
        - "ic": IC 时序
        - "ic_stats": IC 统计量 (只含 IC均值, ICIR)
        - "ic_stats_formatted": 格式化后的 IC 统计量 (百分号)
    """
    ic = calc_information_coefficient(clean_data, method=method)
    ic_stats = calc_ic_stats(ic, periods_per_year=periods_per_year)
    ic_stats_fmt = format_ic_stats(ic_stats)

    return {
        "ic": ic,
        "ic_stats": ic_stats,
        "ic_stats_formatted": ic_stats_fmt,
    }


def create_returns_tear_sheet(
    clean_data: pd.DataFrame,
    period: str = "period_1",
    periods_per_year: int = 252,
    benchmark_returns: Optional[pd.Series] = None,
    excess: bool = True,
    benchmark_dict: Optional[dict] = None,
    normalize: bool = False,
) -> dict:
    """
    分组收益分析报告 (以多头为核心)

    Parameters
    ----------
    benchmark_dict : dict, optional
        多基准字典 {名称: 收益时序}，用于多头多基准统计
        None 则自动获取沪深300/中证500/中证1000
    normalize : bool, default False
        是否将 n 日前向收益除以 n 转为日度等效收益。
        n 日前向收益在时间轴上重叠, 直接累加会放大 n 倍;
        归一化后任意周期的收益量纲统一, 可跨周期比较(观察信号衰减)。

    Returns
    -------
    dict
        - "group_returns": 分组收益时序 (超额或绝对)
        - "long_returns": 多头收益时序
        - "cumulative_returns": 分组累计净值 (单利)
        - "cumulative_long": 多头累计净值 (单利)
        - "returns_stats": 收益统计量
        - "returns_stats_formatted": 格式化收益统计量
        - "long_multi_benchmark": 多头多基准统计 (如有)
    """
    group_returns = calc_group_returns(
        clean_data, period=period,
        benchmark_returns=benchmark_returns, excess=excess,
        normalize=normalize,
    )
    long_returns = calc_long_returns(
        clean_data, period=period,
        benchmark_returns=benchmark_returns, excess=excess,
        normalize=normalize,
    )

    # 累计收益 (单利)
    cum_group = calc_cumulative_returns(group_returns, simple_interest=True)
    cum_long = calc_cumulative_returns(long_returns, simple_interest=True)

    # 统计量
    returns_stats = calc_returns_stats(
        group_returns,
        periods_per_year=periods_per_year,
        long_returns=long_returns,
    )
    returns_stats_fmt = format_returns_stats(returns_stats)

    result = {
        "group_returns": group_returns,
        "long_returns": long_returns,
        "cumulative_returns": cum_group,
        "cumulative_long": cum_long,
        "returns_stats": returns_stats,
        "returns_stats_formatted": returns_stats_fmt,
    }

    # 多头多基准统计
    if benchmark_dict is None and excess:
        # 自动构建基准字典
        benchmark_dict = {"全市场": None}
        try:
            from .data import get_benchmark_returns
            date_min = str(clean_data.index.get_level_values(0).min().date())
            date_max = str(clean_data.index.get_level_values(0).max().date())
            # 基准周期须与组合收益周期一致, 否则 normalize 时基准贡献被缩小
            bench_days = _period_days(period)
            for code, name in _DEFAULT_BENCHMARKS.items():
                try:
                    bench = get_benchmark_returns(
                        benchmark=code,
                        start_date=date_min,
                        end_date=date_max,
                        periods=[bench_days],
                    )
                    benchmark_dict[name] = bench[f"period_{bench_days}"]
                except Exception:
                    pass
        except Exception:
            pass

    if benchmark_dict:
        long_multi = calc_long_stats_multi_benchmark(
            clean_data,
            benchmark_dict=benchmark_dict,
            period=period,
            periods_per_year=periods_per_year,
            normalize=normalize,
        )
        result["long_multi_benchmark"] = long_multi
        result["long_multi_benchmark_formatted"] = format_returns_stats(long_multi)

    return result


def create_turnover_tear_sheet(
    clean_data: pd.DataFrame,
    group_returns: Optional[pd.DataFrame] = None,
    periods_per_year: int = 252,
    turnover_period: int = 1,
    normalize: bool = False,
) -> dict:
    """
    换手率分析报告 (含分年度统计)

    Parameters
    ----------
    group_returns : pd.DataFrame, optional
        分组收益时序 (用于分年度统计)。None 则自动计算
    turnover_period : int, default 1
        调仓间隔(交易日), 应与因子收益的前向周期一致:
        例如因子收益用5日前向收益, 则比较相隔5个交易日的两期分组
    normalize : bool, default False
        是否将 turnover_period 日换手率除以周期转为日度等效,
        与分组收益的 normalize 口径一致, 使任意调仓周期的换手率可比

    Returns
    -------
    dict
        - "turnover": 换手率时序 (双边)
        - "autocorr": 因子自相关系数
        - "turnover_stats": 换手率统计量
        - "turnover_stats_formatted": 格式化换手率统计量
        - "yearly_stats": 分年度统计 (多头收益+双边换手率)
        - "yearly_stats_formatted": 格式化分年度统计
    """
    turnover = calc_group_turnover(
        clean_data, double_sided=True, period=turnover_period, normalize=normalize,
    )
    autocorr = calc_factor_autocorrelation(clean_data, lag=1)
    turnover_stats = calc_turnover_stats(turnover, autocorr_data=autocorr)
    turnover_stats_fmt = format_turnover_stats(turnover_stats)

    result = {
        "turnover": turnover,
        "autocorr": autocorr,
        "turnover_stats": turnover_stats,
        "turnover_stats_formatted": turnover_stats_fmt,
    }

    # 分年度统计
    if group_returns is not None:
        yearly = calc_yearly_stats(
            group_returns, turnover,
            long_quantile=None,  # 取最高分组
            periods_per_year=periods_per_year,
        )
        result["yearly_stats"] = yearly
        result["yearly_stats_formatted"] = format_yearly_stats(yearly)

    return result


def create_summary_tear_sheet(
    clean_data: pd.DataFrame,
    method: str = "spearman",
    period: str = "period_1",
    periods_per_year: int = 252,
    benchmark_returns: Optional[pd.Series] = None,
    excess: bool = True,
    normalize: bool = False,
) -> dict:
    """摘要分析报告"""
    ic_result = create_information_tear_sheet(
        clean_data, method=method, periods_per_year=periods_per_year
    )
    returns_result = create_returns_tear_sheet(
        clean_data, period=period, periods_per_year=periods_per_year,
        benchmark_returns=benchmark_returns, excess=excess,
        normalize=normalize,
    )
    turnover_result = create_turnover_tear_sheet(
        clean_data,
        group_returns=returns_result["group_returns"],
        periods_per_year=periods_per_year,
    )

    summary = summary_statistics(
        ic_stats=ic_result["ic_stats"],
        returns_stats=returns_result["returns_stats"],
        turnover_stats=turnover_result["turnover_stats"],
        yearly_stats=turnover_result.get("yearly_stats"),
        long_multi_benchmark_stats=returns_result.get("long_multi_benchmark"),
    )

    return {
        "ic_stats": ic_result["ic_stats"],
        "ic_stats_formatted": ic_result["ic_stats_formatted"],
        "returns_stats": returns_result["returns_stats"],
        "returns_stats_formatted": returns_result["returns_stats_formatted"],
        "turnover_stats": turnover_result["turnover_stats"],
        "turnover_stats_formatted": turnover_result["turnover_stats_formatted"],
        "yearly_stats": turnover_result.get("yearly_stats"),
        "yearly_stats_formatted": turnover_result.get("yearly_stats_formatted"),
        "long_multi_benchmark": returns_result.get("long_multi_benchmark"),
        "long_multi_benchmark_formatted": returns_result.get("long_multi_benchmark_formatted"),
        "summary": summary,
    }


def create_full_tear_sheet(
    clean_data: pd.DataFrame,
    method: str = "spearman",
    period: str = "period_1",
    periods_per_year: int = 252,
    benchmark_returns: Optional[pd.Series] = None,
    excess: bool = True,
    verbose: bool = True,
    normalize: bool = False,
) -> dict:
    """
    完整因子分析报告 (以多头为核心)

    一次性计算所有指标: IC、分组收益(多头)、换手率、分年度统计。

    Parameters
    ----------
    clean_data : pd.DataFrame
        清洗数据
    method : str, default "spearman"
        IC 计算方法
    period : str, default "period_1"
        使用的远期收益列
    periods_per_year : int, default 252
    benchmark_returns : pd.Series, optional
        基准收益 (用于分组收益)。None 则全市场等权均值
    excess : bool, default True
        是否计算超额收益
    verbose : bool, default True
        是否打印关键统计量
    normalize : bool, default False
        是否将 n 日前向收益除以 n 转为日度等效收益。
        n 日前向收益在时间轴上重叠, 直接累加会放大 n 倍;
        归一化后任意周期的收益量纲统一, 可跨周期比较(观察信号衰减)。

    Returns
    -------
    dict
        完整分析结果
    """
    result = {}

    # IC 分析
    ic_result = create_information_tear_sheet(
        clean_data, method=method, periods_per_year=periods_per_year
    )
    result["ic"] = ic_result["ic"]
    result["ic_stats"] = ic_result["ic_stats"]
    result["ic_stats_formatted"] = ic_result["ic_stats_formatted"]

    # 收益分析 (含多头多基准)
    returns_result = create_returns_tear_sheet(
        clean_data, period=period, periods_per_year=periods_per_year,
        benchmark_returns=benchmark_returns, excess=excess,
        normalize=normalize,
    )
    result["group_returns"] = returns_result["group_returns"]
    result["long_returns"] = returns_result["long_returns"]
    result["cumulative_returns"] = returns_result["cumulative_returns"]
    result["cumulative_long"] = returns_result["cumulative_long"]
    result["returns_stats"] = returns_result["returns_stats"]
    result["returns_stats_formatted"] = returns_result["returns_stats_formatted"]
    if "long_multi_benchmark" in returns_result:
        result["long_multi_benchmark"] = returns_result["long_multi_benchmark"]
        result["long_multi_benchmark_formatted"] = returns_result["long_multi_benchmark_formatted"]

    # 换手率分析 (含分年度)
    # normalize=True 时, 调仓间隔与收益前向周期一致(如5日收益→5日调仓),
    # 且换手率÷周期转日度等效, 与收益口径统一
    turnover_result = create_turnover_tear_sheet(
        clean_data,
        group_returns=returns_result["group_returns"],
        periods_per_year=periods_per_year,
        turnover_period=_period_days(period) if normalize else 1,
        normalize=normalize,
    )
    result["turnover"] = turnover_result["turnover"]
    result["autocorr"] = turnover_result["autocorr"]
    result["turnover_stats"] = turnover_result["turnover_stats"]
    result["turnover_stats_formatted"] = turnover_result["turnover_stats_formatted"]
    if "yearly_stats" in turnover_result:
        result["yearly_stats"] = turnover_result["yearly_stats"]
        result["yearly_stats_formatted"] = turnover_result["yearly_stats_formatted"]

    # 汇总
    result["summary"] = summary_statistics(
        ic_stats=result["ic_stats"],
        returns_stats=result["returns_stats"],
        turnover_stats=result["turnover_stats"],
        yearly_stats=result.get("yearly_stats"),
        long_multi_benchmark_stats=result.get("long_multi_benchmark"),
    )

    if verbose:
        _print_summary(result)

    return result


def _print_summary(result: dict):
    """打印关键统计量"""
    print("=" * 80)
    print("因子分析汇总报告 (多头为核心, 单利模式)")
    print("=" * 80)

    # IC 统计
    print("\n【IC 统计量】(百分号)")
    print("-" * 80)
    ic_fmt = result.get("ic_stats_formatted")
    if ic_fmt is not None and not ic_fmt.empty:
        print(ic_fmt.to_string())

    # 分组收益统计
    print("\n【分组收益统计量】(超额收益, 单利回撤)")
    print("-" * 80)
    rs_fmt = result.get("returns_stats_formatted")
    if rs_fmt is not None and not rs_fmt.empty:
        print(rs_fmt.to_string())

    # 多头多基准
    print("\n【多头多基准统计】(超额收益)")
    print("-" * 80)
    lmb_fmt = result.get("long_multi_benchmark_formatted")
    if lmb_fmt is not None and not lmb_fmt.empty:
        print(lmb_fmt.to_string())

    # 换手率
    print("\n【换手率统计量】(双边)")
    print("-" * 80)
    ts_fmt = result.get("turnover_stats_formatted")
    if ts_fmt is not None and not ts_fmt.empty:
        print(ts_fmt.to_string())

    # 分年度
    print("\n【分年度统计】(分组年化超额收益% + 多头年化双边换手率)")
    print("-" * 80)
    ys_fmt = result.get("yearly_stats_formatted")
    if ys_fmt is not None and not ys_fmt.empty:
        print(ys_fmt.to_string())

    # 关键指标
    print("\n【关键指标】")
    print("-" * 80)
    summary = result.get("summary", {})

    if "ic" in summary:
        ic = summary["ic"]
        print(f"IC 均值 : {ic.get('IC均值', 'N/A')}")
        print(f"ICIR    : {ic.get('ICIR', 'N/A')}")

    if "多头" in summary:
        long_s = summary["多头"]
        print(f"\n多头年化收益 : {long_s.get('年化收益', 'N/A')}")
        print(f"多头夏普比率 : {long_s.get('夏普比率', 'N/A')}")
        print(f"多头最大回撤 : {long_s.get('最大回撤', 'N/A')}")

    if "多头多基准" in summary:
        print("\n多头 vs 各基准年化收益:")
        for bench, stats in summary["多头多基准"].items():
            print(f"  vs {bench}: {stats.get('年化收益', 'N/A')}")

    if "换手率" in summary:
        t = summary["换手率"]
        print(f"\n多头平均双边换手率: {t.get('平均双边换手率', 'N/A')}")

    print("\n" + "=" * 80)
