"""
因子分析模组 (Factor Analysis Module)

参考 Alphalens 设计，基于本地 local_api (米筐本地化改造) 实现因子分析功能。

核心功能:
    - 因子 IC (信息系数) 时序计算 (向量化)
    - 分组收益时序计算 (超额收益, 支持多基准)
    - 分组换手率时序计算 (双边, 向量化)
    - 多头收益统计 (相对全市场/沪深300/中证500/中证1000)
    - 分年度统计 (多头收益 + 双边换手率)
    - 单利累计收益和单利回撤

典型用法:
    >>> import factor_analysis as fa
    >>> clean_data = fa.get_clean_factor_and_forward_returns(
    ...     factor=factor, periods=[1,5,10,20], quantiles=5,
    ... )
    >>> results = fa.create_full_tear_sheet(clean_data)
"""

from .data import (
    get_factor_prices,
    get_stock_pool,
    get_trading_dates_range,
    get_benchmark_returns,
    get_market_mean_returns,
)
from .clean import get_clean_factor_and_forward_returns
from .performance import (
    calc_information_coefficient,
    calc_group_returns,
    calc_long_returns,
    calc_long_short_returns,
    calc_cumulative_returns,
    calc_group_turnover,
    calc_quantile_turnover,
    calc_factor_autocorrelation,
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
from .tears import (
    create_full_tear_sheet,
    create_summary_tear_sheet,
    create_information_tear_sheet,
    create_returns_tear_sheet,
    create_turnover_tear_sheet,
)

__all__ = [
    # 数据获取
    "get_factor_prices",
    "get_stock_pool",
    "get_trading_dates_range",
    "get_benchmark_returns",
    "get_market_mean_returns",
    # 数据清洗
    "get_clean_factor_and_forward_returns",
    # 性能计算
    "calc_information_coefficient",
    "calc_group_returns",
    "calc_long_returns",
    "calc_long_short_returns",
    "calc_cumulative_returns",
    "calc_group_turnover",
    "calc_quantile_turnover",
    "calc_factor_autocorrelation",
    # 统计量
    "calc_ic_stats",
    "format_ic_stats",
    "calc_returns_stats",
    "format_returns_stats",
    "calc_turnover_stats",
    "format_turnover_stats",
    "calc_yearly_stats",
    "format_yearly_stats",
    "calc_long_stats_multi_benchmark",
    "summary_statistics",
    # 完整分析
    "create_full_tear_sheet",
    "create_summary_tear_sheet",
    "create_information_tear_sheet",
    "create_returns_tear_sheet",
    "create_turnover_tear_sheet",
]

__version__ = "0.2.0"
