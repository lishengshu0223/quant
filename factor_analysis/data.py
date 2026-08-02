"""
因子分析模组 - 数据获取与准备

对接 local_api (基于米筐本地化改造) 获取价格数据和股票池。
优先使用本地数据，本地无数据时回退到在线流量。
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Union, List, Optional

# 确保能导入 local_api
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from local_api import get_price, get_trading_dates, index_weights
from local_api.config import INDEX_FULL_CODES


# 指数代码到本地代码的映射
_INDEX_POOL_MAP = {
    "csi300": "000300.XSHG",
    "csi500": "000905.XSHG",
    "csi1000": "000852.XSHG",
    "csi2000": "932000.CSI",
    "sse50": "000016.XSHG",
    "hs300": "000300.XSHG",
    "zz500": "000905.XSHG",
    "zz1000": "000852.XSHG",
}


def _normalize_to_full_code(code: str) -> str:
    """将短代码转换为完整代码 (带交易所后缀)"""
    if not isinstance(code, str):
        return code
    # 已经是完整格式 (XXXXXX.XSHG/XSHE)
    if "." in code:
        return code
    # QuantaAlpha 格式: SH600000 / SZ000001 -> 600000.XSHG / 000001.XSHE
    if code.startswith(("SH", "SZ")) and len(code) == 8:
        pure_code = code[2:]
        exchange = "XSHG" if code.startswith("SH") else "XSHE"
        return f"{pure_code}.{exchange}"
    # 指数代码 (如 000300)
    if code in INDEX_FULL_CODES:
        return INDEX_FULL_CODES[code]
    # 6 位纯数字代码
    if len(code) == 6:
        if code.startswith("6") or code.startswith("9") or code.startswith("7"):
            return f"{code}.XSHG"
        else:
            return f"{code}.XSHE"
    return code


def normalize_factor_codes(factor: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """
    标准化因子数据的股票代码格式为 local_api 格式 (XXXXXX.XSHG/XSHE)

    支持的输入格式:
        - SH600000 / SZ000001 (QuantaAlpha 格式)
        - 600000.XSHG / 000001.XSHE (local_api 格式)
        - 600000 / 000001 (纯数字)

    Parameters
    ----------
    factor : pd.Series or pd.DataFrame
        因子数据，MultiIndex (date, code)

    Returns
    -------
    pd.Series or pd.DataFrame
        代码标准化后的因子数据
    """
    result = factor.copy()
    codes = result.index.get_level_values(1).astype(str)
    normalized_codes = [_normalize_to_full_code(c) for c in codes]
    # 重建 MultiIndex
    dates = result.index.get_level_values(0)
    result.index = pd.MultiIndex.from_arrays(
        [dates, normalized_codes], names=result.index.names
    )
    return result


def get_stock_pool(
    stock_pool: Union[str, List[str]],
    date=None,
) -> List[str]:
    """
    获取股票池代码列表

    Parameters
    ----------
    stock_pool : str or list
        股票池标识。支持:
        - "all": 全部 A 股
        - "csi300"/"hs300": 沪深 300
        - "csi500"/"zz500": 中证 500
        - "csi1000"/"zz1000": 中证 1000
        - "sse50": 上证 50
        - 具体股票代码列表
    date : str or pd.Timestamp, optional
        指数成分股日期，None 则使用最新

    Returns
    -------
    list[str]
        完整股票代码列表 (带 .XSHG/.XSHE 后缀)
    """
    if stock_pool == "all":
        from local_api import get_stock_codes
        return get_stock_codes()

    if isinstance(stock_pool, str):
        stock_pool = [stock_pool]

    codes = set()
    for sp in stock_pool:
        if sp.lower() in _INDEX_POOL_MAP:
            index_code = _INDEX_POOL_MAP[sp.lower()]
            weights = index_weights(index_code, date=date)
            if not weights.empty:
                # weights index 是短代码
                for code in weights.index:
                    codes.add(_normalize_to_full_code(str(code)))
        else:
            codes.add(_normalize_to_full_code(sp))

    return sorted(list(codes))


def get_trading_dates_range(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[pd.Timestamp]:
    """
    获取交易日历

    Parameters
    ----------
    start_date : str, optional
        开始日期
    end_date : str, optional
        结束日期

    Returns
    -------
    list[pd.Timestamp]
        交易日列表
    """
    return get_trading_dates(start_date=start_date, end_date=end_date)


def get_factor_prices(
    factor: Union[pd.Series, pd.DataFrame],
    stock_pool: Optional[Union[str, List[str]]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    fields: Optional[List[str]] = None,
    max_forward_period: int = 20,
) -> pd.DataFrame:
    """
    根据因子数据获取对应的价格数据

    会自动扩展 end_date 以容纳最长的远期收益计算周期。

    Parameters
    ----------
    factor : pd.Series or pd.DataFrame
        因子数据，MultiIndex (date, code)
    stock_pool : str or list, optional
        股票池。None 则使用因子中出现的所有股票
    start_date : str, optional
        开始日期。None 则使用因子数据的最早日期
    end_date : str, optional
        结束日期。None 则自动扩展以容纳远期收益计算
    fields : list, optional
        需要的价格字段，默认 ["close"]
    max_forward_period : int, default 20
        最长远期收益周期 (交易日)，用于自动扩展 end_date

    Returns
    -------
    pd.DataFrame
        价格数据，MultiIndex (date, code)，后复权
    """
    # 从因子数据推断股票池和日期范围
    if isinstance(factor, pd.DataFrame):
        factor_idx = factor.index
    else:
        factor_idx = factor.index

    factor_dates = pd.DatetimeIndex(factor_idx.get_level_values(0)).unique().sort_values()
    factor_codes = factor_idx.get_level_values(1).unique()

    if start_date is None:
        start_date = factor_dates[0].strftime("%Y-%m-%d")

    if end_date is None:
        # 扩展约 max_forward_period 个交易日 (约 1.5 倍自然日)
        extended_end = factor_dates[-1] + pd.Timedelta(days=int(max_forward_period * 1.5))
        end_date = extended_end.strftime("%Y-%m-%d")

    if stock_pool is None:
        stock_pool = [str(c) for c in factor_codes]
    elif isinstance(stock_pool, str) and stock_pool.lower() in _INDEX_POOL_MAP:
        stock_pool = get_stock_pool(stock_pool)
    elif isinstance(stock_pool, str):
        stock_pool = [stock_pool]

    if fields is None:
        fields = ["close"]

    # 使用 local_api 获取后复权价格数据
    prices = get_price(
        order_book_ids=stock_pool,
        start_date=start_date,
        end_date=end_date,
        frequency="1d",
        fields=fields,
        adjust_type="post",  # 后复权，保证收益计算连续性
        expect_df=True,
    )

    # 对停牌日进行前向填充, 模拟 rqdatac 行为
    # rqdatac 在停牌日返回 (close=前收盘价, total_turnover=0),
    # 本地数据在停牌日不返回记录, 需要填充以保证 IC/收益计算与 rqdatac 一致
    if not prices.empty and "close" in prices.columns:
        prices = _fill_suspended_prices(prices, factor_dates)

    return prices


def _fill_suspended_prices(
    prices: pd.DataFrame,
    factor_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    对停牌日进行前向填充, 模拟 rqdatac 的行为

    rqdatac 在停牌日返回 close=前收盘价, 本地数据停牌日无记录。
    对每只股票的 close 列做 ffill, 保证停牌日也有价格用于远期收益计算。

    Parameters
    ----------
    prices : pd.DataFrame
        价格数据, MultiIndex (date, code), 至少含 close 列
    factor_dates : pd.DatetimeIndex
        因子数据的交易日列表 (未使用, 保留接口兼容)

    Returns
    -------
    pd.DataFrame
        填充后的价格数据, 停牌日 close=前收盘价, 其他列保持 NaN
    """
    # 转为宽格式 (date × code), 对每只股票做 ffill
    close_wide = prices["close"].unstack("code").sort_index()
    close_wide = close_wide.ffill()

    # stack 回长格式 (包含新增的停牌日记录)
    close_filled = close_wide.stack()
    close_filled.index.names = ["date", "code"]
    close_filled.name = "close"

    # 用填充后的 close 替换原 DataFrame
    # 停牌日的新记录: close=前收盘价, 其他列=NaN
    result = prices.drop(columns=["close"]).join(close_filled.to_frame(), how="outer")
    result = result.sort_index()
    return result


def get_factor_data_from_h5(
    h5_path: str,
    factor_name: Optional[str] = None,
) -> pd.Series:
    """
    从 QuantaAlpha 生成的 result.h5 文件加载因子数据

    Parameters
    ----------
    h5_path : str
        h5 文件路径
    factor_name : str, optional
        因子列名。None 则取第一列

    Returns
    -------
    pd.Series
        因子数据，MultiIndex (datetime, instrument)
    """
    df = pd.read_hdf(h5_path, key="data")
    if isinstance(df, pd.DataFrame):
        if factor_name is None:
            factor_name = df.columns[0]
        return df[factor_name]
    return df


# =============================================================================
# Benchmark 基准收益
# =============================================================================

# 默认基准: 沪深 300
_DEFAULT_BENCHMARK = "000300.XSHG"


def get_benchmark_returns(
    benchmark: str = _DEFAULT_BENCHMARK,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    periods: List[int] = (1, 5, 10, 20),
) -> pd.DataFrame:
    """
    获取 benchmark 指数的多周期远期收益

    用于计算分组超额收益。

    Parameters
    ----------
    benchmark : str, default "000300.XSHG"
        基准指数代码。常见值:
        - "000300.XSHG": 沪深 300 (默认)
        - "000905.XSHG": 中证 500
        - "000852.XSHG": 中证 1000
        - "000016.XSHG": 上证 50
        也可传短代码 "000300" 等
    start_date : str, optional
        开始日期
    end_date : str, optional
        结束日期。None 则自动扩展以容纳远期收益
    periods : list[int], default (1, 5, 10, 20)
        远期收益周期 (交易日)

    Returns
    -------
    pd.DataFrame
        基准远期收益，index=date, columns=["period_1", "period_5", ...]
    """
    from local_api import get_index_price

    # 标准化 benchmark 代码
    benchmark = _normalize_to_full_code(benchmark)

    # 自动扩展 end_date 以容纳远期收益
    if end_date is not None and periods:
        extended_end = pd.Timestamp(end_date) + pd.Timedelta(
            days=int(max(periods) * 1.5)
        )
        end_date = extended_end.strftime("%Y-%m-%d")

    # 获取指数价格
    prices = get_index_price(
        order_book_ids=[benchmark],
        start_date=start_date,
        end_date=end_date,
        frequency="1d",
        fields=["close"],
    )

    if prices.empty:
        raise ValueError(f"无法获取基准指数 {benchmark} 的价格数据")

    # 转为宽格式 (index=date, columns=code)
    close_wide = prices["close"].unstack("code").sort_index()

    # 计算多周期远期收益
    forward_returns = {}
    for period in periods:
        future_price = close_wide.shift(-period)
        fwd_ret = future_price / close_wide - 1.0
        # 只取该指数的一列
        if isinstance(fwd_ret, pd.DataFrame):
            fwd_ret = fwd_ret.iloc[:, 0]
        forward_returns[f"period_{period}"] = fwd_ret

    result = pd.DataFrame(forward_returns)
    result.index.name = "date"
    return result


def get_market_mean_returns(
    clean_data: pd.DataFrame,
    periods: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    计算全市场等权平均远期收益 (作为默认基准)

    对每个交易日，计算所有股票的等权平均远期收益。

    Parameters
    ----------
    clean_data : pd.DataFrame
        get_clean_factor_and_forward_returns 返回的清洗数据
    periods : list[str], optional
        需要计算的周期列名。None 则自动检测所有 "period_*" 列

    Returns
    -------
    pd.DataFrame
        全市场等权均值收益，index=date, columns=["period_1", ...]
    """
    if periods is None:
        periods = [c for c in clean_data.columns if c.startswith("period_")]

    if not periods:
        raise ValueError("clean_data 中未找到 period_* 远期收益列")

    # 按 date 分组计算所有股票的等权平均远期收益
    market_mean = clean_data.groupby(level=0)[periods].mean()
    market_mean.index.name = "date"

    return market_mean
