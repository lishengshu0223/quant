"""
本地化因子挖掘项目 - 分钟频率因子表达式引擎入口(日频输出)

分钟模式设计(对应需求: 基于分钟频率挖掘, 最终因子仍为日频):
- 数据是 股票|日期|分钟(1分钟K线) 三维; 最终因子必须降维成 股票|日期 二维宽表。
- 算子分三类(均按股票、按天 groupby 处理):
  ① 聚合算子(降维, 3D->2D): SUM/MEAN/STD/MAX/MIN/MEDIAN/SKEW/KURT/LAST/FIRST/COUNT/
     QUANTILE/CORR/TS_AUTOCORR/TS_ARGMAX/TS_ARGMIN/REGRESSION_SLOPE/REGRESSION_INTERCEPT
  ② 维持维度算子(3D->3D): SLICE(分时)/ MASK(掩码)
  ③ 日频算子(与日频引擎完全相同, 作用于聚合后的二维数据)
- 运算顺序: 先分时(SLICE) -> 再切割(MASK) -> 最后降维(聚合)。
- 执行方式: 分钟数据(全市场约51GB)无法整体载入内存, 按股票批次流式计算:
  每个聚合算子节点按批次遍历分钟数据 -> 得到该节点的全市场日频宽表(缓存),
  最后在缓存之上对整棵AST求值(支持全市场截面算子RANK/ZSCORE等)。

本文件是拆分后的**薄入口**: 对外只暴露 compute_factor_minute 与关键类型/常量,
实现按职责分布在 minute/ 子包:
- minute/minute_parser.py    解析/类型推断/校验 + 全部规格常量
- minute/minute_kernels.py   numba 稠密归约内核
- minute/minute_data.py      分钟数据加载 + 长表聚合 + MinuteExpr/MinuteFieldCache/MinuteMarketData
- minute/minute_sparse_eval.py  长表路径求值(_BatchRunner) + CachedEvaluator + 3D/2D 运算辅助
- minute/minute_dense_eval.py   稠密加速路径(DenseEvaluator) + 稠密入口
"""

import pandas as pd

from .expr_engine import ExprError, Evaluator
from .minute.minute_parser import (
    MINUTE_FIELDS, MINUTE_AGG_SPEC, MINUTE_KEEP_SPEC, MINUTE_CROSS_SPEC,
    MINUTE_ROLL_SPEC, MINUTE_MAX_ROLL_WINDOW, CROSS_SECTIONAL_OPS, MASK_OPS,
    MINUTE_MAX_LAG, MINUTE_MAX_BASE_FEATURES,
    _parse_time, canonical, minute_validate, minute_fields_used,
)
from .minute.minute_data import (
    EQ_MINUTE_DIR, MinuteExpr, MinuteFieldCache, MinuteMarketData, SlicedData,
    _get_memory_cache, _raw_field_name, _finalize,
)
from .minute.minute_sparse_eval import _BatchRunner, _subtree_has_cross, CachedEvaluator
from .minute.minute_dense_eval import DenseEvaluator, compute_factor_minute_dense
from .minute.minute_kernels import _HAS_NUMBA

__all__ = [
    "compute_factor_minute", "minute_validate", "minute_fields_used", "canonical",
    "MinuteExpr", "MinuteMarketData", "MinuteFieldCache", "SlicedData", "DenseEvaluator",
    "EQ_MINUTE_DIR", "MINUTE_FIELDS", "MINUTE_AGG_SPEC", "MINUTE_KEEP_SPEC",
    "MINUTE_CROSS_SPEC", "MINUTE_ROLL_SPEC", "MASK_OPS",
]


def compute_factor_minute(expr: str, data, cfg) -> pd.DataFrame:
    """
    分钟模式入口: 校验并计算因子, 返回日频宽表(行=日期, 列=股票代码)。
    """
    ast, types, agg_nodes, agg_ids = minute_validate(expr, cfg)

    if not agg_nodes:
        # 纯日频公式(分钟模式下的普通日频因子): 直接按日频求值
        result = Evaluator(data).eval(ast)
        if not isinstance(result, pd.DataFrame):
            raise ExprError("公式计算结果为标量, 不是有效的截面因子")
        return _finalize(result)

    fields = minute_fields_used(ast, types)
    mmd = MinuteMarketData(cfg, data, fields)
    # 供批内求值识别嵌套聚合节点(阈值中的分钟聚合等)
    types["__agg_ids__"] = agg_ids

    # 稠密加速路径: 公式所需分钟字段均已常驻内存(稠密矩阵)时, 直接在[日×240×股]上numpy归约,
    # 替代长表groupby(实测SKEW等算子耗时从20分钟级降到秒级)。否则回退长表路径。
    needed = {_raw_field_name(f) for f in fields if f != "return"}
    if "return" in fields:
        needed.add("close")
    cache = _get_memory_cache(mmd)
    if cache is not None and needed <= cache.fields:
        return compute_factor_minute_dense(ast, types, agg_nodes, agg_ids, mmd, cache, data, cfg)

    # 公式含分钟级截面算子(RANK/ZSCORE/SCALE/CS_*)时, 需全市场分钟数据在场 -> 全市场分块模式
    needs_full_market = any(_subtree_has_cross(n) for n in agg_nodes)
    if needs_full_market:
        cache = mmd.compute_all_aggs_full_market(agg_nodes, types, data, cfg)
    else:
        # 单趟遍历全市场批次, 一次算出所有(最大)聚合节点
        cache = mmd.compute_all_aggs(agg_nodes, types, data, cfg)
    node_cache = {id(node): cache[canonical(node)] for node in agg_nodes}

    result = CachedEvaluator(data, node_cache).eval(ast)
    if not isinstance(result, pd.DataFrame):
        raise ExprError("公式计算结果为标量, 不是有效的截面因子")
    return _finalize(result)
