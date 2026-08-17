"""
本地化因子挖掘项目 - 分钟稠密加速路径求值（minute_engine 拆分模块⑤）

当公式所需分钟字段均已常驻内存(稠密矩阵)时, 在 [日×240×股] 上直接 numpy/numba 归约,
替代长表 groupby(实测 SKEW 等算子耗时从 20 分钟级降到秒级)。

职责:
- DenseEvaluator: 分钟表达式在稠密矩阵上的求值器(含截面/日内滚动/聚合)
- compute_factor_minute_dense: 稠密加速入口(无截面按股票分批 / 含截面按日期分块)

依赖: minute_kernels(numba) -> minute_data -> minute_parser -> expr_engine; minute_sparse_eval(_wide_binop/_subtree_has_cross/CachedEvaluator)。
"""

import time

import numpy as np
import pandas as pd

from .. import console
from ..expr_engine import (
    ExprError, eval_call_daily, Num, Str, Var, Call, Bin, Unary, Ternary,
)
from .minute_parser import MINUTE_CROSS_SPEC, MINUTE_ROLL_SPEC, _parse_time, canonical
from .minute_data import _raw_field_name, _get_memory_cache, _finalize
from .minute_kernels import (
    _HAS_NUMBA, _nb_skew, _nb_kurt, _nb_std, _nb_regression, _nb_corr,
    _nb_autocorr, _nb_last,
)
from .minute_sparse_eval import _wide_binop, _subtree_has_cross, CachedEvaluator


def _dense_binop(op, a, b):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if op == "+":
            return (a + b).astype(np.float32)
        if op == "-":
            return (a - b).astype(np.float32)
        if op == "*":
            return (a * b).astype(np.float32)
        if op == "/":
            out = a / b
            return np.where(np.isfinite(out), out, np.nan).astype(np.float32)
        if op == ">":
            return a > b
        if op == "<":
            return a < b
        if op == ">=":
            return a >= b
        if op == "<=":
            return a <= b
        if op == "==":
            return a == b
        if op == "!=":
            return a != b
        raise ExprError(f"未支持的分钟运算符 {op}")


class DenseEvaluator:
    """稠密路径: 分钟表达式在 [日×240×股] float32 稠密矩阵上求值, 聚合沿分钟轴 numpy 归约。
    self.sl: 当前股票窗口索引(用于控制中间内存); None=全部股票(截面算子需要)。"""

    def __init__(self, mmd, cache, data, cfg, types, agg_ids):
        self.mmd = mmd
        self.cache = cache
        self.data = data
        self.cfg = cfg
        self.types = types
        self.agg_ids = agg_ids
        self.daily_index = mmd.daily_index
        self.all_codes = list(data.close.columns)
        self.codes = cache.codes          # 有分钟文件的股票(与稠密数组列对齐)
        self.n_stocks = len(self.codes)
        self.n_days = len(cache.day_ints)
        self.t_grid = cache.t_grid
        self.sl = None                    # 当前股票窗口索引; None=全部股票(截面算子需要)
        self.day_slice = slice(None)      # 当前日期窗口(截面模式按日期分块控制中间内存)

    # ---- 基础字段 ----
    def _codes(self):
        return self.codes if self.sl is None else [self.codes[i] for i in self.sl]

    def field_dense(self, name):
        if name == "$minute":
            # 零内存广播视图: 不按成交掩码置NaN。与价格字段配对(如回归斜率对分钟时间)时
            # 由对方字段的valid掩码过滤, 结果与长表完全一致; 单独聚合$minute(如MEAN($minute))
            # 会因恒有值而成为常数因子, 无实际意义。
            n_days = self.n_days if self.day_slice == slice(None) else (
                self.day_slice.stop - self.day_slice.start)
            n_win = self.n_stocks if self.sl is None else len(self.sl)
            return np.broadcast_to(self.t_grid.astype(np.float32)[None, :, None],
                                   (n_days, 240, n_win))
        if name == "$return":
            v = self._dense_return()[self.day_slice]
            return v if self.sl is None else v[:, :, self.sl]
        raw = _raw_field_name(name[1:])
        arr = self.cache.arrays[raw][self.day_slice]      # [days,240,stocks]
        return arr if self.sl is None else arr[:, :, self.sl]

    def _dense_return(self):
        """返回缓存内置的 $return 数组(_build 时用原始 float64 close 逐股票构建,
        与长表 groupby(code,date) pct_change 数值一致)。仅在字段未常驻 close 却用到
        $return 时(理论上不会, 稠密路径恒需 close)由 _build_return_full 兜底重建。"""
        if self.cache.ret_arr is None:
            self.cache.ret_arr = self._build_return_full()
        return self.cache.ret_arr

    def _build_return_full(self):
        """构建全市场 $return: close[t]/上一根有成交close - 1(与长表pct_change一致, 跨停牌分钟跳号)。
        分天块计算, 峰值临时内存约5GB(避免整表float64中间量导致MemoryError)。
        注意: 前缀最大(accumulate)是"含自身"的最后有效位置, 必须先错位再累积才能得到
        "严格早于当前分钟"的上一个有效位置(否则每根有效分钟都会把自己判为当日首根而全部置NaN)。"""
        close = self.cache.arrays["close"]
        n, m, s = close.shape
        ret = np.full((n, m, s), np.nan, dtype=np.float32)
        chunk = 256
        t_idx = np.arange(m)[None, :, None]
        for i in range(0, n, chunk):
            c = close[i:i + chunk].astype(np.float64)   # float64中间量, 减小除法舍入
            mask = np.isfinite(c)
            # 第1趟: 前缀最大 -> prev_idx[t] = 到t为止(含自身)最后一个有效分钟位置; 无有效则为-1
            prev_idx = np.where(mask, t_idx, -1).astype(np.int32)
            np.maximum.accumulate(prev_idx, axis=1, out=prev_idx)
            # 第2趟: 错位(prev_idx[t] = 第1趟的 t-1 处)后再前缀最大
            #        -> 严格早于t的最后一个有效位置; 无则-1
            shifted = np.empty_like(prev_idx)
            shifted[:, 0, :] = -1
            shifted[:, 1:, :] = prev_idx[:, :-1, :]
            np.maximum.accumulate(shifted, axis=1, out=shifted)
            first = mask & (shifted < 0)                    # 当日首根有效分钟(无更早有效)
            prev = np.take_along_axis(c, np.maximum(shifted, 0), axis=1)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                r = (c - prev) / prev
                r = np.where(mask & ~first & np.isfinite(r), r, np.nan).astype(np.float32)
            ret[i:i + chunk] = r
            del mask, prev_idx, shifted, first, prev, r
        return ret

    def _shift_axis1_rows(self, v, lag):
        """按(日,股)内'有效分钟行号'滞后, 与长表 groupby.shift(lag) 一致:
        跨停牌分钟跳号(按成交顺序数lag根), 而不是按240根网格错位。"""
        mask = np.isfinite(v)
        d, m, s = v.shape
        cum = np.cumsum(mask, axis=1, dtype=np.int32)           # 当日有效分钟行号(1..k)
        # 反向索引: rev[d, rank, s] = 该有效分钟在轴1上的位置
        rev = np.full((d, m + 1, s), -1, dtype=np.int32)
        d_idx, t_idx, s_idx = np.nonzero(mask)
        rev[d_idx, cum[d_idx, t_idx, s_idx], s_idx] = t_idx
        src = cum - lag                                         # 目标有效行号
        ok = mask & (src >= 1)
        src_c = np.clip(src, 0, m)
        dd = np.arange(d)[:, None, None]
        ss = np.arange(s)[None, None, :]
        gathered = v[dd, rev[dd, src_c, ss], ss]
        return np.where(ok, gathered, np.nan).astype(np.float32)

    # ---- 求值 ----
    def _as_dense(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy(dtype=np.float32)
        if isinstance(x, np.ndarray):
            if x.ndim == 3:
                return x
            if x.ndim == 2:
                return x[:, None, :]      # 宽表 [days,stocks] -> [days,1,stocks] 广播
        return np.asarray(x)

    def eval(self, node):
        if id(node) in self.agg_ids:
            evals = [self.eval_arg(a) for a in node.args]
            return self.aggregate(node.name, evals)
        if self.types[id(node)] == "M":
            return self.eval_m(node)
        return self.eval_w(node)

    def eval_arg(self, node):
        """与长表 _BatchRunner.eval_arg 等价的分发: 分钟/聚合/标量/日频"""
        t = self.types.get(id(node))
        if t == "M":
            return self.eval_m(node)
        if id(node) in self.agg_ids:       # 嵌套聚合(如 MASK 阈值、MEAN(STD(...)))
            evals = [self.eval_arg(a) for a in node.args]
            return self.aggregate(node.name, evals)
        if isinstance(node, (Num, Str)):
            return node.value
        return self.eval_w(node)

    def eval_m(self, node):
        if isinstance(node, Num):
            return node.value
        if isinstance(node, Var):
            return self.field_dense(node.name)
        if isinstance(node, Unary):
            x = self._as_dense(self.eval_m(node.x))
            return (-x).astype(np.float32) if node.op == "-" else x
        if isinstance(node, Bin):
            l = self._as_dense(self.eval(node.left))
            r = self._as_dense(self.eval(node.right))
            return _dense_binop(node.op, l, r)
        if isinstance(node, Ternary):
            c = self._as_dense(self.eval(node.cond)).astype(bool)
            t = self._as_dense(self.eval(node.true))
            f = self._as_dense(self.eval(node.false))
            return np.where(c, t, f).astype(np.float32)
        if isinstance(node, Call):
            name = node.name
            if name in MINUTE_CROSS_SPEC:
                if self.sl is not None:
                    raise ExprError("分钟截面算子必须在全市场口径下计算(稠密路径内部错误)")
                return self.cs_transform(name, self.eval_m(node.args[0]))
            if name in MINUTE_ROLL_SPEC:
                x = self.eval_m(node.args[0])
                n = int(node.args[1].value) if len(node.args) > 1 else 5
                return self.intraday_rolling(x, name, n)
            if name == "SLICE":
                x = self.eval_m(node.args[0])
                s = _parse_time(node.args[1].value)
                e = _parse_time(node.args[2].value)
                m = (self.t_grid >= s) & (self.t_grid <= e)
                return np.where(m[None, :, None], x, np.nan).astype(np.float32)
            if name == "MASK":
                x = self.eval_m(node.args[0])
                op = node.args[1].value
                thr = self._as_dense(self.eval(node.args[2]))
                cond = _dense_cmp(op, x, thr)
                return np.where(cond, x, np.nan).astype(np.float32)
            raise ExprError(f"分钟上下文中不允许的函数 {name}")
        raise ExprError(f"无法求值: {type(node).__name__}")

    def eval_w(self, node):
        """日频(宽表)子表达式求值, 返回 [days,当前codes] 宽表或标量;
        截面分块模式(day_slice)下对宽表行切片, 与分钟块的行数对齐"""
        res = self._eval_w_inner(node)
        ds = self.day_slice
        if ds != slice(None) and isinstance(res, pd.DataFrame):
            res = res.iloc[ds]
        return res

    def _eval_w_inner(self, node):
        if isinstance(node, Num):
            return node.value
        if isinstance(node, Str):
            return node.value
        if isinstance(node, Var):
            return self.data.var(node.name).reindex(columns=self._codes())
        if isinstance(node, Unary):
            x = self.eval_w(node.x)
            return -x if node.op == "-" else x
        if isinstance(node, Bin):
            return _wide_binop(node.op, self.eval_w(node.left), self.eval_w(node.right))
        if isinstance(node, Ternary):
            c = self.eval_w(node.cond)
            t = self.eval_w(node.true)
            f = self.eval_w(node.false)
            c = c.astype(bool) if isinstance(c, np.ndarray) else bool(c)
            return np.where(c, t, f)
        if isinstance(node, Call):
            args = [self.eval_w(a) for a in node.args]
            return eval_call_daily(node.name, args)
        raise ExprError(f"无法求值日频子表达式: {type(node).__name__}")

    # ---- 分钟截面(3D->3D, axis=2, 需全市场) ----
    def cs_transform(self, name, x):
        v = x
        valid = np.isfinite(v)
        n_valid = valid.sum(axis=2, keepdims=True).astype(np.float64)   # [days,240,1]
        if name == "RANK":
            # 与长表 groupby(["date","t"])["v"].rank(pct=True) 一致:
            # 仅在有效值之间排名(并列取平均位置), pct = (平均位置0based+1)/n_valid。
            # 注意: 不能用"总股票数S_算双向排名"——稀疏截面(大量停牌)时会引入无效位置的偏移。
            v64 = np.asarray(x, dtype=np.float64)
            valid = np.isfinite(v64)
            n_valid = valid.sum(axis=2, keepdims=True).astype(np.float64)
            S_ = v64.shape[2]
            xm = np.where(valid, v64, np.inf)
            order = np.argsort(xm, axis=2)                       # 有效升序, inf(无效)最后
            sorted_v = np.take_along_axis(xm, order, axis=2)
            idx_grid = np.broadcast_to(np.arange(S_)[None, None, :], sorted_v.shape)
            # 并列分组: start_pos=组起始位置, end_pos=组结束位置(全排序0-based)
            is_start = np.ones_like(sorted_v, dtype=bool)
            is_start[:, :, 1:] = sorted_v[:, :, 1:] != sorted_v[:, :, :-1]
            start_pos = np.where(is_start, idx_grid, 0)
            np.maximum.accumulate(start_pos, axis=2, out=start_pos)
            is_end = np.ones_like(sorted_v, dtype=bool)
            is_end[:, :, :-1] = sorted_v[:, :, :-1] != sorted_v[:, :, 1:]
            end_pos = np.where(is_end, idx_grid, S_ - 1)
            rev = end_pos[:, :, ::-1]
            np.minimum.accumulate(rev, axis=2, out=rev)
            end_pos = rev[:, :, ::-1]
            avg_pos = (start_pos + end_pos) / 2.0
            pct = (avg_pos + 1.0) / np.maximum(n_valid, 1)
            # 关键: pct 是在排序空间(按 order 排列)计算的, 必须 scatter 回原始位置,
            # 否则输出的是"排序后的排名序列"而非"原始位置的排名"。
            # 注意 inf(无效值)组在排序空间也有 pct, scatter 后必须把无效位置重新置 NaN。
            out = np.full_like(v64, np.nan)
            np.put_along_axis(out, order, pct, axis=2)
            out[~valid] = np.nan
            return out.astype(np.float32)
        if name == "ZSCORE":
            mu = np.nansum(np.where(valid, v, 0.0), axis=2, keepdims=True) / np.maximum(n_valid, 1)
            d = np.where(valid, v - mu, 0.0)
            ss = np.nansum(d * d, axis=2, keepdims=True)
            sd = np.sqrt(ss / np.maximum(n_valid - 1, 1))   # 样本std(与pandas transform一致)
            with np.errstate(divide="ignore", invalid="ignore"):
                z = (v - mu) / np.where(sd > 0, sd, np.nan)
            return np.where(valid & np.isfinite(z), z, np.nan).astype(np.float32)
        if name == "SCALE":
            s = np.nansum(np.where(valid, np.abs(v), 0.0), axis=2, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                out = v / np.where(s > 0, s, np.nan)
            return np.where(valid & np.isfinite(out), out, np.nan).astype(np.float32)
        if name == "CS_MEAN":
            mu = np.nansum(np.where(valid, v, 0.0), axis=2, keepdims=True) / np.maximum(n_valid, 1)
            return np.where(valid, mu, np.nan).astype(np.float32)
        if name == "CS_STD":
            mu = np.nansum(np.where(valid, v, 0.0), axis=2, keepdims=True) / np.maximum(n_valid, 1)
            d = np.where(valid, v - mu, 0.0)
            ss = np.nansum(d * d, axis=2, keepdims=True)
            sd = np.sqrt(ss / np.maximum(n_valid - 1, 1))
            return np.where(valid, sd, np.nan).astype(np.float32)
        raise ExprError(f"分钟截面算子 {name} 未实现")

    # ---- 日内滚动(3D->3D, axis=1) ----
    def intraday_rolling(self, x, name, n):
        if n < 1:
            raise ExprError("日内滚动窗口必须 >= 1")
        xp = np.pad(x, ((0, 0), (n - 1, 0), (0, 0)), mode="constant", constant_values=np.nan)
        # 注意: sliding_window_view 对单个 int axis 把窗口维度追加到末尾 -> [days,240,stocks,n]
        win = np.lib.stride_tricks.sliding_window_view(xp, n, axis=1)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if name == "INTRADAY_SUM":
                cnt = np.sum(~np.isnan(win), axis=-1)
                out = np.nansum(win, axis=-1)
                out = np.where(cnt > 0, out, np.nan)
            elif name == "INTRADAY_MEAN":
                out = np.nanmean(win, axis=-1)
            elif name == "INTRADAY_STD":
                out = np.nanstd(win, axis=-1, ddof=1)
            elif name == "INTRADAY_MAX":
                out = np.nanmax(win, axis=-1)
            elif name == "INTRADAY_MIN":
                out = np.nanmin(win, axis=-1)
            elif name == "INTRADAY_MEDIAN":
                out = np.nanmedian(win, axis=-1)
            else:
                raise ExprError(f"日内滚动算子 {name} 未实现")
        return out.astype(np.float32)

    # ---- 聚合(3D->2D, axis=1) ----
    def aggregate(self, name, evals):
        v = evals[0]
        if not isinstance(v, np.ndarray) or v.ndim != 3:
            raise ExprError(f"聚合算子 {name} 的分钟参数类型异常")
        v = np.asarray(v, dtype=np.float32)
        valid = ~np.isnan(v)
        n = valid.sum(axis=1).astype(np.float64)          # [days,stocks]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if name == "COUNT":
                # 与长表一致: 全日无成交(无K线行)的股票->NaN
                out = np.where(n > 0, n, np.nan)
            elif name == "SUM":
                s = np.nansum(np.where(valid, v, np.float32(0)), axis=1, dtype=np.float64)
                out = np.where(n > 0, s, np.nan)
            elif name == "MEAN":
                s = np.nansum(np.where(valid, v, np.float32(0)), axis=1, dtype=np.float64)
                out = np.where(n > 0, s / np.maximum(n, 1), np.nan)
            elif name == "STD":
                if _HAS_NUMBA:
                    out = _nb_std(v)
                else:
                    mu = np.nansum(np.where(valid, v, np.float32(0)), axis=1, dtype=np.float64) / np.maximum(n, 1)
                    # 均值中心化在float32上做(与源数据精度一致), 累加用float64, 避免整表float64临时数组
                    d = np.where(valid, v - mu[:, None, :].astype(np.float32), np.float32(0)).astype(np.float32)
                    ss = np.nansum(d * d, axis=1, dtype=np.float64)
                    out = np.where(n >= 2, np.sqrt(ss / np.maximum(n - 1, 1)), np.nan)
            elif name == "MAX":
                out = np.nanmax(v, axis=1)
                out = np.where(n > 0, out, np.nan)
            elif name == "MIN":
                out = np.nanmin(v, axis=1)
                out = np.where(n > 0, out, np.nan)
            elif name == "MEDIAN":
                out = np.nanmedian(v, axis=1)
            elif name == "SKEW":
                if _HAS_NUMBA:
                    out = _nb_skew(v)
                else:
                    mu = np.nansum(np.where(valid, v, np.float32(0)), axis=1, dtype=np.float64) / np.maximum(n, 1)
                    d = np.where(valid, v - mu[:, None, :].astype(np.float32), np.float32(0)).astype(np.float32)
                    d2 = np.nansum(d * d, axis=1, dtype=np.float64)
                    d3 = np.nansum(d * d * d, axis=1, dtype=np.float64)
                    # 与 pandas .skew()(修正Fisher-Pearson)一致
                    skew = n * np.sqrt(n - 1) * d3 / ((n - 2) * np.power(d2, 1.5))
                    out = np.where((n >= 3) & (d2 > 0), skew, np.nan)
            elif name == "KURT":
                if _HAS_NUMBA:
                    out = _nb_kurt(v)
                else:
                    mu = np.nansum(np.where(valid, v, np.float32(0)), axis=1, dtype=np.float64) / np.maximum(n, 1)
                    d = np.where(valid, v - mu[:, None, :].astype(np.float32), np.float32(0)).astype(np.float32)
                    d2 = np.nansum(d * d, axis=1, dtype=np.float64)
                    d4 = np.nansum(d * d * d * d, axis=1, dtype=np.float64)
                    mu2 = d2 / np.maximum(n, 1)
                    mu4 = d4 / np.maximum(n, 1)
                    kurt = (n - 1.0) / ((n - 2.0) * (n - 3.0)) * ((n + 1.0) * (mu4 / mu2 ** 2.0) - 3.0 * (n - 1.0))
                    out = np.where((n >= 4) & (mu2 > 0), kurt, np.nan)
            elif name == "QUANTILE":
                out = np.nanquantile(v, float(evals[1]), axis=1)
            elif name in ("LAST", "FIRST"):
                # 与长表一致: 两者都取"当日最后一个有值分钟的值"
                if _HAS_NUMBA:
                    out = _nb_last(v)
                else:
                    last_pos = v.shape[1] - 1 - np.argmax(valid[:, ::-1, :], axis=1)
                    out = v[np.arange(v.shape[0])[:, None], np.maximum(last_pos, 0),
                            np.arange(v.shape[2])[None, :]]
                    out = np.where(n > 0, out, np.nan)
            elif name == "TS_ARGMAX":
                pos = np.argmax(np.where(valid, v, -np.inf), axis=1).astype(np.float64)
                out = np.where(n >= 2, pos / np.maximum(n - 1, 1), np.nan)
            elif name == "TS_ARGMIN":
                pos = np.argmin(np.where(valid, v, np.inf), axis=1).astype(np.float64)
                out = np.where(n >= 2, pos / np.maximum(n - 1, 1), np.nan)
            elif name == "CORR":
                if _HAS_NUMBA:
                    out = _nb_corr(v, np.asarray(evals[1], dtype=np.float32))
                else:
                    out = self._corr_axis1(v, np.asarray(evals[1], dtype=np.float32))
            elif name == "TS_AUTOCORR":
                lag = int(evals[1])
                if _HAS_NUMBA:
                    out = _nb_autocorr(v, lag)
                else:
                    y = self._shift_axis1_rows(v, lag)
                    out = self._corr_axis1(v, y)
            elif name in ("REGRESSION_SLOPE", "REGRESSION_INTERCEPT"):
                if _HAS_NUMBA:
                    want = 0 if name == "REGRESSION_SLOPE" else 1
                    out = _nb_regression(v, np.asarray(evals[1], dtype=np.float32), want)
                else:
                    out = self._regression_axis1(v, np.asarray(evals[1], dtype=np.float32), name)
            else:
                raise ExprError(f"聚合算子 {name} 未实现(稠密路径)")
        df = pd.DataFrame(np.asarray(out, dtype=np.float32),
                          index=self.daily_index[self.day_slice],
                          columns=self._codes())
        if self.sl is None:
            df = df.reindex(index=self.daily_index[self.day_slice], columns=self.all_codes)
        return df

    def _corr_axis1(self, x, y):
        """皮尔逊相关(均值中心化两趟算法, float32算术+float64累加):
        常数序列方差精确为0 -> NaN, 与长表pandas行为一致; 峰值内存仅float32级"""
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        valid = np.isfinite(x) & np.isfinite(y)
        n = valid.sum(axis=1).astype(np.float64)

        def s(a):
            return np.nansum(np.where(valid, a, np.float32(0)), axis=1, dtype=np.float64)
        sx, sy = s(x), s(y)
        mu_x = (sx / np.maximum(n, 1)).astype(np.float32)
        mu_y = (sy / np.maximum(n, 1)).astype(np.float32)
        dx = np.where(valid, x - mu_x[:, None], np.float32(0)).astype(np.float32)
        dy = np.where(valid, y - mu_y[:, None], np.float32(0)).astype(np.float32)
        varx = np.nansum(dx * dx, axis=1, dtype=np.float64)
        vary = np.nansum(dy * dy, axis=1, dtype=np.float64)
        cov = np.nansum(dx * dy, axis=1, dtype=np.float64)
        corr = cov / np.sqrt(varx * vary)
        return np.where((n >= 2) & (varx > 0) & (vary > 0), corr, np.nan)

    def _regression_axis1(self, v, y, want):
        """一元回归(均值中心化两趟算法, float32算术+float64累加): y 为常数时方差为0 -> NaN"""
        v = np.asarray(v, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        valid = np.isfinite(v) & np.isfinite(y)
        n = valid.sum(axis=1).astype(np.float64)

        def s(a):
            return np.nansum(np.where(valid, a, np.float32(0)), axis=1, dtype=np.float64)
        sv, sy = s(v), s(y)
        mu_v = (sv / np.maximum(n, 1)).astype(np.float32)
        mu_y = (sy / np.maximum(n, 1)).astype(np.float32)
        dv = np.where(valid, v - mu_v[:, None], np.float32(0)).astype(np.float32)
        dy = np.where(valid, y - mu_y[:, None], np.float32(0)).astype(np.float32)
        vary = np.nansum(dy * dy, axis=1, dtype=np.float64)
        cov = np.nansum(dv * dy, axis=1, dtype=np.float64)
        slope = cov / vary
        intercept = mu_v - slope * mu_y
        res = slope if want == "REGRESSION_SLOPE" else intercept
        return np.where((n >= 2) & (vary > 0), res, np.nan)


def _dense_cmp(op, a, b):
    if op == ">":
        return a > b
    if op == "<":
        return a < b
    if op == ">=":
        return a >= b
    if op == "<=":
        return a <= b
    if op == "==":
        return a == b
    if op == "!=":
        return a != b
    raise ExprError(f"未支持的比较运算符 {op}")


def compute_factor_minute_dense(ast, types, agg_nodes, agg_ids, mmd, cache, data, cfg):
    """稠密加速入口: 分钟聚合在[日×240×股]上 numpy 归约。
    - 无截面算子: 按股票窗口分批(step=minute_dense_batch), 控制中间内存;
    - 含截面算子: 按日期分块(块天数=minute_dense_chunk_days), 每块加载全部股票使截面等价全市场。"""
    de = DenseEvaluator(mmd, cache, data, cfg, types, agg_ids)
    node_cache = {}
    t0 = time.time()
    for node in agg_nodes:
        if id(node) in node_cache:
            continue
        k0 = time.time()
        if _subtree_has_cross(node):
            rows = []
            chunk = int(getattr(cfg, "minute_dense_chunk_days", 200))
            for s in range(0, de.n_days, chunk):
                de.day_slice = slice(s, min(s + chunk, de.n_days))
                de.sl = None
                evals = [de.eval_arg(a) for a in node.args]
                rows.append(de.aggregate(node.name, evals))
                del evals
            de.day_slice = slice(None)
            de.sl = None
            node_cache[id(node)] = pd.concat(rows, axis=0).reindex(
                index=de.daily_index, columns=de.all_codes)
        else:
            parts = []
            step = int(getattr(cfg, "minute_dense_batch", 1000))
            for i in range(0, len(de.codes), step):
                de.sl = list(range(i, min(i + step, len(de.codes))))
                evals = [de.eval_arg(a) for a in node.args]
                parts.append(de.aggregate(node.name, evals))
                del evals
            de.sl = None
            node_cache[id(node)] = pd.concat(parts, axis=1).reindex(columns=de.all_codes)
        console.log(f"    [稠密聚合] {canonical(node)} 耗时 {time.time()-k0:.1f} 秒")
    console.log(f"    [稠密聚合] {len(agg_nodes)} 个聚合节点完成, 总计 {time.time()-t0:.1f} 秒")
    result = CachedEvaluator(data, node_cache).eval(ast)
    if not isinstance(result, pd.DataFrame):
        raise ExprError("公式计算结果为标量, 不是有效的截面因子")
    return _finalize(result)
