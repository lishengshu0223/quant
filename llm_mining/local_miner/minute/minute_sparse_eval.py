"""
本地化因子挖掘项目 - 分钟稀疏路径求值（minute_engine 拆分模块④）

职责:
- _BatchRunner: 单批次内的分钟/日频子表达式求值器(长表路径, 按股票批次流式计算)
- _cs_transform/_intraday_rolling: 分钟级截面变换与日内滚动算子(3D->3D)
- CachedEvaluator: 聚合节点命中缓存宽表后的整棵AST日频求值(全市场横截面)
- _wide_binop/_bool_wide: 日频宽表二元运算(稀疏/稠密/入口共用)

依赖: minute_data -> minute_parser -> expr_engine; 不含稠密路径(在 minute_dense_eval)。
"""

import numpy as np
import pandas as pd

from ..expr_engine import (
    ExprError, eval_call_daily, where_any, FUNC_SPEC,
    Num, Str, Var, Call, Bin, Unary, Ternary,
)
from .minute_parser import (
    MINUTE_CROSS_SPEC, MINUTE_ROLL_SPEC, MINUTE_AGG_SPEC, MINUTE_KEEP_SPEC,
    CROSS_SECTIONAL_OPS, _parse_time,
)
from .minute_data import MinuteExpr, _mk_df, SlicedData


def _binop_vals(op, a, b):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if op == "+":
            return (a + b).astype(np.float32)
        if op == "-":
            return (a - b).astype(np.float32)
        if op == "*":
            return (a * b).astype(np.float32)
        if op == "/":
            out = a / b
            out = np.where(np.isfinite(out), out, np.nan).astype(np.float32)
            return out
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
        if op == "&&":
            return np.asarray(a, dtype=bool) & np.asarray(b, dtype=bool)
        if op == "||":
            return np.asarray(a, dtype=bool) | np.asarray(b, dtype=bool)
    raise ExprError(f"分钟运算不支持 {op}")


def _m_binop(op, left, right):
    """分钟表达式二元运算(操作数在进入前已对齐为 MinuteExpr 或行对齐数组/标量)。
    注意保持操作数顺序: left op right, 混用"广播日频值(数组)"与"分钟值"时不能调换(非交换运算)。"""
    if isinstance(left, MinuteExpr) and isinstance(right, MinuteExpr):
        if not left.aligned(right):
            raise ExprError("分钟表达式无法对齐(两侧来自不同的分时/掩码结果), 不支持直接运算")
        code, date, t, a, b = left.code, left.date, left.t, left.v, right.v
    elif isinstance(left, MinuteExpr):
        code, date, t, a = left.code, left.date, left.t, left.v
        b = right
    elif isinstance(right, MinuteExpr):
        code, date, t = right.code, right.date, right.t
        a = left                      # 广播值(日频子表达式)在前, 保持 left op right 顺序
        b = right.v
    else:
        raise ExprError("分钟运算两侧都不是分钟表达式")
    return MinuteExpr(code, date, t, _binop_vals(op, a, b))


def _cs_transform(name, mv):
    """分钟级截面变换(3D->3D): 对(日期, 分钟)分组、在当日该分钟的全部股票上做截面。
    需全市场分钟数据在场(引擎在公式含截面算子时自动切换全市场分块模式)。"""
    df = _mk_df(mv.code, mv.date, mv.v, t=mv.t)
    g = df.groupby(["date", "t"])
    if name == "RANK":
        r = g["v"].rank(pct=True)
    elif name == "ZSCORE":
        mu = g["v"].transform("mean")
        sd = g["v"].transform("std")
        with np.errstate(divide="ignore", invalid="ignore"):
            r = ((df["v"] - mu) / sd).replace([np.inf, -np.inf], np.nan)
    elif name == "SCALE":
        df["av"] = df["v"].abs()
        s = df.groupby(["date", "t"])["av"].transform("sum").replace(0, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (df["v"] / s).replace([np.inf, -np.inf], np.nan)
    elif name == "CS_MEAN":
        r = g["v"].transform("mean")
    elif name == "CS_STD":
        r = g["v"].transform("std")
    else:
        raise ExprError(f"分钟截面算子 {name} 未实现")
    return MinuteExpr(mv.code, mv.date, mv.t, r.to_numpy(dtype=np.float32))


def _intraday_rolling(x, name, n):
    """日内滚动算子(3D->3D): 按股票、按天在分钟序列上滚动 n 分钟窗口, 保持每天240根分钟。
    每天前 n-1 根为 NaN(窗口不足)。"""
    df = _mk_df(x.code, x.date, x.v, t=x.t).sort_values(["code", "date", "t"]).reset_index(drop=True)
    grp = df.groupby(["code", "date"])["v"].rolling(n, min_periods=1)
    if name == "INTRADAY_MEAN":
        r = grp.mean()
    elif name == "INTRADAY_STD":
        r = grp.std()
    elif name == "INTRADAY_SUM":
        r = grp.sum()
    elif name == "INTRADAY_MAX":
        r = grp.max()
    elif name == "INTRADAY_MIN":
        r = grp.min()
    elif name == "INTRADAY_MEDIAN":
        r = grp.median()
    else:
        raise ExprError(f"日内滚动算子 {name} 未实现")
    # r 索引为 (code,date,原行号) 三层 MultiIndex, 排序后与 df 行序一致
    r = r.sort_index()
    return MinuteExpr(df["code"].to_numpy(), df["date"].to_numpy(), df["t"].to_numpy(),
                      r.to_numpy(dtype=np.float32))


def _subtree_has_cross(node) -> bool:
    """节点子树内是否含截面算子(分钟级或日频级, 出现在分钟聚合子表达式内需全市场分钟数据)"""
    if isinstance(node, Call):
        if node.name in CROSS_SECTIONAL_OPS:
            return True
        return any(_subtree_has_cross(a) for a in node.args)
    if isinstance(node, Bin):
        return _subtree_has_cross(node.left) or _subtree_has_cross(node.right)
    if isinstance(node, Unary):
        return _subtree_has_cross(node.x)
    if isinstance(node, Ternary):
        return (_subtree_has_cross(node.cond) or _subtree_has_cross(node.true)
                or _subtree_has_cross(node.false))
    return False


def _wide_binop(op, left, right):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            out = left / right
            if isinstance(out, pd.DataFrame):
                out = out.replace([np.inf, -np.inf], np.nan)
            return out
        if op == ">":
            return left > right
        if op == "<":
            return left < right
        if op == ">=":
            return left >= right
        if op == "<=":
            return left <= right
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        if op == "&&":
            return _bool_wide(left) & _bool_wide(right)
        if op == "||":
            return _bool_wide(left) | _bool_wide(right)
    raise ExprError(f"日频批内运算不支持 {op}")


def _bool_wide(x):
    if isinstance(x, pd.DataFrame):
        return x.fillna(False).astype(bool)
    return bool(x)


class _BatchRunner:
    """单批次内的分钟/日频子表达式求值器"""

    def __init__(self, mmd, data, cfg, types):
        self.mmd = mmd
        self.data = data
        self.cfg = cfg
        self.types = types
        self.frame = None
        self.batch_codes = []
        self.sliced_data = None

    def set_batch(self, frame, batch_codes):
        self.frame = frame
        self.batch_codes = batch_codes
        self.sliced_data = SlicedData(self.data, batch_codes)

    # ---- 参数按类型分发 ----
    def eval_arg(self, node):
        t = self.types.get(id(node))
        if t == "M":
            return self.eval_m(node)
        if isinstance(node, Num):
            return node.value
        if isinstance(node, Str):
            return node.value
        return self.eval_w_batch(node)

    # ---- 分钟上下文 ----
    def eval_m(self, node):
        if isinstance(node, Var):
            return self.mmd.field_of(self.frame, node.name)
        if isinstance(node, Unary):
            x = self.eval_m(node.x)
            return MinuteExpr(x.code, x.date, x.t, -x.v)
        if isinstance(node, Bin):
            left = self.eval_arg(node.left)
            right = self.eval_arg(node.right)
            if isinstance(left, MinuteExpr) and not isinstance(right, MinuteExpr):
                right = self._broadcast(right, left)
            elif isinstance(right, MinuteExpr) and not isinstance(left, MinuteExpr):
                left = self._broadcast(left, right)
            if isinstance(left, MinuteExpr) or isinstance(right, MinuteExpr):
                return _m_binop(node.op, left, right)
            raise ExprError("分钟上下文中的二元运算两侧都非分钟表达式")
        if isinstance(node, Ternary):
            c = self.eval_m(node.cond)
            tt = self.eval_m(node.true)
            ff = self.eval_m(node.false)
            if not (c.aligned(tt) and c.aligned(ff)):
                raise ExprError("分钟条件表达式的三个分支无法对齐")
            keep = c.v.astype(bool)
            return MinuteExpr(c.code, c.date, c.t, np.where(keep, tt.v, ff.v).astype(np.float32))
        if isinstance(node, Call):
            name = node.name
            if name in MINUTE_CROSS_SPEC:
                x = self.eval_m(node.args[0])
                return _cs_transform(name, x)
            if name in MINUTE_ROLL_SPEC:
                x = self.eval_m(node.args[0])
                n = int(node.args[1].value) if len(node.args) > 1 else 5
                return _intraday_rolling(x, name, n)
            if name == "SLICE":
                x = self.eval_m(node.args[0])
                return x.slice_time(_parse_time(node.args[1].value), _parse_time(node.args[2].value))
            if name == "MASK":
                x = self.eval_m(node.args[0])
                op = node.args[1].value
                thr = self.eval_arg(node.args[2])
                if not isinstance(thr, MinuteExpr):
                    thr = self._broadcast(thr, x)
                return x.mask(op, thr)
            raise ExprError(f"分钟上下文中不允许的函数 {name}")
        if isinstance(node, Str):
            raise ExprError("字符串常量不能作为分钟表达式")
        raise ExprError(f"分钟上下文中的非法节点: {type(node).__name__}")

    def _broadcast(self, val, mv: MinuteExpr):
        """把标量或批内宽表广播为与 mv 行对齐的数组"""
        if isinstance(val, (int, float, np.number)):
            return np.full(len(mv.v), float(val), dtype=np.float32)
        if not isinstance(val, pd.DataFrame):
            raise ExprError("分钟表达式与不支持的类型做运算")
        # val 的列是批内 code 位置(0..n-1); 把宽表索引转成整数日期后按行对齐
        w = val.reindex(columns=range(len(self.batch_codes)))
        s = w.stack(future_stack=True)  # MultiIndex (日期, code_pos)
        dates = s.index.get_level_values(0).strftime("%Y%m%d").astype(np.int32)
        s.index = pd.MultiIndex.from_arrays([dates, s.index.get_level_values(1)])
        idx = pd.MultiIndex.from_arrays([mv.date, mv.code])
        return s.reindex(idx).to_numpy(dtype=np.float32)

    # ---- 日频上下文(批内, 用于分钟聚合内的阈值/日频子表达式) ----
    def eval_w_batch(self, node):
        res = self._eval_w_inner(node)
        if isinstance(res, pd.DataFrame):
            res = res.copy()
            res.columns = range(len(self.batch_codes))
        return res

    def _eval_w_inner(self, node):
        if isinstance(node, Num):
            return node.value
        if isinstance(node, Str):
            return node.value
        if isinstance(node, Var):
            return self.sliced_data.var(node.name)
        if isinstance(node, Unary):
            return -self.eval_w_batch(node.x)
        if isinstance(node, Bin):
            return _wide_binop(node.op, self.eval_w_batch(node.left), self.eval_w_batch(node.right))
        if isinstance(node, Ternary):
            return where_any(self.eval_w_batch(node.cond),
                             self.eval_w_batch(node.true),
                             self.eval_w_batch(node.false))
        if isinstance(node, Call):
            name = node.name
            if name in MINUTE_AGG_SPEC and id(node) in self.types.get("__agg_ids__", set()):
                evals = [self.eval_arg(a) for a in node.args]
                return self.mmd.aggregate(name, evals)
            if name in MINUTE_KEEP_SPEC:
                raise ExprError("SLICE/MASK 只能在分钟上下文中使用")
            if name not in FUNC_SPEC:
                raise ExprError(f"分钟模式: 未声明的日频函数 {name}")
            args = [self.eval_w_batch(a) for a in node.args]
            return eval_call_daily(name, args)
        raise ExprError(f"日频批内求值: 非法节点 {type(node).__name__}")


# =============================================================================
# 对外求值(聚合缓存之上)
# =============================================================================

class CachedEvaluator:
    """整棵AST求值: 聚合节点命中缓存宽表, 其余走日频求值(全市场横截面)"""

    def __init__(self, data, node_cache):
        self.data = data
        self.node_cache = node_cache

    def eval(self, node):
        if isinstance(node, Num):
            return node.value
        if isinstance(node, Str):
            raise ExprError("字符串常量不能直接参与日频求值")
        if isinstance(node, Var):
            return self.data.var(node.name)
        if isinstance(node, Unary):
            return -self.eval(node.x)
        if isinstance(node, Bin):
            return _wide_binop(node.op, self.eval(node.left), self.eval(node.right))
        if isinstance(node, Ternary):
            return where_any(self.eval(node.cond), self.eval(node.true), self.eval(node.false))
        if isinstance(node, Call):
            if id(node) in self.node_cache:
                return self.node_cache[id(node)]
            if node.name in MINUTE_KEEP_SPEC:
                raise ExprError("SLICE/MASK 只能在分钟聚合算子内部使用")
            args = [self.eval(a) for a in node.args]
            return eval_call_daily(node.name, args)
        raise ExprError("未知AST节点")
