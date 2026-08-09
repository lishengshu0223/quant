"""
本地化因子挖掘项目 - 分钟频率因子表达式引擎(日频输出)

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
"""

import os
import re
import time

import h5py
import numpy as np
import pandas as pd

# numba 可选加速(稠密归约算子: 零中间3D数组 + 多线程). 未安装时自动回退 numpy 实现
try:
    from numba import njit, prange
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

from . import console
from .expr_engine import (
    ALLOWED_VARS, FUNC_SPEC, Num, Str, Var, Call, Bin, Unary, Ternary, Parser,
    ExprError, ast_depth, bracket_scan, eval_call_daily, tokenize, where_any,
)

# 本地分钟数据目录(与 local_api.stock_minute 一致: rq_backtest_data/h5/equities)
EQ_MINUTE_DIR = r"F:\Trade_data\rq_backtest_data\h5\equities"

# 分钟模式可用变量(在分钟算子的参数中解析为分钟序列)
MINUTE_FIELDS = {"$open", "$close", "$high", "$low", "$volume", "$amount", "$return", "$minute"}

# 聚合(降维)算子: 3D -> 2D. name: (最少参数, 最多参数, 各参数类型)
#   类型: 'M'=分钟序列表达式, 'S'=数字常量(或字符串常量)
MINUTE_AGG_SPEC = {
    "SUM": (1, 1, ["M"]),
    "MEAN": (1, 1, ["M"]),
    "STD": (1, 1, ["M"]),
    "MAX": (1, 1, ["M"]),
    "MIN": (1, 1, ["M"]),
    "MEDIAN": (1, 1, ["M"]),
    "SKEW": (1, 1, ["M"]),
    "KURT": (1, 1, ["M"]),
    "LAST": (1, 1, ["M"]),
    "FIRST": (1, 1, ["M"]),
    "COUNT": (1, 1, ["M"]),
    "QUANTILE": (2, 2, ["M", "S"]),
    "CORR": (2, 2, ["M", "M"]),
    "TS_AUTOCORR": (1, 2, ["M", "S"]),
    "TS_ARGMAX": (1, 1, ["M"]),
    "TS_ARGMIN": (1, 1, ["M"]),
    "REGRESSION_SLOPE": (2, 2, ["M", "M"]),
    "REGRESSION_INTERCEPT": (2, 2, ["M", "M"]),
}

# 维持维度算子: 3D -> 3D
MINUTE_KEEP_SPEC = {
    "SLICE": (3, 3),   # SLICE(x, "HH:MM", "HH:MM"): 保留该时段的分钟
    "MASK": (3, 3),    # MASK(x, 比较符, 阈值): 只保留 x 满足比较条件的分钟; 阈值可为日频标量(宽表/标量)或分钟表达式
}

# 分钟级截面算子(3D -> 3D): 对"当日该分钟的全部股票"做截面变换, 不减少每天分钟数。
# 需要全市场分钟数据同时在场, 引擎检测到后自动切换为"按日期分块、每块加载全市场"的全市场模式。
# RANK/ZSCORE/SCALE 与日频函数同名: 参数为分钟序列时按分钟截面解析, 参数为日频值时按日频函数解析。
MINUTE_CROSS_SPEC = {"RANK", "ZSCORE", "SCALE", "CS_MEAN", "CS_STD"}

# 日内滚动算子(3D -> 3D): 按股票、按天在分钟序列上滚动 n 分钟窗口, 保持每天240根分钟。
MINUTE_ROLL_SPEC = {
    "INTRADAY_MEAN": (1, 2), "INTRADAY_STD": (1, 2), "INTRADAY_SUM": (1, 2),
    "INTRADAY_MAX": (1, 2), "INTRADAY_MIN": (1, 2), "INTRADAY_MEDIAN": (1, 2),
}
MINUTE_MAX_ROLL_WINDOW = 240   # 日内滚动窗口上限(1分钟线一天240根)

# 日频截面算子(出现在分钟聚合子表达式内时需全市场分钟数据, 由引擎自动切换全市场模式)
CROSS_SECTIONAL_OPS = {"RANK", "ZSCORE", "SCALE", "CS_MEAN", "CS_STD"}

TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$")
MASK_OPS = {">", "<", ">=", "<=", "==", "!="}
INVALID_CHAR_MINUTE_RE = re.compile(r"[^A-Za-z0-9_$+\-*/><=&|?:.,()\"'\s]")

MINUTE_MAX_LAG = 240           # TS_AUTOCORR 滞后上限(1分钟线一天240根)
MINUTE_MAX_BASE_FEATURES = 12  # 分钟模式允许的基础变量个数(比日频宽松)


def _parse_time(s: str) -> int:
    """把 "HH:MM[:SS]" 解析为当日秒数"""
    if not isinstance(s, str):
        raise ExprError(f"时间常量必须是字符串(如\"09:31\"), 实际: {s!r}")
    m = TIME_RE.match(s.strip())
    if not m:
        raise ExprError(f"时间常量格式应为\"HH:MM\"或\"HH:MM:SS\", 实际: {s!r}")
    hh, mm = int(m.group(1)), int(m.group(2))
    ss = int(m.group(3)) if m.group(3) else 0
    if hh > 23 or mm > 59 or ss > 59:
        raise ExprError(f"时间常量越界: {s!r}")
    return hh * 3600 + mm * 60 + ss


def canonical(node) -> str:
    """节点的规范化字符串(用于聚合子树去重缓存)"""
    if isinstance(node, Num):
        return str(node.value)
    if isinstance(node, Str):
        return repr(node.value)
    if isinstance(node, Var):
        return node.name
    if isinstance(node, Unary):
        return f"-{canonical(node.x)}"
    if isinstance(node, Bin):
        return f"({canonical(node.left)}{node.op}{canonical(node.right)})"
    if isinstance(node, Ternary):
        return f"({canonical(node.cond)}?{canonical(node.true)}:{canonical(node.false)})"
    if isinstance(node, Call):
        return f"{node.name}({','.join(canonical(a) for a in node.args)})"
    raise ExprError("未知AST节点")


# =============================================================================
# 类型推断与校验
# =============================================================================

def _infer(node, exp, types, agg_ids, max_window, in_mctx=False):
    """
    递归类型推断。
    exp: 期望类型 'W'|'M'|'S'
    types: id(node) -> 'W'|'M'|'S'
    agg_ids: 记录真正"分钟聚合"的节点id(排除歧义后退化为日频函数的节点)
    in_mctx: 是否处于分钟上下文(用于禁止截面算子出现在分钟子表达式内)
    """
    if isinstance(node, Num):
        types[id(node)] = "S"
        return
    if isinstance(node, Str):
        types[id(node)] = "S"
        return
    if isinstance(node, Var):
        if exp == "M":
            if node.name not in MINUTE_FIELDS:
                raise ExprError(f"分钟上下文中变量 {node.name} 不可用, 可用: {sorted(MINUTE_FIELDS)}")
            types[id(node)] = "M"
        else:
            if node.name not in ALLOWED_VARS:
                raise ExprError(f"日频上下文中变量 {node.name} 不可用, 可用: {sorted(ALLOWED_VARS)}")
            types[id(node)] = "W"
        return
    if isinstance(node, Unary):
        _infer(node.x, exp, types, agg_ids, max_window, in_mctx)
        types[id(node)] = types[id(node.x)]
        return
    if isinstance(node, Ternary):
        _infer(node.cond, exp, types, agg_ids, max_window, in_mctx)
        _infer(node.true, exp, types, agg_ids, max_window, in_mctx)
        _infer(node.false, exp, types, agg_ids, max_window, in_mctx)
        t = {types[id(node.true)], types[id(node.false)]}
        types[id(node)] = "M" if "M" in t else "W"
        return
    if isinstance(node, Bin):
        _infer(node.left, exp, types, agg_ids, max_window, in_mctx)
        _infer(node.right, exp, types, agg_ids, max_window, in_mctx)
        lt, rt = types[id(node.left)], types[id(node.right)]
        types[id(node)] = "M" if (lt == "M" or rt == "M") else "W"
        return
    # ---- Call ----
    name = node.name
    # ① 分钟级截面算子(参数为分钟序列时): 3D->3D, 需全市场分钟数据(引擎自动切换全市场模式)
    if name in MINUTE_CROSS_SPEC:
        if len(node.args) == 1:
            _infer(node.args[0], "M", types, agg_ids, max_window, True)
            if types[id(node.args[0])] == "M":
                types[id(node)] = "M"
                return
    # ①b 日内滚动算子(3D->3D): INTRADAY_MEAN(x, n) 等
    if name in MINUTE_ROLL_SPEC:
        mn, mx = MINUTE_ROLL_SPEC[name]
        if not (mn <= len(node.args) <= mx):
            raise ExprError(f"函数 {name} 参数个数应为 {mn}~{mx}, 实际 {len(node.args)}")
        _infer(node.args[0], "M", types, agg_ids, max_window, True)
        if len(node.args) == 2:
            if not isinstance(node.args[1], Num) or not float(node.args[1].value).is_integer() \
                    or not (1 <= int(node.args[1].value) <= MINUTE_MAX_ROLL_WINDOW):
                raise ExprError(f"函数 {name} 的窗口参数必须是 1~{MINUTE_MAX_ROLL_WINDOW} 的整数")
        types[id(node)] = "M"
        return
    # ② 聚合算子: 先尝试"分钟聚合"解释(参数在分钟上下文)
    if name in MINUTE_AGG_SPEC:
        mn, mx, kinds = MINUTE_AGG_SPEC[name]
        if mn <= len(node.args) <= mx:
            arg_ts = []
            ok = True
            for arg in node.args:
                _infer(arg, "M", types, agg_ids, max_window, True)
                arg_ts.append(types[id(arg)])
            if all(t in ("M", "S") for t in arg_ts):
                types[id(node)] = "W"
                agg_ids.add(id(node))
                # 参数数值校验
                if name == "QUANTILE":
                    if not isinstance(node.args[1], Num) or not (0 < node.args[1].value < 1):
                        raise ExprError("QUANTILE 的分位数参数必须是 0~1 的小数")
                if name == "TS_AUTOCORR":
                    if not isinstance(node.args[1], Num) or not float(node.args[1].value).is_integer() \
                            or not (1 <= int(node.args[1].value) <= MINUTE_MAX_LAG):
                        raise ExprError(f"TS_AUTOCORR 的滞后参数必须是 1~{MINUTE_MAX_LAG} 的整数")
                return
    # ③ 维持维度算子
    if name in MINUTE_KEEP_SPEC:
        if name == "SLICE":
            if len(node.args) != 3:
                raise ExprError("SLICE 需要3个参数: SLICE(分钟表达式, 开始时间, 结束时间)")
            _infer(node.args[0], "M", types, agg_ids, max_window, True)
            for a in node.args[1:]:
                if not isinstance(a, Str):
                    raise ExprError("SLICE 的时间参数必须是字符串常量(如\"09:31\")")
            s, e = _parse_time(node.args[1].value), _parse_time(node.args[2].value)
            if s > e:
                raise ExprError(f"SLICE 开始时间 {node.args[1].value} 晚于结束时间 {node.args[2].value}")
            types[id(node)] = "M"
            return
        # MASK(x, 比较符, 阈值)
        if len(node.args) != 3:
            raise ExprError("MASK 需要3个参数: MASK(分钟表达式, 比较符, 阈值)")
        _infer(node.args[0], "M", types, agg_ids, max_window, True)
        op_node = node.args[1]
        if not isinstance(op_node, Str) or op_node.value not in MASK_OPS:
            raise ExprError(f"MASK 的比较符必须是 {'/'.join(sorted(MASK_OPS))} 之一(字符串常量)")
        # 阈值: 期望分钟上下文(裸变量按分钟解析; 聚合/日频表达式按其自身类型)
        _infer(node.args[2], "M", types, agg_ids, max_window, True)
        types[id(node)] = "M"
        return
    # ④ 日频函数
    if name in FUNC_SPEC:
        min_a, max_a, win_idx, float_idx = FUNC_SPEC[name]
        if not (min_a <= len(node.args) <= max_a):
            raise ExprError(f"函数 {name} 参数个数应为 {min_a}~{max_a}, 实际 {len(node.args)}")
        for i, arg in enumerate(node.args):
            if i in win_idx:
                if not isinstance(arg, Num):
                    raise ExprError(f"函数 {name} 的第{i+1}个参数(时间窗口)必须是数字常量")
                n = arg.value
                if not float(n).is_integer() or int(n) < 1 or int(n) > max_window:
                    raise ExprError(f"函数 {name} 的时间窗口必须是 1~{max_window} 的整数, 实际 {n}")
                _infer(arg, "S", types, agg_ids, max_window, in_mctx)
            elif i in float_idx:
                if not isinstance(arg, Num) or not (0 < arg.value < 1):
                    raise ExprError(f"函数 {name} 的分位数参数必须是 0~1 的小数")
                _infer(arg, "S", types, agg_ids, max_window, in_mctx)
            else:
                _infer(arg, "W", types, agg_ids, max_window, in_mctx)
        types[id(node)] = "W"
        return
    raise ExprError(f"分钟模式: 未声明的函数或变量 {name}")


def _walk_collect_agg(node, inside_agg: bool, agg_ids: set, out: list):
    """收集"最大"分钟聚合节点(未被其它聚合节点包含的)"""
    if isinstance(node, Call):
        is_agg = id(node) in agg_ids
        if is_agg and not inside_agg:
            out.append(node)
        for a in node.args:
            _walk_collect_agg(a, inside_agg or is_agg, agg_ids, out)
    elif isinstance(node, Bin):
        _walk_collect_agg(node.left, inside_agg, agg_ids, out)
        _walk_collect_agg(node.right, inside_agg, agg_ids, out)
    elif isinstance(node, Unary):
        _walk_collect_agg(node.x, inside_agg, agg_ids, out)
    elif isinstance(node, Ternary):
        _walk_collect_agg(node.cond, inside_agg, agg_ids, out)
        _walk_collect_agg(node.true, inside_agg, agg_ids, out)
        _walk_collect_agg(node.false, inside_agg, agg_ids, out)


def minute_validate(expr: str, cfg):
    """
    分钟模式公式校验, 返回 (ast, types, agg_nodes)。
    types: id(node) -> 'W'|'M'|'S'; agg_nodes: 最大分钟聚合节点列表。
    """
    if not isinstance(expr, str) or not expr.strip():
        raise ExprError("公式为空")
    expr = expr.strip()

    max_len = getattr(cfg, "formula_max_symbol_length", cfg.max_symbol_length)
    max_depth = getattr(cfg, "formula_max_depth", cfg.max_depth)
    if len(expr) > max_len:
        raise ExprError(f"公式长度 {len(expr)} 超过上限 {max_len}")
    m = INVALID_CHAR_MINUTE_RE.search(expr)
    if m:
        raise ExprError(f"公式含非法字符: '{m.group()}'")
    bd = bracket_scan(expr)
    if bd > max_depth:
        raise ExprError(f"括号嵌套深度 {bd} 超过上限 {max_depth}")

    ast = Parser(tokenize(expr)).parse()
    d = ast_depth(ast)
    if d > max_depth:
        raise ExprError(f"语法树深度 {d} 超过上限 {max_depth}")

    types = {}
    agg_ids = set()
    _infer(ast, "W", types, agg_ids, cfg.max_window)
    root_t = types.get(id(ast))
    if root_t != "W":
        raise ExprError(
            "公式最外层结果必须是日频宽表(需包含至少一个分钟聚合算子将分钟数据降维成日频), "
            f"当前最外层类型为 {root_t}")

    agg_nodes = []
    _walk_collect_agg(ast, False, agg_ids, agg_nodes)

    # 基础变量个数(分钟+日频)
    max_base = getattr(cfg, "minute_max_base_features", MINUTE_MAX_BASE_FEATURES)
    vars_used = set()
    _collect_vars(ast, vars_used)
    if len(vars_used) > max_base:
        raise ExprError(f"基础变量个数 {len(vars_used)} 超过上限 {max_base}")
    return ast, types, agg_nodes, agg_ids


def _collect_vars(node, out: set):
    if isinstance(node, Var):
        out.add(node.name)
    elif isinstance(node, Bin):
        _collect_vars(node.left, out)
        _collect_vars(node.right, out)
    elif isinstance(node, Unary):
        _collect_vars(node.x, out)
    elif isinstance(node, Ternary):
        _collect_vars(node.cond, out)
        _collect_vars(node.true, out)
        _collect_vars(node.false, out)
    elif isinstance(node, Call):
        for a in node.args:
            _collect_vars(a, out)


def minute_fields_used(ast, types) -> list:
    """公式中实际使用的分钟字段(h5字段名)"""
    fields = set()

    def walk(node):
        if isinstance(node, Var) and types.get(id(node)) == "M":
            f = node.name[1:]
            if f != "minute":
                fields.add(f)
        elif isinstance(node, Call):
            for a in node.args:
                walk(a)
        elif isinstance(node, Bin):
            walk(node.left)
            walk(node.right)
        elif isinstance(node, Unary):
            walk(node.x)
        elif isinstance(node, Ternary):
            walk(node.cond)
            walk(node.true)
            walk(node.false)

    walk(ast)
    return sorted(fields)


# =============================================================================
# 分钟表达式(三维数据)
# =============================================================================

class MinuteExpr:
    """
    分钟表达式: (code_idx, date, t) 三列整数键 + 值数组 v。
    code_idx: 当前批次内的股票编号; date: YYYYMMDD 整数; t: 当日秒数。
    """
    __slots__ = ("code", "date", "t", "v")

    def __init__(self, code, date, t, v):
        self.code = code
        self.date = date
        self.t = t
        self.v = v

    def aligned(self, other) -> bool:
        return (np.array_equal(self.code, other.code)
                and np.array_equal(self.date, other.date)
                and np.array_equal(self.t, other.t))

    def slice_time(self, s_sec: int, e_sec: int) -> "MinuteExpr":
        keep = (self.t >= s_sec) & (self.t <= e_sec)
        return MinuteExpr(self.code[keep], self.date[keep], self.t[keep], self.v[keep])

    def mask(self, op: str, threshold) -> "MinuteExpr":
        """只保留 x 满足 op 条件的分钟。threshold: 标量 | 行对齐数组 | MinuteExpr(行级比较)"""
        if isinstance(threshold, MinuteExpr):
            if not self.aligned(threshold):
                raise ExprError("MASK 的阈值分钟表达式与主表达式无法对齐(来自不同的分时/掩码结果)")
            th = threshold.v
        else:
            th = threshold
        v = self.v
        if op == ">":
            keep = v > th
        elif op == "<":
            keep = v < th
        elif op == ">=":
            keep = v >= th
        elif op == "<=":
            keep = v <= th
        elif op == "==":
            keep = v == th
        elif op == "!=":
            keep = v != th
        else:
            raise ExprError(f"MASK 比较符 {op} 不受支持")
        return MinuteExpr(self.code[keep], self.date[keep], self.t[keep], v[keep])


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


# =============================================================================
# 聚合实现(按 code,date 分组 -> 日频宽表)
# =============================================================================

def _mk_df(code, date, v, **cols):
    d = {"code": code, "date": date, "v": v}
    d.update(cols)
    return pd.DataFrame(d)


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


def _agg_series(name, mv, extra):
    """对分钟表达式聚合, 返回 (code,date) MultiIndex Series"""
    code, date, v = mv.code, mv.date, mv.v

    if name in ("SUM", "MEAN", "STD", "MAX", "MIN", "MEDIAN", "SKEW", "QUANTILE"):
        df = _mk_df(code, date, v)
        g = df.groupby(["code", "date"])
        if name == "SUM":
            ser = g["v"].sum(min_count=1)
        elif name == "MEAN":
            ser = g["v"].mean()
        elif name == "STD":
            ser = g["v"].std()
        elif name == "MAX":
            ser = g["v"].max()
        elif name == "MIN":
            ser = g["v"].min()
        elif name == "MEDIAN":
            ser = g["v"].median()
        elif name == "SKEW":
            ser = g["v"].skew()
            ser = ser.where(g.size() >= 3)
        else:  # QUANTILE
            ser = g["v"].quantile(float(extra[0]))
        return ser

    if name == "KURT":
        df = _mk_df(code, date, v.astype(np.float64))
        df = df[df["v"].notna()]
        if df.empty:
            return pd.Series(dtype=float)
        df["mu"] = df.groupby(["code", "date"])["v"].transform("mean")
        df["d2"] = (df["v"] - df["mu"]) ** 2
        df["d4"] = df["d2"] ** 2
        g = df.groupby(["code", "date"])
        n = g.size().astype(np.float64)
        m2 = g["d2"].sum()
        m4 = g["d4"].sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            # 与 scipy.stats.kurtosis(bias=False) 一致的偏差校正超额峰度
            mu2 = m2 / n
            mu4 = m4 / n
            kurt = (n - 1.0) / ((n - 2.0) * (n - 3.0)) * ((n + 1.0) * (mu4 / mu2 ** 2.0) - 3.0 * (n - 1.0))
        return kurt.where((n >= 4) & (mu2 > 0)).replace([np.inf, -np.inf], np.nan)

    if name == "COUNT":
        df = _mk_df(code, date, np.where(np.isnan(v), 0.0, 1.0))
        return df.groupby(["code", "date"])["v"].sum()

    if name in ("LAST", "FIRST"):
        df = _mk_df(code, date, v, t=mv.t).sort_values(["code", "date", "t"])
        if name == "LAST":
            df["vv"] = df.groupby(["code", "date"])["v"].ffill()
        else:
            df["vv"] = df.groupby(["code", "date"])["v"].bfill()
        return df.groupby(["code", "date"])["vv"].last()

    if name == "CORR":
        y = extra[0]
        if not mv.aligned(y):
            raise ExprError("CORR 的两个分钟表达式无法对齐")
        return _corr_grouped(code, date, v, y.v)

    if name == "TS_AUTOCORR":
        lag = int(extra[0])
        df = _mk_df(code, date, v, t=mv.t).sort_values(["code", "date", "t"])
        df["vl"] = df.groupby(["code", "date"])["v"].shift(lag)
        return _corr_grouped(df["code"].to_numpy(), df["date"].to_numpy(),
                             df["v"].to_numpy(), df["vl"].to_numpy())

    if name in ("REGRESSION_SLOPE", "REGRESSION_INTERCEPT"):
        y = extra[0]
        if not mv.aligned(y):
            raise ExprError(f"{name} 的两个分钟表达式无法对齐")
        df = _mk_df(code, date, v.astype(np.float64), y=y.v.astype(np.float64))
        df = df[df["v"].notna() & df["y"].notna()]
        if df.empty:
            return pd.Series(dtype=float)
        df["yy"] = df["y"] * df["y"]
        df["xy"] = df["v"] * df["y"]
        g = df.groupby(["code", "date"])
        n = g.size()
        sx, sy = g["v"].sum(), g["y"].sum()
        syy, sxy = g["yy"].sum(), g["xy"].sum()
        denom = n * syy - sy * sy
        slope = (n * sxy - sx * sy) / denom
        with np.errstate(divide="ignore", invalid="ignore"):
            intercept = (sx - slope * sy) / n
        ok = (n >= 2) & (denom > 0)
        res = slope if name == "REGRESSION_SLOPE" else intercept
        return res.where(ok).replace([np.inf, -np.inf], np.nan)

    if name in ("TS_ARGMAX", "TS_ARGMIN"):
        df = _mk_df(code, date, v, t=mv.t)
        df["pos"] = df.groupby(["code", "date"]).cumcount()
        dfv = df[df["v"].notna()]
        if dfv.empty:
            return pd.Series(dtype=float)
        g = dfv.groupby(["code", "date"])
        idx = g["v"].idxmax() if name == "TS_ARGMAX" else g["v"].idxmin()
        # idx: index=(code,date) 分组键, 值为 dfv 内的行标签 -> 取出该行组内位置
        pos = dfv["pos"].reindex(idx.to_numpy())
        pos.index = idx.index
        n_all = df.groupby(["code", "date"]).size().rename("n")
        res_df = pos.rename("pos").to_frame().join(n_all, how="left")
        with np.errstate(divide="ignore", invalid="ignore"):
            res = res_df["pos"] / (res_df["n"] - 1)
        return res.where(res_df["n"] >= 2)

    raise ExprError(f"聚合算子 {name} 未实现")


def _corr_grouped(code, date, x, y):
    df = _mk_df(code, date, x.astype(np.float64), y=y.astype(np.float64))
    df = df[df["v"].notna() & df["y"].notna()]
    if df.empty:
        return pd.Series(dtype=float)
    df["xx"] = df["v"] * df["v"]
    df["yy"] = df["y"] * df["y"]
    df["xy"] = df["v"] * df["y"]
    g = df.groupby(["code", "date"])
    n = g.size()
    sx, sy = g["v"].sum(), g["y"].sum()
    sxx, syy, sxy = g["xx"].sum(), g["yy"].sum(), g["xy"].sum()
    denom = np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
    corr = (n * sxy - sx * sy) / denom
    return corr.where((n >= 2) & (denom > 0)).replace([np.inf, -np.inf], np.nan)


def _ser_to_wide(ser: pd.Series, daily_index, batch_codes):
    """(code,date) Series -> 日频宽表(index=日期, columns=股票代码)"""
    if ser.empty:
        return pd.DataFrame(index=daily_index, columns=batch_codes, dtype="float32")
    df = ser.unstack("date")  # index=code_idx, columns=date_int
    df.columns = pd.to_datetime(df.columns.astype(str), format="%Y%m%d")
    df = df.T  # index=日期, columns=code_idx
    df.columns = [batch_codes[i] for i in df.columns]
    return df.reindex(index=daily_index, columns=batch_codes)


# =============================================================================
# 分钟字段稠密内存缓存(可选): 把常用字段一次性读入 [日×240×股] 稠密矩阵, worker 进程内跨因子复用
# =============================================================================

_MINUTE_MEMORY_CACHE = {}   # id(data) -> MinuteFieldCache


def _raw_field_name(fld: str) -> str:
    return "total_turnover" if fld == "amount" else fld


def _get_memory_cache(mmd) -> "MinuteFieldCache | None":
    """按 cfg.minute_memory_fields 配置构建/复用与 data 绑定的分钟内存缓存(每进程仅构建一次)。"""
    cfg = mmd.cfg
    fields = [f.strip() for f in (getattr(cfg, "minute_memory_fields", "") or "").split(",") if f.strip()]
    if not fields:
        return None
    raw_fields = frozenset({_raw_field_name(f) for f in fields})
    key = id(mmd.data)
    cache = _MINUTE_MEMORY_CACHE.get(key)
    if cache is not None and cache.fields == raw_fields:
        return cache
    console.log(f"    [分钟内存] 一次性读入全市场分钟字段 {sorted(raw_fields)} 到内存"
                f"(每字段约13.7GB, 约15-25分钟, 仅一次)...")
    t0 = time.time()
    cache = MinuteFieldCache(mmd, raw_fields)
    _MINUTE_MEMORY_CACHE[key] = cache
    console.log(f"    [分钟内存] 构建完成, 耗时 {time.time()-t0:.0f} 秒, "
                f"常驻字段 {sorted(raw_fields)}")
    return cache


class MinuteFieldCache:
    """分钟字段稠密缓存 [n_days × 240 × n_stocks] float32, 无成交位置为NaN。
    一个 worker 进程内共享, 后续因子计算直接从内存聚合, 不再重复读盘。"""

    def __init__(self, mmd, fields: frozenset):
        self.fields = fields
        self.day_ints = np.array([int(str(d.date()).replace("-", "")) for d in mmd.daily_index],
                                 dtype=np.int64)
        self.t_grid = np.concatenate([np.arange(34260, 41460, 60),
                                      np.arange(46860, 54060, 60)]).astype(np.int32)
        assert len(self.t_grid) == 240, "1分钟线一天应为240根"
        self.codes = mmd.codes_with_minute
        self.col_map = {c: i for i, c in enumerate(self.codes)}
        self.arrays = self._build(mmd)
        # $return 派生数组(与 close 同型, float32): _build 内用原始float64 close 逐股票构建,
        # 与长表 groupby(code,date)["close"].pct_change(fill_method=None) 数值一致(避免float32舍入放大)。
        self.ret_arr = self._ret if "close" in self.fields else None

    def _build(self, mmd) -> dict:
        n_days, n_stocks = len(self.day_ints), len(self.codes)
        arrays = {f: np.full((n_days, 240, n_stocks), np.nan, dtype=np.float32) for f in self.fields}
        self._ret = np.full((n_days, 240, n_stocks), np.nan, dtype=np.float32) if "close" in self.fields else None
        start_int, end_int = mmd.start_int, mmd.end_int
        t0 = time.time()
        for col, code in enumerate(self.codes):
            fp = os.path.join(EQ_MINUTE_DIR, f"{code}.h5")
            try:
                with h5py.File(fp, "r") as f:
                    idx = f["index"][:]
                    dset = f["data"]
                    mask = (idx["date"] >= start_int) & (idx["date"] <= end_int)
                    if not mask.any():
                        continue
                    sel = np.where(mask)[0]
                    s = int(idx["line_no"][sel[0]])
                    e = int(idx["line_no"][sel[-1] + 1]) if sel[-1] + 1 < len(idx) else dset.shape[0]
                    raw = dset[s:e]
            except Exception as exc:  # 单只股票读取失败, 跳过(不影响其它股票)
                console.log(f"    [分钟内存] 读取 {code} 失败: {exc}")
                continue
            dt = raw["datetime"]
            y = dt // 10000000000
            mo = (dt // 100000000) % 100
            d = (dt // 1000000) % 100
            hh = (dt // 10000) % 100
            mm = (dt // 100) % 100
            ss = dt % 100
            dates = y * 10000 + mo * 100 + d
            ts = hh * 3600 + mm * 60 + ss
            didx = np.searchsorted(self.day_ints, dates, side="right") - 1
            ok = (didx >= 0) & (self.day_ints[didx] == dates)
            # 上午 09:31~11:30 (t∈[34260,41400]) 为 0..119; 下午 13:01~15:00 (t∈[46860,54000]) 为 120..239
            tidx = np.where(ts < 46860, (ts - 34260) // 60, 120 + (ts - 46860) // 60)
            ok &= (tidx >= 0) & (tidx < 240) & (self.t_grid[tidx] == ts)
            didx = didx[ok]
            tidx = tidx[ok]
            if len(didx) == 0:
                continue
            for f in self.fields:
                arrays[f][didx, tidx, col] = raw[f][ok].astype(np.float32)
            if self._ret is not None:
                # 用原始float64 close 逐股票构建 $return(当日有效分钟序列内 pct_change, 跨日断):
                # 与长表 groupby(code,date)["close"].pct_change(fill_method=None) 一致
                o = np.lexsort((tidx, didx))
                s_d, s_t = didx[o], tidx[o]
                c = raw["close"][ok][o].astype(np.float64)
                r = np.full(len(c), np.nan)
                with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                    r[1:] = np.where(s_d[1:] == s_d[:-1], (c[1:] - c[:-1]) / c[:-1], np.nan)
                self._ret[s_d, s_t, col] = r.astype(np.float32)
            if (col + 1) % 1000 == 0:
                console.log(f"    [分钟内存] 已读入 {col+1}/{n_stocks} 只, 耗时 {time.time()-t0:.0f} 秒")
        return arrays


# =============================================================================
# 分钟数据加载与分批求值
# =============================================================================

class SlicedData:
    """按批内股票切片后的日频数据视图(供日频子表达式在批内求值)"""

    def __init__(self, data, codes):
        self.data = data
        self.codes = codes

    def var(self, name):
        return self.data.var(name).reindex(columns=self.codes)


class MinuteMarketData:
    """全市场分钟数据分批加载器 + 聚合节点计算"""

    def __init__(self, cfg, data, fields):
        self.cfg = cfg
        self.data = data
        # $return 由 $close 的分钟环比计算, 需要额外加载 close
        if "return" in fields and "close" not in fields:
            fields = list(fields) + ["close"]
        self.fields = [f for f in fields if f != "minute"]
        self.daily_index = data.close.index
        self.all_codes = list(data.close.columns)
        self.start_int = int(cfg.data_start_date.replace("-", ""))
        self.end_int = int(str(data.close.index.max().date()).replace("-", ""))
        self.batch_size = int(getattr(cfg, "minute_batch_size", 35))
        self.batch_codes = []
        # 只有存在分钟文件的股票才进入批次; 其余股票因子恒为NaN
        self.codes_with_minute = [c for c in self.all_codes
                                  if os.path.exists(os.path.join(EQ_MINUTE_DIR, f"{c}.h5"))]
        self.batches = [self.codes_with_minute[i:i + self.batch_size]
                        for i in range(0, len(self.codes_with_minute), self.batch_size)]

    def _load_batch(self, batch_codes, start_int=None, end_int=None):
        """加载一批股票的分钟长表(批内 code_idx 编码, 已按 code,date,t 排序)。
        start_int/end_int: 可选, 只加载该日期整数区间(YYYYMMDD); 默认全区间。
        若公式所需字段已常驻内存(稠密缓存), 直接从内存构建, 避免重复读盘。"""
        cache = _get_memory_cache(self)
        if cache is not None:
            needed = {_raw_field_name(f) for f in self.fields if f != "return"}
            if needed <= cache.fields:
                return self._load_batch_from_memory(batch_codes, cache, start_int, end_int)
        if start_int is None:
            start_int = self.start_int
        if end_int is None:
            end_int = self.end_int
        rows = []
        for ci, code in enumerate(batch_codes):
            fp = os.path.join(EQ_MINUTE_DIR, f"{code}.h5")
            if not os.path.exists(fp):
                continue
            try:
                with h5py.File(fp, "r") as f:
                    idx = f["index"][:]
                    dset = f["data"]
                    mask = (idx["date"] >= start_int) & (idx["date"] <= end_int)
                    if not mask.any():
                        continue
                    sel = np.where(mask)[0]
                    s = int(idx["line_no"][sel[0]])
                    e = int(idx["line_no"][sel[-1] + 1]) if sel[-1] + 1 < len(idx) else dset.shape[0]
                    raw = dset[s:e]
            except Exception as exc:  # 单只股票读取失败, 跳过(不影响其它股票)
                console.log(f"    [分钟数据] 读取 {code} 失败: {exc}")
                continue
            dt = raw["datetime"]
            y = dt // 10000000000
            mo = (dt // 100000000) % 100
            d = (dt // 1000000) % 100
            hh = (dt // 10000) % 100
            mm = (dt // 100) % 100
            ss = dt % 100
            rec = {
                "code": np.full(len(raw), ci, dtype=np.int32),
                "date": (y * 10000 + mo * 100 + d).astype(np.int32),
                "t": (hh * 3600 + mm * 60 + ss).astype(np.int32),
            }
            for fld in self.fields:
                if fld == "return":
                    continue  # return 在拼接后统一计算
                raw_fld = "total_turnover" if fld == "amount" else fld
                rec[raw_fld] = raw[raw_fld].astype(np.float32)
            rows.append(rec)
        if not rows:
            return None
        data = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
        # 每只股票的文件内已按(date,t)有序、批内按 batch_codes 顺序追加 -> 拼接后天然按(code,date,t)有序,
        # 无需全局 sort_values(mergesort 2.8亿行会额外占用数GB内存, 也是此前批次过大时MemoryError的来源)
        df = pd.DataFrame(data)
        if "return" in self.fields:
            if "close" not in self.fields:
                raise ExprError("分钟 $return 依赖 $close, 请同时使用 $close")
            df["return"] = df.groupby(["code", "date"])["close"].pct_change(fill_method=None)
        return df

    def _load_batch_from_memory(self, batch_codes, cache, start_int=None, end_int=None):
        """从稠密内存缓存构建批次长表(与磁盘路径输出完全一致: code/date/t 有序 + 字段列)"""
        if start_int is None:
            start_int = self.start_int
        if end_int is None:
            end_int = self.end_int
        lo = np.searchsorted(cache.day_ints, start_int)
        hi = np.searchsorted(cache.day_ints, end_int, side="right")
        valid_field = "close" if "close" in cache.fields else sorted(cache.fields)[0]
        rows = []
        for ci, code in enumerate(batch_codes):
            col = cache.col_map.get(code)
            if col is None:
                continue
            day_idx, t_idx = np.nonzero(np.isfinite(cache.arrays[valid_field][lo:hi, :, col]))
            if len(day_idx) == 0:
                continue
            day_idx = day_idx + lo
            rec = {
                "code": np.full(len(day_idx), ci, dtype=np.int32),
                "date": cache.day_ints[day_idx].astype(np.int32),
                "t": cache.t_grid[t_idx].astype(np.int32),
            }
            for fld in self.fields:
                if fld == "return":
                    continue
                raw = _raw_field_name(fld)
                rec[raw] = cache.arrays[raw][day_idx, t_idx, col].astype(np.float32)
            rows.append(rec)
        if not rows:
            return None
        data = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
        df = pd.DataFrame(data)
        if "return" in self.fields:
            df["return"] = df.groupby(["code", "date"])["close"].pct_change(fill_method=None)
        return df

    def field_of(self, df, name):
        """从批次长表取分钟字段, 包装为 MinuteExpr"""
        code = df["code"].to_numpy()
        date = df["date"].to_numpy()
        t = df["t"].to_numpy()
        if name == "$minute":
            return MinuteExpr(code, date, t, t.astype(np.float32))
        fld = name[1:]
        if fld == "amount":
            fld = "total_turnover"
        if fld not in df.columns:
            raise ExprError(f"分钟字段 {name} 不可用(本批数据未加载)")
        return MinuteExpr(code, date, t, df[fld].to_numpy(dtype=np.float32))

    def aggregate(self, name, evals):
        """对一个批次聚合节点参数求值并聚合 -> 批内日频宽表(列=批内股票代码)"""
        ser = _agg_series(name, evals[0], evals[1:])
        return _ser_to_wide(ser, self.daily_index, self.batch_codes)

    def compute_all_aggs(self, agg_nodes, types, data, cfg):
        """单趟遍历全部批次, 一次计算所有(最大)聚合节点的全市场日频宽表。
        返回 {canonical(node): 宽表}。相比逐节点遍历, 每个批次只加载一次分钟数据,
        一次算出全部聚合节点(大幅降低重复读取)。"""
        t0 = time.time()
        keys, seen = [], set()
        for node in agg_nodes:
            k = canonical(node)
            if k not in seen:
                seen.add(k)
                keys.append(k)
        parts = {k: [] for k in keys}
        n_batch = 0
        for batch_codes in self.batches:
            df = self._load_batch(batch_codes)
            if df is None:
                continue
            self.batch_codes = batch_codes
            try:
                runner = _BatchRunner(self, data, cfg, types)
                runner.set_batch(df, batch_codes)
                done = set()   # 同规范字符串的聚合节点(公式中重复出现)只算一次, 避免宽表列重复
                for node in agg_nodes:
                    k = canonical(node)
                    if k in done:
                        continue
                    done.add(k)
                    evals = [runner.eval_arg(a) for a in node.args]
                    parts[k].append(self.aggregate(node.name, evals))
            finally:
                del df
            n_batch += 1
        if not n_batch:
            raise ExprError("分钟数据为空, 无法计算聚合节点")
        cache = {}
        for k in keys:
            cache[k] = pd.concat(parts[k], axis=1).reindex(
                index=self.daily_index, columns=self.all_codes)
        console.log(f"    [分钟聚合] {len(keys)} 个聚合节点单趟遍历完成, "
                    f"{n_batch} 个批次, 耗时 {time.time()-t0:.1f} 秒")
        return cache

    def compute_all_aggs_full_market(self, agg_nodes, types, data, cfg):
        """全市场分钟截面模式: 按日期分块, 每块加载全部股票的分钟数据, 一次计算所有聚合节点。
        块内包含全部股票, 因此分钟级截面算子(RANK/ZSCORE/SCALE/CS_*)在块内即等价于全市场截面。
        每块内存 ≈ 块天数 × 240 × 全部股票数, 用 config.minute_chunk_days 控制(默认100天)。"""
        t0 = time.time()
        chunk_days = int(getattr(cfg, "minute_chunk_days", 100))
        dates = pd.Series(self.daily_index)
        chunks = [dates.iloc[i:i + chunk_days] for i in range(0, len(dates), chunk_days)]
        keys, seen = [], set()
        for node in agg_nodes:
            k = canonical(node)
            if k not in seen:
                seen.add(k)
                keys.append(k)
        parts = {k: [] for k in keys}
        all_codes = self.codes_with_minute
        n_chunk = 0
        for chunk in chunks:
            s_int = int(str(chunk.iloc[0].date()).replace("-", ""))
            e_int = int(str(chunk.iloc[-1].date()).replace("-", ""))
            df = self._load_batch(all_codes, s_int, e_int)
            if df is None:
                continue
            self.batch_codes = all_codes
            try:
                runner = _BatchRunner(self, data, cfg, types)
                runner.set_batch(df, all_codes)
                done = set()   # 同规范字符串的聚合节点只算一次, 避免宽表列重复
                for node in agg_nodes:
                    k = canonical(node)
                    if k in done:
                        continue
                    done.add(k)
                    evals = [runner.eval_arg(a) for a in node.args]
                    wide = self.aggregate(node.name, evals)
                    parts[k].append(wide.reindex(chunk.tolist()))
            finally:
                del df
            n_chunk += 1
        if not n_chunk:
            raise ExprError("分钟数据为空, 无法计算聚合节点")
        cache = {}
        for k in keys:
            cache[k] = pd.concat(parts[k], axis=0).reindex(
                index=self.daily_index, columns=self.all_codes)
        console.log(f"    [分钟聚合·全市场] {len(keys)} 个聚合节点(含截面算子)按日期分块完成, "
                    f"{n_chunk} 个分块, 耗时 {time.time()-t0:.1f} 秒")
        return cache


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


# =============================================================================
# 对外入口
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


def _finalize(out: pd.DataFrame):
    """与日频引擎一致的收尾检查"""
    out = out.astype("float64").replace([np.inf, -np.inf], np.nan)
    valid_ratio = float(out.notna().mean().mean())
    if valid_ratio < 0.3:
        raise ExprError(f"因子有效值比例过低({valid_ratio:.1%}), 公式可能无意义")
    cross_std = out.std(axis=1)
    if (cross_std.dropna() > 1e-12).mean() < 0.5:
        raise ExprError("因子横截面几乎无变化(所有股票取值相同), 无法分组")
    return out


def compute_factor_minute(expr: str, data, cfg) -> pd.DataFrame:
    """
    分钟模式入口: 校验并计算因子, 返回日频宽表(行=日期, 列=股票代码)。
    """
    from .expr_engine import Evaluator
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


# ======================================================================
# numba 稠密归约算子(可选加速): 融合循环、零中间3D数组、多线程。
# 与 numpy 版算法完全一致(float32 输入 + float64 累加, NaN 按无效处理),
# 输出 [日, 股] float64(由调用方转 float32)。数值一致性已实测到 1e-7。
# 注意: 不能用 fastmath=True(LLVM 假设无 NaN 会把 np.isnan 判断优化掉)。
# ======================================================================

@njit(parallel=True, cache=True)
def _nb_skew(x):
    """SKEW: Fisher-Pearson 修正偏度(与 pandas .skew() 一致)"""
    D_, M_, S_ = x.shape
    out = np.empty((D_, S_), dtype=np.float64)
    for d in prange(D_):
        n = np.zeros(S_, dtype=np.int64)
        s1 = np.zeros(S_)
        for i in range(M_):
            row = x[d, i]
            for s in range(S_):
                v = row[s]
                if not np.isnan(v):
                    n[s] += 1
                    s1[s] += v
        d2s = np.zeros(S_)
        d3s = np.zeros(S_)
        for i in range(M_):
            row = x[d, i]
            for s in range(S_):
                v = row[s]
                if not np.isnan(v):
                    dv = v - s1[s] / n[s]
                    d2s[s] += dv * dv
                    d3s[s] += dv * dv * dv
        for s in range(S_):
            n_ = n[s]
            if n_ >= 3 and d2s[s] > 0:
                out[d, s] = n_ * np.sqrt(n_ - 1) * d3s[s] / ((n_ - 2) * d2s[s] ** 1.5)
            else:
                out[d, s] = np.nan
    return out


@njit(parallel=True, cache=True)
def _nb_kurt(x):
    """KURT: 超额峰度(与 pandas .kurt() 一致)"""
    D_, M_, S_ = x.shape
    out = np.empty((D_, S_), dtype=np.float64)
    for d in prange(D_):
        n = np.zeros(S_, dtype=np.int64)
        s1 = np.zeros(S_)
        for i in range(M_):
            row = x[d, i]
            for s in range(S_):
                v = row[s]
                if not np.isnan(v):
                    n[s] += 1
                    s1[s] += v
        d2s = np.zeros(S_)
        d4s = np.zeros(S_)
        for i in range(M_):
            row = x[d, i]
            for s in range(S_):
                v = row[s]
                if not np.isnan(v):
                    dv = v - s1[s] / n[s]
                    d2s[s] += dv * dv
                    d4s[s] += dv * dv * dv * dv
        for s in range(S_):
            n_ = n[s]
            mu2 = d2s[s] / n_
            if n_ >= 4 and mu2 > 0:
                mu4 = d4s[s] / n_
                out[d, s] = (n_ - 1.0) / ((n_ - 2.0) * (n_ - 3.0)) * (
                    (n_ + 1.0) * (mu4 / mu2 ** 2.0) - 3.0 * (n_ - 1.0))
            else:
                out[d, s] = np.nan
    return out


@njit(parallel=True, cache=True)
def _nb_std(x):
    """STD: 样本标准差(与 pandas .std() 一致, ddof=1)"""
    D_, M_, S_ = x.shape
    out = np.empty((D_, S_), dtype=np.float64)
    for d in prange(D_):
        n = np.zeros(S_, dtype=np.int64)
        s1 = np.zeros(S_)
        for i in range(M_):
            row = x[d, i]
            for s in range(S_):
                v = row[s]
                if not np.isnan(v):
                    n[s] += 1
                    s1[s] += v
        ss = np.zeros(S_)
        for i in range(M_):
            row = x[d, i]
            for s in range(S_):
                v = row[s]
                if not np.isnan(v):
                    dv = v - s1[s] / n[s]
                    ss[s] += dv * dv
        for s in range(S_):
            n_ = n[s]
            if n_ >= 2:
                out[d, s] = np.sqrt(ss[s] / (n_ - 1))
            else:
                out[d, s] = np.nan
    return out


@njit(parallel=True, cache=True)
def _nb_regression(v, y, want):
    """REGRESSION_SLOPE/INTERCEPT: 一元回归 y~v, want=0 斜率, want=1 截距;
    y 为常数时方差为0 -> NaN(与 pandas 一致)"""
    D_, M_, S_ = v.shape
    slope = np.empty((D_, S_), dtype=np.float64)
    inter = np.empty((D_, S_), dtype=np.float64)
    for d in prange(D_):
        n = np.zeros(S_, dtype=np.int64)
        sv = np.zeros(S_)
        sy = np.zeros(S_)
        for i in range(M_):
            rv = v[d, i]
            ry = y[d, i]
            for s in range(S_):
                a = rv[s]
                b = ry[s]
                if not np.isnan(a) and not np.isnan(b):
                    n[s] += 1
                    sv[s] += a
                    sy[s] += b
        vary = np.zeros(S_)
        cov = np.zeros(S_)
        for i in range(M_):
            rv = v[d, i]
            ry = y[d, i]
            for s in range(S_):
                a = rv[s]
                b = ry[s]
                if not np.isnan(a) and not np.isnan(b):
                    da = a - sv[s] / n[s]
                    db = b - sy[s] / n[s]
                    vary[s] += db * db
                    cov[s] += da * db
        for s in range(S_):
            n_ = n[s]
            if n_ >= 2 and vary[s] > 0:
                sl = cov[s] / vary[s]
                slope[d, s] = sl
                inter[d, s] = sv[s] / n_ - sl * (sy[s] / n_)
            else:
                slope[d, s] = np.nan
                inter[d, s] = np.nan
    return slope if want == 0 else inter


@njit(parallel=True, cache=True)
def _nb_corr(x, y):
    """CORR: 皮尔逊相关(两趟均值中心化; 常数序列方差为0 -> NaN)"""
    D_, M_, S_ = x.shape
    out = np.empty((D_, S_), dtype=np.float64)
    for d in prange(D_):
        n = np.zeros(S_, dtype=np.int64)
        sx = np.zeros(S_)
        sy = np.zeros(S_)
        for i in range(M_):
            rx = x[d, i]
            ry = y[d, i]
            for s in range(S_):
                a = rx[s]
                b = ry[s]
                if not np.isnan(a) and not np.isnan(b):
                    n[s] += 1
                    sx[s] += a
                    sy[s] += b
        varx = np.zeros(S_)
        vary = np.zeros(S_)
        cov = np.zeros(S_)
        for i in range(M_):
            rx = x[d, i]
            ry = y[d, i]
            for s in range(S_):
                a = rx[s]
                b = ry[s]
                if not np.isnan(a) and not np.isnan(b):
                    da = a - sx[s] / n[s]
                    db = b - sy[s] / n[s]
                    varx[s] += da * da
                    vary[s] += db * db
                    cov[s] += da * db
        for s in range(S_):
            n_ = n[s]
            if n_ >= 2 and varx[s] > 0 and vary[s] > 0:
                out[d, s] = cov[s] / np.sqrt(varx[s] * vary[s])
            else:
                out[d, s] = np.nan
    return out


@njit(parallel=True, cache=True)
def _nb_autocorr(x, lag):
    """TS_AUTOCORR: 按'当日有效分钟序列'滞后 lag 的自相关
    (跨停牌分钟跳号, 与长表 groupby.shift(lag) 后 corr 一致)"""
    D_, M_, S_ = x.shape
    out = np.empty((D_, S_), dtype=np.float64)
    for d in prange(D_):
        buf = np.empty(M_, dtype=np.float64)
        for s in range(S_):
            k = 0
            for i in range(M_):
                v = x[d, i, s]
                if not np.isnan(v):
                    buf[k] = v
                    k += 1
            if k < lag + 2:
                out[d, s] = np.nan
                continue
            n_ = k - lag
            s1 = 0.0
            s2 = 0.0
            for i in range(lag, k):
                s1 += buf[i]
                s2 += buf[i - lag]
            mu1 = s1 / n_
            mu2 = s2 / n_
            vv1 = 0.0
            vv2 = 0.0
            cov = 0.0
            for i in range(lag, k):
                a = buf[i] - mu1
                b = buf[i - lag] - mu2
                vv1 += a * a
                vv2 += b * b
                cov += a * b
            if vv1 > 0 and vv2 > 0:
                out[d, s] = cov / np.sqrt(vv1 * vv2)
            else:
                out[d, s] = np.nan
    return out


@njit(parallel=True, cache=True)
def _nb_last(x):
    """LAST/FIRST: 取当日最后一个有值分钟的值(与长表一致)"""
    D_, M_, S_ = x.shape
    out = np.empty((D_, S_), dtype=np.float64)
    for d in prange(D_):
        for s in range(S_):
            val = np.nan
            for i in range(M_ - 1, -1, -1):
                v = x[d, i, s]
                if not np.isnan(v):
                    val = v
                    break
            out[d, s] = val
    return out


# =============================================================================
# 稠密加速路径: 分钟部分在 [日×240×股] 稠密矩阵上直接 numpy 归约(替代长表 groupby, 秒级)
# =============================================================================

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
        from .expr_engine import Num, Var, Bin, Unary, Ternary, Call
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
        from .expr_engine import Num, Str, Var, Bin, Unary, Ternary, Call, eval_call_daily
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
