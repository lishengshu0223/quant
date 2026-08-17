"""
本地化因子挖掘项目 - 分钟频率公式解析/类型推断/校验（minute_engine 拆分模块①）

职责:
- 分钟模式全部规格常量(聚合/维持维度/截面/日内滚动算子规格、变量集合、常量校验规则)
- 公式字符串 -> AST 的校验(minute_validate): 长度/非法字符/括号深度/语法树深度/最外层必须日频
- 节点类型推断(_infer): 'W'日频宽表 | 'M'分钟序列 | 'S'标量常量
- 聚合节点收集(_walk_collect_agg) 与 分钟字段收集(minute_fields_used)

本模块只依赖 expr_engine(日频引擎的 AST 类型/解析器/函数规格), 不依赖本包其它分钟模块。
"""

import re

from ..expr_engine import (
    ALLOWED_VARS, FUNC_SPEC, Num, Str, Var, Call, Bin, Unary, Ternary, Parser,
    ExprError, ast_depth, bracket_scan, tokenize,
)

# 分钟模式可用变量(在分钟算子的参数中解析为分钟序列)
MINUTE_FIELDS = {"$open", "$close", "$high", "$low", "$volume", "$amount", "$turnover",
                 "$return", "$minute"}

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
