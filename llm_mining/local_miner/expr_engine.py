"""
本地化因子挖掘项目 - 因子公式引擎(纯pandas实现, 脱离qlib)

流程: 公式字符串 -> 字符级括号深度扫描 -> 词法分析 -> 递归下降语法分析(AST)
      -> 校验(深度/函数签名/变量/复杂度) -> 递归求值(宽表: 日期×股票)

设计要点(对应需求3.2.2):
- 括号嵌套深度与语法树深度双重校验, 上限 max_depth 由最外层接口传入
- 运行前完成全部合法性检查, 非法公式在执行前被拦截, 不会导致主程序崩溃
- 所有计算基于 pandas 宽表(行=日期, 列=股票代码), 输出与因子评价格式对齐
"""

import re

import numpy as np
import pandas as pd

ALLOWED_VARS = {"$open", "$close", "$high", "$low", "$volume", "$amount", "$return"}

# 函数注册表: 名称 -> (最少参数, 最多参数, 窗口参数位置列表(0基), 浮点参数位置列表)
FUNC_SPEC = {
    "DELAY": (1, 2, [1], []),
    "DELTA": (1, 2, [1], []),
    "TS_MEAN": (1, 2, [1], []),
    "TS_SUM": (1, 2, [1], []),
    "TS_STD": (1, 2, [1], []),
    "TS_MAX": (1, 2, [1], []),
    "TS_MIN": (1, 2, [1], []),
    "TS_MEDIAN": (1, 2, [1], []),
    "TS_RANK": (1, 2, [1], []),
    "TS_ARGMAX": (1, 2, [1], []),
    "TS_ARGMIN": (1, 2, [1], []),
    "HIGHDAY": (1, 2, [1], []),
    "LOWDAY": (1, 2, [1], []),
    "TS_CORR": (2, 3, [2], []),
    "TS_COV": (2, 3, [2], []),
    "TS_ZSCORE": (1, 2, [1], []),
    "TS_QUANTILE": (2, 3, [1], [2]),
    "EMA": (2, 2, [1], []),
    "DECAYLINEAR": (2, 2, [1], []),
    "COUNT": (2, 2, [1], []),
    "PROD": (1, 2, [1], []),
    "RANK": (1, 1, [], []),
    "ZSCORE": (1, 1, [], []),
    "SCALE": (1, 1, [], []),
    "MEAN": (1, 1, [], []),
    "MAX": (1, 2, [], []),
    "MIN": (1, 2, [], []),
    "ABS": (1, 1, [], []),
    "SIGN": (1, 1, [], []),
    "EXP": (1, 1, [], []),
    "SQRT": (1, 1, [], []),
    "LOG": (1, 1, [], []),
    "INV": (1, 1, [], []),
    "POW": (2, 2, [], [1]),
    "WHERE": (3, 3, [], []),
}

# 窗口参数缺省值
WINDOW_DEFAULTS = {
    "DELAY": 1, "DELTA": 1, "TS_MEAN": 5, "TS_SUM": 5, "TS_STD": 20,
    "TS_MAX": 5, "TS_MIN": 5, "TS_MEDIAN": 5, "TS_RANK": 5,
    "TS_ARGMAX": 5, "TS_ARGMIN": 5, "HIGHDAY": 5, "LOWDAY": 5,
    "TS_CORR": 5, "TS_COV": 5, "TS_ZSCORE": 5, "TS_QUANTILE": 5,
    "COUNT": 20, "PROD": 5,
}

INVALID_CHAR_RE = re.compile(r"[^A-Za-z0-9_$+\-*/><=&|?:.,() \t\r\n]")

TOKEN_RE = re.compile(
    r"""
      (?P<NUM>\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?)
     |(?P<VAR>\$[A-Za-z_][A-Za-z0-9_]*)
     |(?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)
     |(?P<OP>>=|<=|==|!=|&&|\|\||[+\-*/><&|?:(),])
     |(?P<WS>\s+)
    """,
    re.VERBOSE,
)


class ExprError(Exception):
    """公式非法(在执行前被拦截)"""
    pass


# =============================================================================
# AST 节点
# =============================================================================

class Num:
    __slots__ = ("value",)
    def __init__(self, value): self.value = value

class Var:
    __slots__ = ("name",)
    def __init__(self, name): self.name = name

class Call:
    __slots__ = ("name", "args")
    def __init__(self, name, args): self.name = name; self.args = args

class Bin:
    __slots__ = ("op", "left", "right")
    def __init__(self, op, left, right): self.op = op; self.left = left; self.right = right

class Unary:
    __slots__ = ("op", "x")
    def __init__(self, op, x): self.op = op; self.x = x

class Ternary:
    __slots__ = ("cond", "true", "false")
    def __init__(self, cond, true, false): self.cond = cond; self.true = true; self.false = false


# =============================================================================
# 预检: 字符级括号深度扫描(快速拦截超深度/不匹配公式)
# =============================================================================

def bracket_scan(expr: str) -> int:
    """扫描括号匹配关系, 返回最大嵌套深度; 不匹配则抛 ExprError"""
    depth = 0
    max_depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1
            if depth > max_depth:
                max_depth = depth
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise ExprError("括号不匹配: 存在多余的右括号 ')'")
    if depth != 0:
        raise ExprError(f"括号不匹配: 缺少 {depth} 个右括号 ')'")
    return max_depth


# =============================================================================
# 词法分析
# =============================================================================

def tokenize(expr: str) -> list:
    tokens = []
    pos = 0
    n = len(expr)
    while pos < n:
        m = TOKEN_RE.match(expr, pos)
        if m is None:
            raise ExprError(f"公式含非法字符: '{expr[pos]}' (位置{pos})")
        kind = m.lastgroup
        value = m.group()
        pos = m.end()
        if kind == "WS":
            continue
        tokens.append((kind, value))
    return tokens


# =============================================================================
# 递归下降语法分析
# =============================================================================

class Parser:
    def __init__(self, tokens):
        self.toks = tokens
        self.i = 0

    def peek(self):
        if self.i < len(self.toks):
            return self.toks[self.i]
        return (None, None)

    def next(self):
        tok = self.peek()
        self.i += 1
        return tok

    def expect(self, value):
        kind, val = self.next()
        if val != value:
            raise ExprError(f"语法错误: 期望 '{value}', 实际得到 '{val}'")

    def parse(self):
        if not self.toks:
            raise ExprError("公式为空")
        node = self.parse_ternary()
        if self.i < len(self.toks):
            raise ExprError(f"语法错误: 公式末尾有多余内容 '{self.peek()[1]}'")
        return node

    def parse_ternary(self):
        cond = self.parse_or()
        if self.peek()[1] == "?":
            self.next()
            true = self.parse_ternary()
            self.expect(":")
            false = self.parse_ternary()
            return Ternary(cond, true, false)
        return cond

    def parse_or(self):
        left = self.parse_and()
        while self.peek()[1] in ("||", "|"):
            self.next()
            right = self.parse_and()
            left = Bin("||", left, right)
        return left

    def parse_and(self):
        left = self.parse_compare()
        while self.peek()[1] in ("&&", "&"):
            self.next()
            right = self.parse_compare()
            left = Bin("&&", left, right)
        return left

    def parse_compare(self):
        left = self.parse_add()
        while self.peek()[1] in (">", "<", ">=", "<=", "==", "!="):
            op = self.next()[1]
            right = self.parse_add()
            left = Bin(op, left, right)
        return left

    def parse_add(self):
        left = self.parse_mul()
        while self.peek()[1] in ("+", "-"):
            op = self.next()[1]
            right = self.parse_mul()
            left = Bin(op, left, right)
        return left

    def parse_mul(self):
        left = self.parse_unary()
        while self.peek()[1] in ("*", "/"):
            op = self.next()[1]
            right = self.parse_unary()
            left = Bin(op, left, right)
        return left

    def parse_unary(self):
        if self.peek()[1] == "-":
            self.next()
            return Unary("-", self.parse_unary())
        if self.peek()[1] == "+":
            self.next()
            return self.parse_unary()
        return self.parse_primary()

    def parse_primary(self):
        kind, value = self.next()
        if kind == "NUM":
            return Num(float(value))
        if kind == "VAR":
            return Var(value.lower())
        if kind == "IDENT":
            name = value.upper()
            if name in ("TRUE", "FALSE"):
                return Num(1.0 if name == "TRUE" else 0.0)
            if self.peek()[1] != "(":
                raise ExprError(f"标识符 '{value}' 不是合法变量(变量需以$开头), 或函数调用缺少括号")
            self.next()  # 吃掉 '('
            args = []
            if self.peek()[1] != ")":
                args.append(self.parse_ternary())
                while self.peek()[1] == ",":
                    self.next()
                    args.append(self.parse_ternary())
            self.expect(")")
            return Call(name, args)
        if value == "(":
            node = self.parse_ternary()
            self.expect(")")
            return node
        raise ExprError(f"语法错误: 意外的符号 '{value}'")


# =============================================================================
# 校验
# =============================================================================

def ast_depth(node) -> int:
    if isinstance(node, (Num, Var)):
        return 1
    if isinstance(node, Unary):
        return 1 + ast_depth(node.x)
    if isinstance(node, Bin):
        return 1 + max(ast_depth(node.left), ast_depth(node.right))
    if isinstance(node, Ternary):
        return 1 + max(ast_depth(node.cond), ast_depth(node.true), ast_depth(node.false))
    if isinstance(node, Call):
        if not node.args:
            return 1
        return 1 + max(ast_depth(a) for a in node.args)
    raise ExprError("未知AST节点")


def _walk_validate(node, cfg, vars_used: set):
    if isinstance(node, Num):
        return
    if isinstance(node, Var):
        if node.name not in ALLOWED_VARS:
            raise ExprError(f"使用了未声明的变量: {node.name}, 可用变量: {sorted(ALLOWED_VARS)}")
        vars_used.add(node.name)
        return
    if isinstance(node, Unary):
        _walk_validate(node.x, cfg, vars_used)
        return
    if isinstance(node, Bin):
        _walk_validate(node.left, cfg, vars_used)
        _walk_validate(node.right, cfg, vars_used)
        return
    if isinstance(node, Ternary):
        _walk_validate(node.cond, cfg, vars_used)
        _walk_validate(node.true, cfg, vars_used)
        _walk_validate(node.false, cfg, vars_used)
        return
    if isinstance(node, Call):
        if node.name not in FUNC_SPEC:
            raise ExprError(f"使用了未声明的函数: {node.name}")
        min_a, max_a, win_idx, float_idx = FUNC_SPEC[node.name]
        if not (min_a <= len(node.args) <= max_a):
            raise ExprError(f"函数 {node.name} 参数个数应为 {min_a}~{max_a}, 实际 {len(node.args)}")
        for idx, arg in enumerate(node.args):
            if idx in win_idx:
                if not isinstance(arg, Num):
                    raise ExprError(f"函数 {node.name} 的第{idx+1}个参数(时间窗口)必须是数字常量")
                n = arg.value
                if not float(n).is_integer() or int(n) < 1 or int(n) > cfg.max_window:
                    raise ExprError(f"函数 {node.name} 的时间窗口必须是 1~{cfg.max_window} 的整数, 实际 {n}")
            if idx in float_idx:
                if not isinstance(arg, Num) or not (0 < arg.value < 1):
                    raise ExprError(f"函数 {node.name} 的分位数参数必须是 0~1 的小数")
            _walk_validate(arg, cfg, vars_used)
        return
    raise ExprError("未知AST节点")


def validate(expr: str, cfg):
    """
    运行前完整合法性检查, 通过后返回 AST。任何非法公式都在此被拦截。
    """
    if not isinstance(expr, str) or not expr.strip():
        raise ExprError("公式为空")
    expr = expr.strip()

    # 1. 非法字符检查(含中文全角括号等)
    m = INVALID_CHAR_RE.search(expr)
    if m:
        raise ExprError(f"公式含非法字符: '{m.group()}'")

    # 2. 长度检查
    if len(expr) > cfg.max_symbol_length:
        raise ExprError(f"公式长度 {len(expr)} 超过上限 {cfg.max_symbol_length}")

    # 3. 字符级括号深度扫描
    bd = bracket_scan(expr)
    if bd > cfg.max_depth:
        raise ExprError(f"括号嵌套深度 {bd} 超过上限 {cfg.max_depth}")

    # 4. 语法分析
    ast = Parser(tokenize(expr)).parse()

    # 5. 语法树深度检查
    depth = ast_depth(ast)
    if depth > cfg.max_depth:
        raise ExprError(f"语法树深度 {depth} 超过上限 {cfg.max_depth}")

    # 6. 函数/变量/参数检查
    vars_used = set()
    _walk_validate(ast, cfg, vars_used)

    # 7. 基础变量个数检查
    if len(vars_used) > cfg.max_base_features:
        raise ExprError(f"基础变量个数 {len(vars_used)} 超过上限 {cfg.max_base_features}")

    return ast


# =============================================================================
# 求值辅助
# =============================================================================

def _safe_div(left, right):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = left / right
    if isinstance(out, pd.DataFrame):
        out = out.replace([np.inf, -np.inf], np.nan)
    elif isinstance(out, (float, np.floating)) and not np.isfinite(out):
        out = np.nan
    return out


def _wrap(arr, ref: pd.DataFrame) -> pd.DataFrame:
    """把 ndarray 结果包回与 ref 同形状的 DataFrame"""
    return pd.DataFrame(arr, index=ref.index, columns=ref.columns)


def _broadcast_row(series: pd.Series, ref: pd.DataFrame) -> pd.DataFrame:
    """把按日的截面值广播成与 ref 同形状的 DataFrame"""
    return pd.DataFrame(
        np.tile(series.values[:, None], (1, ref.shape[1])),
        index=ref.index, columns=ref.columns,
    )


def _argmax_last(window):
    valid = ~np.isnan(window)
    if not valid.any():
        return np.nan
    idx = np.where(valid)[0]
    best = idx[np.argmax(window[valid])]
    return float(len(window) - 1 - best)


def _argmin_last(window):
    valid = ~np.isnan(window)
    if not valid.any():
        return np.nan
    idx = np.where(valid)[0]
    best = idx[np.argmin(window[valid])]
    return float(len(window) - 1 - best)


def _prod_last(window):
    valid = window[~np.isnan(window)]
    if valid.size == 0:
        return np.nan
    return float(np.prod(valid))


# =============================================================================
# 递归求值
# =============================================================================

class Evaluator:
    def __init__(self, data):
        self.data = data

    def eval(self, node):
        if isinstance(node, Num):
            return node.value
        if isinstance(node, Var):
            return self.data.var(node.name)
        if isinstance(node, Unary):
            x = self.eval(node.x)
            return -x
        if isinstance(node, Bin):
            return self._eval_bin(node)
        if isinstance(node, Ternary):
            c = self.eval(node.cond)
            t = self.eval(node.true)
            f = self.eval(node.false)
            return self._where(c, t, f)
        if isinstance(node, Call):
            args = [self.eval(a) for a in node.args]
            return self._eval_call(node.name, args)
        raise ExprError("未知AST节点")

    # ---- 运算符 ----
    def _eval_bin(self, node):
        left = self.eval(node.left)
        right = self.eval(node.right)
        op = node.op
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if op == "+":
                return left + right
            if op == "-":
                return left - right
            if op == "*":
                return left * right
            if op == "/":
                return _safe_div(left, right)
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
                return _to_bool(left) & _to_bool(right)
            if op == "||":
                return _to_bool(left) | _to_bool(right)
        raise ExprError(f"未知运算符: {op}")

    def _where(self, cond, true, false):
        if not isinstance(cond, pd.DataFrame):
            raise ExprError("条件表达式的结果必须是序列, 不能是标量")
        c = cond.fillna(False).astype(bool).values
        t = true.values if isinstance(true, pd.DataFrame) else true
        f = false.values if isinstance(false, pd.DataFrame) else false
        return _wrap(np.where(c, t, f), cond)

    # ---- 函数 ----
    def _eval_call(self, name, args):
        def arg(i, default=None):
            if i < len(args):
                return args[i]
            if default is not None:
                return default
            raise ExprError(f"函数 {name} 缺少第{i+1}个参数")

        def win(i):
            if i < len(args):
                return int(args[i])
            return WINDOW_DEFAULTS.get(name, 5)

        # 时序函数
        if name == "DELAY":
            return arg(0).shift(win(1))
        if name == "DELTA":
            return arg(0).diff(win(1))
        if name in ("TS_MEAN", "TS_SUM", "TS_STD", "TS_MAX", "TS_MIN", "TS_MEDIAN"):
            x, n = arg(0), win(1)
            r = x.rolling(n, min_periods=1)
            return {"TS_MEAN": r.mean, "TS_SUM": r.sum, "TS_STD": r.std,
                    "TS_MAX": r.max, "TS_MIN": r.min, "TS_MEDIAN": r.median}[name]()
        if name == "TS_RANK":
            return arg(0).rolling(win(1), min_periods=1).rank(pct=True)
        if name in ("TS_ARGMAX", "HIGHDAY"):
            return arg(0).rolling(win(1), min_periods=1).apply(_argmax_last, raw=True)
        if name in ("TS_ARGMIN", "LOWDAY"):
            return arg(0).rolling(win(1), min_periods=1).apply(_argmin_last, raw=True)
        if name in ("TS_CORR", "TS_COV"):
            x, y, n = arg(0), arg(1), win(2)
            minp = max(2, n // 2)
            xm = x.rolling(n, min_periods=minp).mean()
            ym = y.rolling(n, min_periods=minp).mean()
            cov = (x * y).rolling(n, min_periods=minp).mean() - xm * ym
            if name == "TS_COV":
                return cov
            xs = x.rolling(n, min_periods=minp).std()
            ys = y.rolling(n, min_periods=minp).std()
            return _safe_div(cov, xs * ys)
        if name == "TS_ZSCORE":
            x, n = arg(0), win(1)
            xm = x.rolling(n, min_periods=1).mean()
            xs = x.rolling(n, min_periods=1).std()
            return _safe_div(x - xm, xs)
        if name == "TS_QUANTILE":
            x, n = arg(0), win(1)
            q = arg(2, 0.5)
            return x.rolling(n, min_periods=1).quantile(float(q))
        if name == "EMA":
            return arg(0).ewm(span=int(arg(1)), min_periods=1, adjust=False).mean()
        if name == "DECAYLINEAR":
            x, n = arg(0), win(1)
            weights = np.arange(1, n + 1, dtype=float)
            def _decay(w):
                valid = ~np.isnan(w)
                if not valid.any():
                    return np.nan
                # 初始窗口长度可能小于 n, 取权重尾部对齐(最新值权重最大)
                wts = weights[n - len(w):]
                return float(np.dot(np.where(valid, w, 0.0), wts) / wts[valid].sum())
            return x.rolling(n, min_periods=1).apply(_decay, raw=True)
        if name == "COUNT":
            cond, n = arg(0), win(1)
            return _to_bool(cond).astype(float).rolling(n, min_periods=1).sum()
        if name == "PROD":
            return arg(0).rolling(win(1), min_periods=1).apply(_prod_last, raw=True)

        # 截面函数
        if name == "RANK":
            return arg(0).rank(axis=1, pct=True)
        if name == "ZSCORE":
            x = arg(0)
            mu = x.mean(axis=1)
            sd = x.std(axis=1).replace(0, np.nan)
            return x.sub(mu, axis=0).div(sd, axis=0)
        if name == "SCALE":
            x = arg(0)
            s = x.abs().sum(axis=1).replace(0, np.nan)
            return x.div(s, axis=0)
        if name == "MEAN":
            x = arg(0)
            return _broadcast_row(x.mean(axis=1), x)
        if name in ("MAX", "MIN"):
            if len(args) == 1:
                x = args[0]
                row = x.max(axis=1) if name == "MAX" else x.min(axis=1)
                return _broadcast_row(row, x)
            with np.errstate(invalid="ignore"):
                return np.maximum(args[0], args[1]) if name == "MAX" else np.minimum(args[0], args[1])

        # 元素级数学函数
        if name == "ABS":
            return arg(0).abs() if isinstance(arg(0), pd.DataFrame) else abs(arg(0))
        if name == "SIGN":
            return np.sign(arg(0))
        if name == "EXP":
            x = arg(0)
            return np.exp(x.clip(-700, 700) if isinstance(x, pd.DataFrame) else np.clip(x, -700, 700))
        if name == "SQRT":
            x = arg(0)
            return np.sqrt(x.clip(lower=0) if isinstance(x, pd.DataFrame) else max(x, 0))
        if name == "LOG":
            x = arg(0)
            xp = (x + 1)
            return np.log(xp.clip(lower=1e-8) if isinstance(xp, pd.DataFrame) else max(xp, 1e-8))
        if name == "INV":
            return _safe_div(1.0, arg(0))
        if name == "POW":
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                return np.power(arg(0), float(arg(1)))
        if name == "WHERE":
            return self._where(args[0], args[1], args[2])

        raise ExprError(f"函数 {name} 未实现")


def _to_bool(x):
    if isinstance(x, pd.DataFrame):
        return x.fillna(False).astype(bool)
    return bool(x)


# =============================================================================
# 对外主入口
# =============================================================================

def compute_factor(expr: str, data, cfg) -> pd.DataFrame:
    """
    校验并计算因子, 返回宽表 DataFrame(行=日期, 列=股票代码)。
    非法公式抛出 ExprError(在运行前被拦截)。
    """
    ast = validate(expr, cfg)
    result = Evaluator(data).eval(ast)
    if not isinstance(result, pd.DataFrame):
        raise ExprError("公式计算结果为标量, 不是有效的截面因子")

    out = result.astype("float64").replace([np.inf, -np.inf], np.nan)

    # 有效性检查
    valid_ratio = float(out.notna().mean().mean())
    if valid_ratio < 0.3:
        raise ExprError(f"因子有效值比例过低({valid_ratio:.1%}), 公式可能无意义")
    cross_std = out.std(axis=1)
    if (cross_std.dropna() > 1e-12).mean() < 0.5:
        raise ExprError("因子横截面几乎无变化(所有股票取值相同), 无法分组")
    return out
