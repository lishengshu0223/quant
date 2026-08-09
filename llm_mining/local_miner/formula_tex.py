"""
本地化因子挖掘项目 - 因子公式 LaTeX 渲染转换器

把挖掘 DSL 公式(如 TS_MEAN(SKEW($return), 5))转换成数学公式样式的 LaTeX 字符串,
供两处使用(同一转换器, 保证一致):
- matplotlib mathtext: 评价图标题/副标题显示 (fig.text(..., f"${tex}$"))
- 静态 HTML 报告: KaTeX 渲染 ($$tex$$)

渲染规则:
- 价格/量变量映射为斜体字母下标: $close->C_t, $volume->V_t, $return->r_t ...
- 算子名保留原样但用正体 \mathrm{} 显示, 时间窗口参数(纯数字)提升为下标:
  TS_MEAN(x, 5) -> \mathrm{TS\_MEAN}_{5}(x)
- 乘除/比较/逻辑等二元运算映射为数学符号(\times, \frac, \geq, \land ...)
"""

from . import expr_engine

# 变量 -> LaTeX 显示
VAR_TEX = {
    "$open": "O_t", "$close": "C_t", "$high": "H_t", "$low": "L_t",
    "$volume": "V_t", "$amount": "A_t", "$return": "r_t", "$minute": "m_t",
}

# 二元运算 -> LaTeX
BIN_TEX = {
    "+": "+", "-": "-", "*": r"\times", "/": r"\frac",
    "^": "^", ">": ">", "<": "<", ">=": r"\geq", "<=": r"\leq",
    "==": "=", "!=": r"\neq", "&&": r"\wedge", "||": r"\vee",
}

# 常用算子中英文对照(仅用于 HTML 报告中的小字注释, mathtext 中只显示算子名)
OP_CN = {
    "TS_MEAN": "时序均值", "TS_STD": "时序标准差", "TS_SKEW": "时序偏度",
    "TS_KURT": "时序峰度", "TS_MAX": "时序最大", "TS_MIN": "时序最小",
    "TS_MEDIAN": "时序中位数", "TS_SUM": "时序求和", "TS_QUANTILE": "时序分位",
    "TS_RANK": "时序排名", "TS_CORR": "时序相关", "TS_AUTOCORR": "时序自相关",
    "REGRESSION_SLOPE": "回归斜率", "REGRESSION_INTERCEPT": "回归截距",
    "SKEW": "偏度", "KURT": "峰度", "CORR": "相关", "MEAN": "均值", "STD": "标准差",
    "MAX": "最大值", "MIN": "最小值", "MEDIAN": "中位数", "SUM": "求和",
    "LAST": "末值", "FIRST": "首值", "COUNT": "计数", "QUANTILE": "分位数",
    "RANK": "截面排名", "ZSCORE": "截面标准化", "SCALE": "截面归一化",
    "INTRADAY_MEAN": "日内均值", "INTRADAY_STD": "日内标准差",
    "INTRADAY_SUM": "日内求和", "INTRADAY_MAX": "日内最大", "INTRADAY_MIN": "日内最小",
    "INTRADAY_MEDIAN": "日内中位", "SLICE": "切片", "MASK": "掩码",
    "ABS": "绝对值", "IF": "条件", "DELTA": "差分", "PCT_CHANGE": "变化率",
}


def _is_num(node) -> bool:
    return isinstance(node, expr_engine.Num)


def _num_tex(node) -> str:
    v = node.value
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _op_tex(name: str) -> str:
    """算子名 -> \mathrm{NAME}(下划线转义)"""
    esc = name.replace("_", r"\_")
    return r"\mathrm{" + esc + "}"


def _call_tex(node) -> str:
    """Call 节点渲染: 纯数字参数(时间窗口)提升为下标, 其余留在括号内"""
    name = node.name
    args = node.args
    sub = ""
    rest = []
    for a in args:
        if _is_num(a):
            sub += (_num_tex(a) + ",")
        else:
            rest.append(a)
    if sub:
        sub = "_{" + sub.rstrip(",") + "}"
    inner = ",".join(_node_tex(a) for a in rest)
    return f"{_op_tex(name)}{sub}\\left({inner}\\right)"


def _node_tex(node) -> str:
    if isinstance(node, expr_engine.Num):
        return _num_tex(node)
    if isinstance(node, expr_engine.Str):
        return r"\mathrm{" + node.value + "}"
    if isinstance(node, expr_engine.Var):
        return VAR_TEX.get(node.name, r"\mathrm{" + node.name.replace("_", r"\_") + "}")
    if isinstance(node, expr_engine.Call):
        return _call_tex(node)
    if isinstance(node, expr_engine.Bin):
        a, op, b = _node_tex(node.left), node.op, _node_tex(node.right)
        if op == "/":
            # 除号 -> \frac{}{}, 分子分母都是简单叶子时才用(避免嵌套分数爆炸)
            simple = (isinstance(node.left, (expr_engine.Num, expr_engine.Var))
                      and isinstance(node.right, (expr_engine.Num, expr_engine.Var)))
            return r"\frac{" + a + "}{" + b + "}" if simple else a + r"\,/\, " + b
        if op == "*":
            op = r"\cdot"
        if op == "^":
            return a + "^{" + b + "}"
        sym = BIN_TEX.get(op, op)
        return a + "\\," + sym + "\\, " + b
    if isinstance(node, expr_engine.Unary):
        if node.op == "-":
            return "-" + _node_tex(node.x)
        return _node_tex(node.x)
    if isinstance(node, expr_engine.Ternary):
        return (f"({_node_tex(node.cond)}\\,?\\,{_node_tex(node.true)}"
                f"\\,:\\,{_node_tex(node.false)})")
    return "?"


def to_tex(expr: str) -> str:
    """DSL 公式 -> LaTeX 字符串(matplotlib mathtext 与 KaTeX 通用子集)"""
    try:
        ast = expr_engine.Parser(expr_engine.tokenize(expr)).parse()
    except Exception:
        return r"\mathrm{" + expr.replace("_", r"\_") + "}"
    return _node_tex(ast)


def to_mathtext(expr: str) -> str:
    """matplotlib mathtext 专用: 包一层 $...$"""
    return "$" + to_tex(expr) + "$"


def op_cn(expr: str) -> str:
    """公式中出现的中文算子注释(HTML 报告小字用)"""
    try:
        ast = expr_engine.Parser(expr_engine.tokenize(expr)).parse()
    except Exception:
        return ""
    seen = []

    def walk(n):
        if isinstance(n, expr_engine.Call):
            cn = OP_CN.get(n.name)
            if cn and cn not in seen:
                seen.append(cn)
            for a in n.args:
                walk(a)
        elif isinstance(n, expr_engine.Bin):
            walk(n.left); walk(n.right)
        elif isinstance(n, expr_engine.Unary):
            walk(n.x)
        elif isinstance(n, expr_engine.Ternary):
            walk(n.cond); walk(n.true); walk(n.false)
    walk(ast)
    return " · ".join(seen)
