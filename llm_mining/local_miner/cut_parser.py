"""
本地化因子挖掘项目 - 因子切割论(CTOP/CBOT)公式解析与校验

切割论思想(用户定义, 统一日频/分钟):
  1. 切割工具(tool): 决定"按什么排序切"。预定义: $amp(bar内振幅) / $bret(bar间收益率) / $volume / $amount;
     也允许任意元素级表达式(如 ABS($bret)、SIGN($bret)、($close-$open)/$open)。
  2. 切割方式: 在滚动窗口内按 tool 值排序, 取前(CTOP)/后(CBOT) N% 或 N 个 bar。
  3. 切割后聚合: 对被选中 bar 的 target(目标字段, 也是元素级表达式)做统计聚合
     MEAN/STD/SKEW/KURT/MEDIAN/SUM/MAX/MIN/COUNT/LAST。
  4. 聚合后计算: 切割聚合结果即"日频高阶特征", 可像基础变量一样与日频算子任意组合
     (加减乘除、TS_MEAN/TS_CORR、截面变换等)。

公式语法(模型输出, 字符串可解析):
  CTOP("tool", ratio, "AGG", "target", window)   # 窗口内取 tool 值最大的前 ratio 比例(或前 N 个) bar
  CBOT("tool", ratio, "AGG", "target", window)   # 取 tool 值最小的后 ratio 比例(或后 N 个)
  说明:
    - tool/target 必须用引号包裹的元素级表达式; 内部禁止嵌套 CTOP/CBOT 与时间窗口/聚合算子。
    - ratio: 0<比例≤1(如0.2=前20%), 或正整数N(前N个bar)。
    - AGG: 聚合方式(引号包裹)。
    - window: 滚动窗口 bar 数。日频=交易日数; 分钟=bar数(240=日内1天, 240×D=滚动D个交易日)。
  示例:
    CTOP("$amp", 0.2, "MEAN", "$return", 20)             # 过去20日振幅最大的20%的bar的日均收益
    CBOT("ABS($bret)", 10, "MEAN", "$return", 60) - CTOP("ABS($bret)", 10, "MEAN", "$return", 60)
    TS_CORR(CTOP("$volume", 0.3, "MEAN", "$bret", 40), $amount, 10)
"""

from .expr_engine import (
    ExprError, Parser, tokenize, bracket_scan, ast_depth, is_outer_rank,
    INVALID_CHAR_RE, Num, Str, Var, Call, Bin, Unary, Ternary,
)

# 切割算子(统一日频/分钟)
CUT_FUNCS = {"CTOP", "CBOT"}

# 预定义切割工具 -> 展开为元素级表达式(求值前字符串替换)
CUT_TOOLS = {
    "$amp": "($high - $low) / ($high + $low + 1e-12) / 2",  # bar内振幅(归一化; 日频=日内高低, 分钟=该分钟高低)
    "$bret": "$return",                                     # bar间收益率(两个close的pct_change, 可取ABS/SIGN)
    "$volume": "$volume",                                   # 成交量
    "$amount": "$amount",                                   # 成交额
    "$turnover": "$turnover",                               # 换手率(%); 日频=当日换手率, 分钟=该分钟成交量/当日流通股本
}                                                           # (均由本地股本数据派生)

# 切割后聚合方式
CUT_AGGS = ("MEAN", "STD", "SKEW", "KURT", "MEDIAN", "SUM", "MAX", "MIN", "COUNT", "LAST")

# tool/target 子表达式(分钟模式)允许的元素级函数
ELEMENT_FUNCS = {"ABS", "SIGN", "EXP", "SQRT", "LOG", "INV", "POW", "WHERE"}


def expand_cut_tools(s: str) -> str:
    """把 tool/target 字符串中的切割工具宏($amp/$bret)展开为元素级表达式"""
    out = s.replace("$amp", CUT_TOOLS["$amp"]).replace("$bret", CUT_TOOLS["$bret"])
    return out


def cut_canonical(node) -> str:
    """切割节点规范化字符串(去重缓存/展示用)"""
    return (f'{node.name}({node.args[0].value},{node.args[1].value},'
            f'{node.args[2].value},{node.args[3].value},{int(node.args[4].value)})')


def _parse_node_args(node) -> tuple:
    """从切割节点取 (tool_str, ratio, agg, target_str, window)"""
    return (node.args[0].value, float(node.args[1].value), node.args[2].value.upper(),
            node.args[3].value, int(node.args[4].value))


def validate_cut(expr: str, cfg):
    """
    切割模式公式完整合法性检查, 返回 (ast, cut_nodes)。
    cut_nodes: 去重后的 CTOP/CBOT 节点列表(任意顺序)。
    任何非法公式都在此被拦截(运行前)。
    """
    if not isinstance(expr, str) or not expr.strip():
        raise ExprError("公式为空")
    expr = expr.strip()

    # 1. 非法字符检查
    m = INVALID_CHAR_RE.search(expr)
    if m:
        raise ExprError(f"公式含非法字符: '{m.group()}'")

    # 2. 长度检查(切割模式放宽上限)
    if len(expr) > cfg.formula_max_symbol_length:
        raise ExprError(f"公式长度 {len(expr)} 超过上限 {cfg.formula_max_symbol_length}")

    # 3. 字符级括号深度扫描
    bd = bracket_scan(expr)
    if bd > cfg.formula_max_depth:
        raise ExprError(f"括号嵌套深度 {bd} 超过上限 {cfg.formula_max_depth}")

    # 4. 语法分析
    ast = Parser(tokenize(expr)).parse()

    # 5. 语法树深度检查
    depth = ast_depth(ast)
    if depth > cfg.formula_max_depth:
        raise ExprError(f"语法树深度 {depth} 超过上限 {cfg.formula_max_depth}")

    # 6. 最外层 RANK 禁止
    if is_outer_rank(ast):
        raise ExprError(
            "公式最外层禁止为截面RANK(含 -RANK(...)、RANK(...)*常数 等仿射变体): "
            "rank非线性变换会使后续中性化/标准化失效。RANK只可作为中间步骤(如 TS_MEAN(RANK(x), n))。"
        )

    # 7. 遍历收集切割节点 + 校验 + 统计基础变量
    cut_nodes = []
    vars_used = set()
    _walk_cut(ast, cfg, cut_nodes, vars_used)
    if len(vars_used) > cfg.formula_max_base_features:
        raise ExprError(
            f"基础变量个数 {len(vars_used)} 超过上限 {cfg.formula_max_base_features}"
            f"(含切割工具/目标内部的变量)")

    # 8. 切割节点去重(同规范字符串只算一次)
    seen, uniq = set(), []
    for n in cut_nodes:
        k = cut_canonical(n)
        if k not in seen:
            seen.add(k)
            uniq.append(n)
    return ast, uniq


def _walk_cut(node, cfg, cut_nodes: list, vars_used: set):
    """递归遍历: 收集切割节点并校验; 统计全部基础变量(外层+工具/目标内)"""
    if isinstance(node, Var):
        if node.name in CUT_TOOLS:
            # 工具宏: 统计其依赖的字段
            sub = Parser(tokenize(expand_cut_tools(node.name))).parse()
            _collect_vars(sub, vars_used)
        else:
            vars_used.add(node.name)
        return
    if isinstance(node, Call):
        if node.name in CUT_FUNCS:
            _validate_cut_node(node, cfg, vars_used)
            cut_nodes.append(node)
            return
        for a in node.args:
            _walk_cut(a, cfg, cut_nodes, vars_used)
        return
    if isinstance(node, Bin):
        _walk_cut(node.left, cfg, cut_nodes, vars_used)
        _walk_cut(node.right, cfg, cut_nodes, vars_used)
        return
    if isinstance(node, Unary):
        _walk_cut(node.x, cfg, cut_nodes, vars_used)
        return
    if isinstance(node, Ternary):
        _walk_cut(node.cond, cfg, cut_nodes, vars_used)
        _walk_cut(node.true, cfg, cut_nodes, vars_used)
        _walk_cut(node.false, cfg, cut_nodes, vars_used)
        return
    # Num/Str 无变量


def _validate_cut_node(node, cfg, vars_used: set):
    """校验单个 CTOP/CBOT 节点的参数合法性与 tool/target 子表达式"""
    name = node.name
    args = node.args
    if len(args) != 5:
        raise ExprError(f"{name} 必须恰好5个参数: {name}(\"工具\", 比例, \"聚合\", \"目标\", 窗口)")

    tool, ratio, agg, target, window = args
    if not isinstance(tool, Str):
        raise ExprError(f"{name} 第1参数(切割工具)必须是引号字符串, 如 \"$amp\" 或 \"ABS($bret)\"")
    if not isinstance(ratio, Num):
        raise ExprError(f"{name} 第2参数(切割比例)必须是数字: 0<比例≤1(取前N%) 或 正整数N(取前N个)")
    if not isinstance(agg, Str):
        raise ExprError(f"{name} 第3参数(聚合方式)必须是引号字符串, 如 \"MEAN\"")
    if not isinstance(target, Str):
        raise ExprError(f"{name} 第4参数(聚合目标)必须是引号字符串, 如 \"$return\"")
    if not isinstance(window, Num):
        raise ExprError(f"{name} 第5参数(窗口)必须是正整数")

    r = float(ratio.value)
    if not ((0 < r <= 1) or (r >= 1 and float(r).is_integer() and r >= 1)):
        raise ExprError(f"{name} 切割比例 {r} 非法: 必须 0<比例≤1 或 正整数N")
    agg_name = agg.value.upper()
    if agg_name not in CUT_AGGS:
        raise ExprError(f"{name} 聚合方式 {agg.value} 非法, 可选: {', '.join(CUT_AGGS)}")
    w = int(window.value)
    if w < 1:
        raise ExprError(f"{name} 窗口 {w} 非法, 必须为正整数")
    max_win = cfg.formula_max_window
    if w > max_win:
        raise ExprError(f"{name} 窗口 {w} 超过上限 {max_win}")

    # 切割工具/目标子表达式校验
    _validate_sub_expr(tool.value, "切割工具", cfg, vars_used)
    _validate_sub_expr(target.value, "聚合目标", cfg, vars_used)


def _validate_sub_expr(expr_str: str, role: str, cfg, vars_used: set):
    """校验 tool/target 子表达式: 可解析、无切割算子、变量合法、分钟模式仅元素级函数"""
    expanded = expand_cut_tools(expr_str)
    if not expanded.strip():
        raise ExprError(f"{role}表达式为空")
    try:
        sub = Parser(tokenize(expanded)).parse()
    except ExprError:
        raise
    except Exception as e:
        raise ExprError(f"{role} 表达式无法解析: {expr_str!r} ({e})")
    if ast_depth(sub) > cfg.formula_max_depth:
        raise ExprError(f"{role} 表达式嵌套过深: {expr_str!r}")
    _walk_sub(sub, role, cfg, vars_used)


def _walk_sub(node, role: str, cfg, vars_used: set):
    """子表达式遍历: 收集变量 + 禁止切割算子/分钟聚合/维持维度算子"""
    if isinstance(node, Var):
        if node.name in CUT_TOOLS:
            sub = Parser(tokenize(expand_cut_tools(node.name))).parse()
            _collect_vars(sub, vars_used)
        else:
            vars_used.add(node.name)
        return
    if isinstance(node, Call):
        if node.name in CUT_FUNCS:
            raise ExprError(f"{role} 表达式内部禁止嵌套切割算子 CTOP/CBOT")
        if cfg.data_frequency == "minute" and node.name not in ELEMENT_FUNCS:
            raise ExprError(
                f"{role} 表达式(分钟模式)仅允许元素级函数({', '.join(sorted(ELEMENT_FUNCS))}), "
                f"禁止使用 {node.name}(时间窗口/聚合算子请放在 CTOP/CBOT 外层)")
        for a in node.args:
            _walk_sub(a, role, cfg, vars_used)
        return
    if isinstance(node, Bin):
        _walk_sub(node.left, role, cfg, vars_used)
        _walk_sub(node.right, role, cfg, vars_used)
        return
    if isinstance(node, Unary):
        _walk_sub(node.x, role, cfg, vars_used)
        return
    if isinstance(node, Ternary):
        _walk_sub(node.cond, role, cfg, vars_used)
        _walk_sub(node.true, role, cfg, vars_used)
        _walk_sub(node.false, role, cfg, vars_used)


def _collect_vars(node, vars_used: set):
    """仅收集变量(不校验)"""
    if isinstance(node, Var):
        vars_used.add(node.name)
    elif isinstance(node, Call):
        for a in node.args:
            _collect_vars(a, vars_used)
    elif isinstance(node, Bin):
        _collect_vars(node.left, vars_used)
        _collect_vars(node.right, vars_used)
    elif isinstance(node, Unary):
        _collect_vars(node.x, vars_used)
    elif isinstance(node, Ternary):
        _collect_vars(node.cond, vars_used)
        _collect_vars(node.true, vars_used)
        _collect_vars(node.false, vars_used)


def cut_fields_used(node, cfg) -> set:
    """切割节点所需的分钟数据字段(不含 $minute), 供分钟数据加载"""
    fields = set()
    for s in (node.args[0].value, node.args[3].value):
        expanded = expand_cut_tools(s)
        try:
            sub = Parser(tokenize(expanded)).parse()
        except Exception:
            continue
        _collect_vars(sub, fields)
    return {f[1:] for f in fields if f.startswith("$") and f != "$minute"}
