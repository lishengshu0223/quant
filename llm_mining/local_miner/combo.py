"""
本地化因子挖掘项目 - 组合类因子检测与比较(最后手段机制)

设计(对应需求: 组合=最后手段 + 事后隐藏 + 枯竭判定):
- 将两个及以上经济逻辑相近/相同的项相加、平均、加权组合, 是"最后手段"——
  通常意味着该因子的单一经济逻辑已优化到尽头。
- detect_combo: 公式 AST 层检测顶层是否为"多子式加法/平均结构"。
  识别: A+B / A+B+C / (A+B)/k / 0.3A+0.4B+0.3C 等;
  不识别(视为结构创新): 门控乘法(核心×门控)、归一化除法(核心÷波动率)、单包裹平滑。
- combo_better: "单逻辑因子 vs 被隐藏组合"谁更好的字典序比较。
  合格性 > 多头年度稳定性IR > 多头年化 > IC(与优化模式采纳标准一致)。
"""

from . import expr_engine


def _strip_unary(node):
    while isinstance(node, expr_engine.Unary):
        node = node.x
    return node


def _has_call(node) -> bool:
    if node is None:
        return False
    if isinstance(node, expr_engine.Call):
        return True
    if isinstance(node, expr_engine.Bin):
        return _has_call(node.left) or _has_call(node.right)
    if isinstance(node, expr_engine.Unary):
        return _has_call(node.x)
    if isinstance(node, expr_engine.Ternary):
        return (_has_call(node.cond) or _has_call(node.true) or _has_call(node.false))
    return False


def _split_additive(node, out: list):
    """把顶层加减树展开成独立的项"""
    node = _strip_unary(node)
    if isinstance(node, expr_engine.Bin) and node.op in ("+", "-"):
        _split_additive(node.left, out)
        _split_additive(node.right, out)
    else:
        out.append(node)


def detect_combo(expr: str) -> bool:
    """
    检测公式是否为"多子式组合"(最后手段信号)。
    命中则视为组合类因子: 评价后应隐藏(不反馈给agent), 并计入枯竭判定。
    """
    if not expr or not expr.strip():
        return False
    try:
        node = expr_engine.Parser(expr_engine.tokenize(expr)).parse()
    except Exception:
        return False
    node = _strip_unary(node)
    # (A+B)/k 或 (A+B)/(分母): 分子是组合
    if isinstance(node, expr_engine.Bin) and node.op == "/":
        terms = []
        _split_additive(node.left, terms)
        return (len(terms) >= 2
                and sum(1 for t in terms if _has_call(t)) >= 2)
    # 顶层直接是加减组合 / 加权和
    terms = []
    _split_additive(node, terms)
    return (len(terms) >= 2
            and sum(1 for t in terms if _has_call(t)) >= 2)


def combo_better(a: dict, b: dict) -> bool:
    """
    a 是否优于 b(用于"单逻辑因子 vs 被隐藏组合"比较)。
    字典序比较: 合格性 > 分组单调性评级(S>A>B>C) > 多头年度稳定性IR(score) > 多头年化 > IC。
    说明: 合格性已内置"单调性非C级"前置门槛; 在此基础上再比较评级的细分差异(B/A/S),
    单调性更好的因子(更接近S)优先胜出。
    """
    a_q, b_q = bool(a.get("qualified")), bool(b.get("qualified"))
    if a_q != b_q:
        return a_q
    # 分组单调性评级: S>A>B>C (旧评价可能无该字段, 缺失则跳过此项比较)
    grade_order = {"S": 0, "A": 1, "B": 2, "C": 3}
    a_g = (a.get("monotonicity_grade") or {}).get("grade")
    b_g = (b.get("monotonicity_grade") or {}).get("grade")
    if a_g and b_g and a_g != b_g:
        return grade_order[a_g] < grade_order[b_g]
    a_ir = (a.get("long_stability") or {}).get("score")
    b_ir = (b.get("long_stability") or {}).get("score")
    if a_ir is not None and b_ir is not None and a_ir != b_ir:
        return a_ir > b_ir
    if a_ir is None and b_ir is not None:
        return False
    if a_ir is not None and b_ir is None:
        return True
    a_an, b_an = a.get("long_annual"), b.get("long_annual")
    if a_an is not None and b_an is not None and a_an != b_an:
        return a_an > b_an
    a_ic, b_ic = a.get("ic_mean"), b.get("ic_mean")
    if a_ic is not None and b_ic is not None and a_ic != b_ic:
        return a_ic > b_ic
    return False
