"""expr_engine 日频公式 解析/校验 单元测试

覆盖 validate() 的合法/非法分支与最外层 RANK 禁止规则。
运行: 在项目根 f:\\quant 下执行  conda run -n multifactor python -m pytest llm_mining/tests/test_expr_engine.py -q
"""

import pytest

from llm_mining.local_miner.config import MiningConfig
from llm_mining.local_miner.expr_engine import (
    ExprError, Parser, ast_depth, bracket_scan, is_outer_rank, tokenize, validate,
    Bin, Call, Num, Unary, Var,
)


@pytest.fixture
def cfg():
    return MiningConfig()


# ---------------- 合法公式 ----------------

def test_valid_simple_var(cfg):
    ast = validate("$close", cfg)
    assert isinstance(ast, Var) and ast.name == "$close"


def test_valid_function(cfg):
    ast = validate("TS_MEAN($close, 5)", cfg)
    assert isinstance(ast, Call) and ast.name == "TS_MEAN"


def test_valid_arithmetic(cfg):
    ast = validate("($close - $open) / $open", cfg)
    assert ast is not None


def test_valid_where_ternary(cfg):
    validate("WHERE($close > $open, $close, $open)", cfg)


def test_valid_rank_as_intermediate(cfg):
    # RANK 作为中间步骤允许(TS_MEAN 内层)
    validate("TS_MEAN(RANK($close), 5)", cfg)


def test_valid_no_window_arg(cfg):
    # 窗口参数可省略(使用 WINDOW_DEFAULTS)
    validate("TS_MEAN($close)", cfg)


def test_valid_multiple_vars(cfg):
    validate("TS_STD($return, 20) * $close", cfg)


# ---------------- 非法公式 ----------------

def test_empty_formula(cfg):
    with pytest.raises(ExprError):
        validate("", cfg)


def test_unknown_var(cfg):
    with pytest.raises(ExprError):
        validate("$badvar", cfg)


def test_unknown_function(cfg):
    with pytest.raises(ExprError):
        validate("NOT_A_FUNC($close)", cfg)


def test_window_zero(cfg):
    with pytest.raises(ExprError):
        validate("TS_MEAN($close, 0)", cfg)


def test_window_float(cfg):
    with pytest.raises(ExprError):
        validate("TS_MEAN($close, 5.5)", cfg)


def test_window_exceeds_max(cfg):
    with pytest.raises(ExprError):
        validate("TS_MEAN($close, 241)", cfg)


def test_arg_count_mismatch(cfg):
    with pytest.raises(ExprError):
        validate("TS_MEAN($close, 5, 6)", cfg)  # TS_MEAN 最多 2 参数


def test_quantile_float_param_invalid(cfg):
    with pytest.raises(ExprError):
        validate("TS_QUANTILE($close, 5, 1.5)", cfg)  # 分位数必须 0~1


def test_unmatched_paren(cfg):
    with pytest.raises(ExprError):
        validate("TS_MEAN($close, 5", cfg)  # 缺少右括号


# ---------------- 最外层 RANK 禁止 ----------------

def test_outer_rank_rejected(cfg):
    with pytest.raises(ExprError):
        validate("RANK($close)", cfg)


def test_unary_minus_rank_rejected(cfg):
    with pytest.raises(ExprError):
        validate("-RANK($close)", cfg)


def test_affine_rank_mul_rejected(cfg):
    with pytest.raises(ExprError):
        validate("RANK($close) * 2", cfg)


def test_affine_rank_add_rejected(cfg):
    with pytest.raises(ExprError):
        validate("RANK($close) + 1", cfg)


def test_outer_rank_detector_unit():
    rank = Parser(tokenize("RANK($close)")).parse()
    assert is_outer_rank(rank)
    unary = Parser(tokenize("-RANK($close)")).parse()
    assert is_outer_rank(unary)
    affine = Parser(tokenize("RANK($close) * 2")).parse()
    assert is_outer_rank(affine)
    inter = Parser(tokenize("TS_MEAN(RANK($close), 5)")).parse()
    assert not is_outer_rank(inter)


# ---------------- 辅助函数 ----------------

def test_bracket_scan():
    assert bracket_scan("TS_MEAN($close, 5)") == 1
    assert bracket_scan("WHERE($close > $open, TS_MEAN($close, 5), $open)") == 2


def test_ast_depth():
    ast = Parser(tokenize("TS_MEAN($close, 5)")).parse()
    assert ast_depth(ast) == 2
    ast2 = Parser(tokenize("TS_MEAN(RANK($close), 5)")).parse()
    assert ast_depth(ast2) == 3


def test_tokenize_types():
    toks = tokenize("TS_MEAN($close, 5)")
    kinds = [k for k, _ in toks]
    assert "IDENT" in kinds and "VAR" in kinds and "NUM" in kinds and "OP" in kinds
