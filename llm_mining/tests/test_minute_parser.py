"""minute_parser 分钟公式 解析/类型推断/校验 单元测试

覆盖 minute_validate() 的合法/非法分支、_parse_time 时间常量解析、canonical 规范化、
minute_fields_used 分钟字段收集。
运行: 在项目根 f:\\quant 下执行  conda run -n multifactor python -m pytest llm_mining/tests/test_minute_parser.py -q
"""

import pytest

from llm_mining.local_miner.config import MiningConfig
from llm_mining.local_miner.expr_engine import Parser, ExprError, Call, Var, tokenize
from llm_mining.local_miner.minute.minute_parser import (
    _parse_time, canonical, minute_fields_used, minute_validate,
)


@pytest.fixture
def cfg():
    return MiningConfig()


# ---------------- 合法分钟公式 ----------------

def test_valid_sum(cfg):
    ast, types, agg_nodes, agg_ids = minute_validate("SUM($close)", cfg)
    assert types[id(ast)] == "W"          # 最外层必须是日频宽表
    assert len(agg_nodes) == 1            # 收集到 1 个最大聚合节点
    assert id(ast) in agg_ids


def test_valid_agg_plus_daily(cfg):
    minute_validate("SUM($close) + 1", cfg)
    minute_validate("SUM($close) + TS_STD($return, 20)", cfg)


def test_valid_slice_agg(cfg):
    ast, types, agg_nodes, _ = minute_validate('MEAN(SLICE($close, "09:31", "14:30"))', cfg)
    assert types[id(ast)] == "W"
    assert len(agg_nodes) == 1


def test_valid_mask_agg(cfg):
    minute_validate('SUM(MASK($close, ">", 0.5))', cfg)


def test_valid_intraday_roll(cfg):
    minute_validate("SUM(INTRADAY_MEAN($close, 30))", cfg)


def test_valid_autocorr(cfg):
    minute_validate("TS_AUTOCORR($close, 10)", cfg)


# ---------------- 非法分钟公式 ----------------

def test_root_must_be_daily(cfg):
    # 纯分钟序列未聚合 -> 拒绝(如 INTRADAY 维持维度算子做最外层)
    with pytest.raises(ExprError):
        minute_validate("INTRADAY_MEAN($close, 30)", cfg)


def test_keep_op_root_rejected(cfg):
    # SLICE 是维持维度算子(3D->3D), 不做聚合, 最外层仍是 M -> 拒绝
    with pytest.raises(ExprError):
        minute_validate('SLICE($close, "09:31", "14:30")', cfg)


def test_unknown_minute_field(cfg):
    with pytest.raises(ExprError):
        minute_validate("SUM($bad)", cfg)


def test_roll_window_zero(cfg):
    with pytest.raises(ExprError):
        minute_validate("INTRADAY_MEAN($close, 0)", cfg)


def test_roll_window_exceeds_240(cfg):
    with pytest.raises(ExprError):
        minute_validate("INTRADAY_MEAN($close, 241)", cfg)


def test_slice_time_out_of_range(cfg):
    with pytest.raises(ExprError):
        minute_validate('SLICE($close, "25:00", "14:30")', cfg)


def test_slice_start_after_end(cfg):
    with pytest.raises(ExprError):
        minute_validate('SLICE($close, "14:30", "09:31")', cfg)


def test_autocorr_lag_invalid(cfg):
    with pytest.raises(ExprError):
        minute_validate("TS_AUTOCORR($close, 0)", cfg)
    with pytest.raises(ExprError):
        minute_validate("TS_AUTOCORR($close, 241)", cfg)


def test_empty_formula(cfg):
    with pytest.raises(ExprError):
        minute_validate("", cfg)


# ---------------- _parse_time 时间常量 ----------------

def test_parse_time_basic():
    assert _parse_time("09:31") == 9 * 3600 + 31 * 60
    assert _parse_time("09:31:05") == 9 * 3600 + 31 * 60 + 5
    assert _parse_time("10:00") == 10 * 3600


def test_parse_time_invalid():
    with pytest.raises(ExprError):
        _parse_time("25:00")
    with pytest.raises(ExprError):
        _parse_time("09:61")
    with pytest.raises(ExprError):
        _parse_time("abc")
    with pytest.raises(ExprError):
        _parse_time(12345)          # 非字符串


# ---------------- canonical 规范化 ----------------

def test_canonical_call():
    ast = Parser(tokenize("TS_MEAN($close, 5)")).parse()
    # 数字字面量统一存为 float, 规范化后带 .0
    assert canonical(ast) == "TS_MEAN($close,5.0)"


def test_canonical_var_num():
    ast = Parser(tokenize("SUM($close)")).parse()
    assert canonical(ast) == "SUM($close)"


# ---------------- minute_fields_used ----------------

def test_fields_used_sum(cfg):
    ast, types, _, _ = minute_validate("SUM($close)", cfg)
    assert minute_fields_used(ast, types) == ["close"]


def test_fields_used_multi(cfg):
    ast, types, _, _ = minute_validate("CORR($open, $volume)", cfg)
    assert set(minute_fields_used(ast, types)) == {"open", "volume"}


# ---------------- AST 结构 ----------------

def test_ast_node_types(cfg):
    ast, types, _, _ = minute_validate("SUM($close)", cfg)
    assert isinstance(ast, Call) and ast.name == "SUM"
    assert isinstance(ast.args[0], Var) and ast.args[0].name == "$close"
    assert types[id(ast.args[0])] == "M"   # 聚合参数在分钟上下文 -> M
