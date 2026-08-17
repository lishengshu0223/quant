"""
本地化因子挖掘项目 - 因子切割论(CTOP/CBOT)求值引擎

执行语义(统一日频/分钟):
  CTOP("tool", ratio, "AGG", "target", window):
    对每只股票, 在滚动窗口(最近 window 个 bar, 含当日/当前bar)内按 tool 值降序排序,
    取前 ratio 比例(0<ratio≤1)或前 N 个(正整数)bar 的 target 值, 做 AGG 聚合 -> 该日频值。
  CBOT(...): 取 tool 值最小的后 ratio 比例(或后 N 个)bar。
  聚合后的"日频高阶特征"宽表被缓存, 外层 AST 用日频求值器(CutWideEvaluator)继续组合。

数据路径:
  - 日频(cut_daily): 宽表(日期×股票)上对每只股票做滚动窗口切割。
  - 分钟(cut_minute): 复用 MinuteMarketData 分批流式读取; 按股票在 bar 序列(时间升序)上
    滚动窗口切割, 每个交易日取"该日最后 bar 位置"的窗口结果 -> 日频宽表。
    窗口=bar数: 240=日内1天; 240×D=滚动D个交易日。
"""

import numpy as np
import pandas as pd

from .expr_engine import ExprError, Evaluator, Parser, tokenize, Var, Call, Bin, Unary, Ternary, Num, Str
from .cut_parser import validate_cut, expand_cut_tools, cut_canonical, _parse_node_args, CUT_AGGS


def _k(ratio: float, window: int) -> int:
    """选中的 bar 个数: ratio<1 时取 ceil(ratio×window) 个; 否则取前 min(ratio, window) 个"""
    if ratio < 1:
        return max(1, int(np.ceil(ratio * window)))
    return min(int(round(ratio)), window)


def _agg_selected(sel: np.ndarray, agg: str) -> np.ndarray:
    """对选中矩阵 (B, k, 含NaN) 按行聚合 -> (B,); 全NaN行直接输出NaN, 避免numpy空切片警告"""
    has = np.isfinite(sel).any(axis=1)
    out = np.full(len(sel), np.nan)
    if agg == "MEAN":
        out[has] = np.nanmean(sel[has], axis=1)
        return out
    if agg == "SUM":
        out[has] = np.nansum(sel[has], axis=1)
        return out
    if agg == "MAX":
        out[has] = np.nanmax(sel[has], axis=1)
        return out
    if agg == "MIN":
        out[has] = np.nanmin(sel[has], axis=1)
        return out
    if agg == "MEDIAN":
        out[has] = np.nanmedian(sel[has], axis=1)
        return out
    if agg == "COUNT":
        return np.isfinite(sel).sum(axis=1).astype(np.float64)
    if agg == "LAST":
        # 排序后第1列 = tool 最极端者(CTOP最大/CBOT最小)的 target
        idx = np.argmax(has, axis=1)
        vals = sel[np.arange(len(sel)), idx]
        return np.where(has, vals, np.nan)
    if agg in ("STD", "SKEW", "KURT"):
        # pandas 行聚合, 与 scipy 偏差校正口径一致; 有效值不足时自动 NaN
        df = pd.DataFrame(sel)
        if agg == "STD":
            return df.std(axis=1).to_numpy()
        if agg == "SKEW":
            return df.skew(axis=1).to_numpy()
        return df.kurt(axis=1).to_numpy()
    raise ExprError(f"切割聚合方式 {agg} 未实现, 可选: {', '.join(CUT_AGGS)}")


def _select_roll(tool_mat: np.ndarray, target_mat: np.ndarray, ratio: float, top: bool):
    """对窗口矩阵 (B, window) 按 tool 取前/后 k 个的 target 选中矩阵 (B, k)。
    tool 无效(NaN)的 bar 不参与切割(置 ±inf 排到最后/最前), target 无效者在聚合前被剔除。"""
    window = tool_mat.shape[1]
    k = _k(ratio, window)
    key = np.where(np.isfinite(tool_mat), tool_mat, -np.inf if top else np.inf)
    if k < window:
        part = np.argpartition(key, (-k if top else (k - 1)), axis=1)
        idx = part[:, -k:] if top else part[:, :k]
        sel_t = np.take_along_axis(tool_mat, idx, axis=1)
        sel_g = np.take_along_axis(target_mat, idx, axis=1)
    else:
        sel_t, sel_g = tool_mat, target_mat
    ok = np.isfinite(sel_t) & np.isfinite(sel_g)
    return np.where(ok, sel_g, np.nan)


def _rolling_cut_batch(tool: np.ndarray, target: np.ndarray, window: int,
                       ratio: float, agg: str, top: bool) -> np.ndarray:
    """日频: tool/target (n, b) 时间升序; 返回 (n, b): 位置 i 取窗口 [i-window+1, i] 切割聚合。
    注意: sliding_window_view 把窗口维放在最后 -> 中间张量形状 (m, b, window),
    排序/取前k/聚合都必须沿 axis=2(窗口维)操作。"""
    n, b = tool.shape
    if n < window:
        return np.full((n, b), np.nan)
    m = n - window + 1
    from numpy.lib.stride_tricks import sliding_window_view
    t_mat = sliding_window_view(tool, window, axis=0)     # (m, b, window)
    g_mat = sliding_window_view(target, window, axis=0)
    k = _k(ratio, window)
    key = np.where(np.isfinite(t_mat), t_mat, -np.inf if top else np.inf)
    if k < window:
        part = np.argpartition(key, (-k if top else (k - 1)), axis=2)
        idx = part[:, :, -k:] if top else part[:, :, :k]  # (m, b, k)
        sel_t = np.take_along_axis(t_mat, idx, axis=2)
        sel_g = np.take_along_axis(g_mat, idx, axis=2)
    else:
        sel_t, sel_g = t_mat, g_mat
    ok = np.isfinite(sel_t) & np.isfinite(sel_g)
    sel_g = np.where(ok, sel_g, np.nan)
    vals = np.empty((m, b), dtype=np.float64)
    for j in range(b):
        vals[:, j] = _agg_selected(sel_g[:, j, :], agg)
    out = np.full((n, b), np.nan)
    out[window - 1:] = vals
    return out


def _rolling_cut_at(tool: np.ndarray, target: np.ndarray, end_pos: np.ndarray,
                    window: int, ratio: float, agg: str, top: bool) -> np.ndarray:
    """分钟: 每股 bar 序列(时间升序) tool/target (n,); 对每个结束位置 end_pos 取
    [end-window+1, end] 窗口切割聚合 -> (E,)"""
    E = len(end_pos)
    if E == 0:
        return np.empty(0)
    rows = end_pos - window + 1
    valid = rows >= 0
    out = np.full(E, np.nan)
    if not valid.any():
        return out
    pos = np.flatnonzero(valid)
    r = rows[pos]
    cols = np.arange(window)
    t_mat = tool[r[:, None] + cols[None, :]]
    g_mat = target[r[:, None] + cols[None, :]]
    sel_g = _select_roll(t_mat, g_mat, ratio, top)
    vals = _agg_selected(sel_g, agg)
    out[pos] = vals
    return out


# =============================================================================
# 外层宽表求值器(日频): 切割节点命中缓存 + $amp/$bret 工具宏特判
# =============================================================================

class CutWideEvaluator(Evaluator):
    """整棵AST日频求值: CTOP/CBOT 节点命中切割缓存宽表, $amp/$bret 宏即时展开,
    其余走标准日频求值(含全部日频算子/截面变换)。"""

    def __init__(self, data, cut_cache: dict):
        super().__init__(data)
        self.cut_cache = cut_cache

    def eval(self, node):
        if isinstance(node, Call) and id(node) in self.cut_cache:
            return self.cut_cache[id(node)]
        if isinstance(node, Var):
            if node.name == "$amp":
                hi, lo = self.data.high, self.data.low
                return ((hi - lo) / (hi + lo + 1e-12) / 2).astype("float32")
            if node.name == "$bret":
                return self.data.ret
            return self.data.var(node.name)
        return super().eval(node)


def _eval_daily_expr(expr_str: str, data) -> pd.DataFrame:
    """在日频宽表上求值元素级/任意日频子表达式(切割工具/目标)"""
    ast = Parser(tokenize(expr_str)).parse()
    res = Evaluator(data).eval(ast)
    if not isinstance(res, pd.DataFrame):
        raise ExprError(f"切割工具/目标表达式计算结果不是序列: {expr_str}")
    return res.reindex(index=data.close.index, columns=data.close.columns)


# =============================================================================
# 日频切割
# =============================================================================

def compute_factor_cut_daily(expr: str, data, cfg) -> pd.DataFrame:
    """日频切割入口: 校验 -> 计算全部切割节点宽表 -> 外层求值 -> 收尾检查"""
    ast, cut_nodes = validate_cut(expr, cfg)
    if not cut_nodes:
        # 退化: 纯日频公式(切割模式允许, 但提示词强制优先使用切割算子)
        from .expr_engine import compute_factor
        return compute_factor(expr, data, cfg)

    cache = {}
    for node in cut_nodes:
        cache[id(node)] = _compute_cut_node_daily(node, data, cfg)

    result = CutWideEvaluator(data, cache).eval(ast)
    if not isinstance(result, pd.DataFrame):
        raise ExprError("公式计算结果为标量, 不是有效的截面因子")
    from .minute.minute_data import _finalize
    return _finalize(result)


def _compute_cut_node_daily(node, data, cfg) -> pd.DataFrame:
    """单个切割节点的全市场日频宽表"""
    tool_str, ratio, agg, target_str, window = _parse_node_args(node)
    tool_wide = _eval_daily_expr(expand_cut_tools(tool_str), data)
    target_wide = _eval_daily_expr(expand_cut_tools(target_str), data)
    top = node.name == "CTOP"

    codes = list(tool_wide.columns)
    n_days = tool_wide.shape[0]
    out = np.full((n_days, len(codes)), np.nan)
    t_arr = tool_wide.to_numpy(dtype=np.float64)
    g_arr = target_wide.to_numpy(dtype=np.float64)
    batch = 200  # 控制 strided 窗口矩阵内存
    for s in range(0, len(codes), batch):
        e = min(s + batch, len(codes))
        out[:, s:e] = _rolling_cut_batch(t_arr[:, s:e], g_arr[:, s:e],
                                         window, ratio, agg, top)
    return pd.DataFrame(out, index=tool_wide.index, columns=codes)


# =============================================================================
# 分钟切割
# =============================================================================

class CutMinuteEval:
    """批内分钟元素级表达式求值器(切割工具/目标专用):
    支持 $amp/$bret 宏与元素级函数(ABS/SIGN/EXP/SQRT/LOG/INV/POW/WHERE)+四则/比较/逻辑/三元。"""

    ELEMENT_OPS = {"ABS", "SIGN", "EXP", "SQRT", "LOG", "INV", "POW", "WHERE"}

    def __init__(self, mmd, frame, cfg):
        self.mmd = mmd
        self.frame = frame
        self.cfg = cfg

    def eval(self, node):
        from .minute.minute_data import MinuteExpr
        from .minute.minute_sparse_eval import _m_binop
        if isinstance(node, Num):
            return node.value
        if isinstance(node, Str):
            raise ExprError("字符串常量不能作为切割工具/目标表达式")
        if isinstance(node, Var):
            return self.mmd.field_of(self.frame, node.name)
        if isinstance(node, Unary):
            x = self.eval(node.x)
            if isinstance(x, MinuteExpr):
                return MinuteExpr(x.code, x.date, x.t, (-x.v).astype(np.float32))
            return -x
        if isinstance(node, Bin):
            left = self.eval(node.left)
            right = self.eval(node.right)
            if isinstance(left, MinuteExpr) and not isinstance(right, MinuteExpr):
                right = self._broadcast(right, left)
            elif isinstance(right, MinuteExpr) and not isinstance(left, MinuteExpr):
                left = self._broadcast(left, right)
            if isinstance(left, MinuteExpr) or isinstance(right, MinuteExpr):
                return _m_binop(node.op, left, right)
            raise ExprError("切割工具/目标表达式两侧都不是分钟序列")
        if isinstance(node, Ternary):
            c = self.eval(node.cond)
            tt = self.eval(node.true)
            ff = self.eval(node.false)
            if isinstance(c, MinuteExpr) and isinstance(tt, MinuteExpr) and isinstance(ff, MinuteExpr):
                if not (c.aligned(tt) and c.aligned(ff)):
                    raise ExprError("分钟条件表达式三分支无法对齐")
                return MinuteExpr(c.code, c.date, c.t,
                                  np.where(c.v.astype(bool), tt.v, ff.v).astype(np.float32))
            raise ExprError("分钟条件表达式三分支必须是分钟序列")
        if isinstance(node, Call):
            return self._eval_call(node)
        raise ExprError(f"切割工具/目标表达式中的非法节点: {type(node).__name__}")

    def _eval_call(self, node):
        from .minute.minute_data import MinuteExpr
        name = node.name
        if name not in self.ELEMENT_OPS:
            raise ExprError(
                f"切割工具/目标表达式(分钟模式)仅允许元素级函数"
                f"({', '.join(sorted(self.ELEMENT_OPS))}), 禁止使用 {name}")
        if name == "WHERE":
            return self._eval_where(node)
        x = self.eval(node.args[0])
        if not isinstance(x, MinuteExpr):
            raise ExprError(f"{name} 的参数必须是分钟序列")
        v = x.v.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if name == "ABS":
                r = np.abs(v)
            elif name == "SIGN":
                r = np.sign(v)
            elif name == "EXP":
                r = np.exp(v)
            elif name == "SQRT":
                r = np.where(v > 0, np.sqrt(v), np.nan)
            elif name == "LOG":
                r = np.log1p(v)  # log(x+1), 与日频口径一致
            elif name == "INV":
                r = 1.0 / (v + 1e-12)
            elif name == "POW":
                p = self.eval(node.args[1])
                r = np.power(v, p)
            else:
                raise ExprError(f"切割工具/目标不允许使用 {name}")
        return MinuteExpr(x.code, x.date, x.t, r.astype(np.float32))

    def _eval_where(self, node):
        from .minute.minute_data import MinuteExpr
        cond = self.eval(node.args[0])
        tt = self.eval(node.args[1])
        ff = self.eval(node.args[2])
        if not isinstance(cond, MinuteExpr):
            raise ExprError("WHERE 条件必须是分钟序列")
        if not isinstance(tt, MinuteExpr) or not isinstance(ff, MinuteExpr):
            raise ExprError("WHERE 的 true/false 分支必须是分钟序列")
        if not (cond.aligned(tt) and cond.aligned(ff)):
            raise ExprError("WHERE 三分支无法对齐")
        return MinuteExpr(cond.code, cond.date, cond.t,
                          np.where(cond.v.astype(bool), tt.v, ff.v).astype(np.float32))

    def _broadcast(self, val, mv):
        if isinstance(val, (int, float, np.number)):
            return np.full(len(mv.v), float(val), dtype=np.float32)
        if not isinstance(val, pd.DataFrame):
            raise ExprError("切割工具/目标表达式与不支持的类型做运算")
        w = val.reindex(columns=range(len(self.mmd.batch_codes)))
        s = w.stack(future_stack=True)
        dates = s.index.get_level_values(0).strftime("%Y%m%d").astype(np.int32)
        s.index = pd.MultiIndex.from_arrays([dates, s.index.get_level_values(1)])
        idx = pd.MultiIndex.from_arrays([mv.date, mv.code])
        return s.reindex(idx).to_numpy(dtype=np.float32)


def compute_factor_cut_minute(expr: str, data, cfg) -> pd.DataFrame:
    """分钟切割入口: 分批流式读取分钟数据 -> 逐切割节点计算全市场宽表 -> 外层求值"""
    ast, cut_nodes = validate_cut(expr, cfg)
    if not cut_nodes:
        from .minute_engine import compute_factor_minute
        return compute_factor_minute(expr, data, cfg)

    from .cut_parser import cut_fields_used
    from .minute.minute_data import MinuteMarketData, _finalize

    fields = set()
    for node in cut_nodes:
        fields |= cut_fields_used(node, cfg)
    mmd = MinuteMarketData(cfg, data, fields)

    keys = list(dict.fromkeys(cut_canonical(n) for n in cut_nodes))
    parts = {k: [] for k in keys}
    n_batch = 0
    for batch_codes in mmd.batches:
        df = mmd._load_batch(batch_codes)
        if df is None:
            continue
        mmd.batch_codes = batch_codes
        try:
            runner = CutMinuteEval(mmd, df, cfg)
            for node in cut_nodes:
                k = cut_canonical(node)
                wide = _compute_cut_node_minute_batch(node, runner, mmd, cfg)
                parts[k].append(wide)
        finally:
            del df
        n_batch += 1
    if not n_batch:
        raise ExprError("分钟数据为空, 无法计算切割节点")

    cache = {}
    for k in keys:
        cache[k] = pd.concat(parts[k], axis=1).reindex(
            index=mmd.daily_index, columns=mmd.all_codes)
    node_cache = {id(n): cache[cut_canonical(n)] for n in cut_nodes}

    result = CutWideEvaluator(data, node_cache).eval(ast)
    if not isinstance(result, pd.DataFrame):
        raise ExprError("公式计算结果为标量, 不是有效的截面因子")
    return _finalize(result)


def _compute_cut_node_minute_batch(node, runner: CutMinuteEval, mmd, cfg) -> pd.DataFrame:
    """单个切割节点在批内的宽表(列=批内股票代码)"""
    tool_str, ratio, agg, target_str, window = _parse_node_args(node)
    t_expr = runner.eval(Parser(tokenize(expand_cut_tools(tool_str))).parse())
    g_expr = runner.eval(Parser(tokenize(expand_cut_tools(target_str))).parse())
    top = node.name == "CTOP"

    codes = mmd.batch_codes
    n_days = len(mmd.daily_index)
    date_to_row = {int(d.strftime("%Y%m%d")): i for i, d in enumerate(mmd.daily_index)}
    out = np.full((n_days, len(codes)), np.nan)

    code_arr, date_arr = t_expr.code, t_expr.date
    if len(code_arr) == 0:
        return pd.DataFrame(out, index=mmd.daily_index, columns=codes)
    tv = t_expr.v.astype(np.float64)
    gv = g_expr.v.astype(np.float64)

    # 批内长表已按 code,date,t 排序 -> 直接分段(每股一个连续区间)
    bounds = np.flatnonzero(np.diff(code_arr) != 0) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(code_arr)]])
    for ci in range(len(codes)):
        if ci >= len(starts):
            continue  # 该股本批无分钟数据(因子恒为NaN)
        s, e = starts[ci], ends[ci]
        if e <= s:
            continue
        c_dates = date_arr[s:e]
        # 每个交易日最后 bar 的位置(窗口结束点)
        day_bounds = np.flatnonzero(np.diff(c_dates) != 0) + 1
        e_pos = np.concatenate([day_bounds, [e - s]]) - 1
        rows_out = np.array([date_to_row.get(int(dd)) for dd in c_dates[e_pos]])
        ok_row = rows_out >= 0
        if not ok_row.any():
            continue
        vals = _rolling_cut_at(tv[s:e], gv[s:e], e_pos[ok_row].astype(np.int64),
                               window, ratio, agg, top)
        out[rows_out[ok_row], ci] = vals
    return pd.DataFrame(out, index=mmd.daily_index, columns=codes)


def compute_factor_cut(expr: str, data, cfg) -> pd.DataFrame:
    """切割模式对外入口: 按 data_frequency 分发日频/分钟切割"""
    if getattr(cfg, "data_frequency", "daily") == "minute":
        return compute_factor_cut_minute(expr, data, cfg)
    return compute_factor_cut_daily(expr, data, cfg)
