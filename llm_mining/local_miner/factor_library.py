"""
本地化因子挖掘项目 - 因子库(系列文件)管理

设计原则(对应需求三 2 与"静态 vs 动态"要求):
- 因子库分为两大目录(与失败因子库 failed/ 对应):
    factor_library/success/<series_id>_<name>/  每个成功因子一个独立文件夹, 内含三件套:
        <id>_<name>.json          结构化因子信息 + 回测摘要(整条迭代路径+诊断+成败经验)
        <id>_<name>_backtest.h5   完整回测宽表(float32+zlib, 可随时重放任意回测与图表)
        <id>_<name>_回测评价.png  因子评价图(标注合格)
    factor_library/failed/        失败因子库(见 failed_library.py: 每轮明细+轮次总结+战役总结)
- 静态写死在文件中的: 因子公式、简介、风格、整条迭代路径(表达式+结构化回测结果)、
  诊断结果、自动归纳的假设历史与成败经验。
- 动态交给agent判断的: 具体优化方向、新假设、避让策略(由 prompts 基于本文件内容即时组装)。

提供:
- 系列文件读写/原子保存/按ID查找/下一可用ID
- 迭代路径追加与最佳因子更新(含诊断)
- 库摘要文本(新因子挖掘防撞车注入) / 系列当前全况文本(优化模式注入)
- 库相关性检查(新合格因子与库内因子截面Spearman相关, 超限拒绝)
"""

import glob
import json
import os
import re
import datetime

import numpy as np
import pandas as pd

from .config import FACTOR_LIBRARY_DIR, ensure_workspace

# 成功因子库目录(与 failed/ 平级): 每个成功因子一个子文件夹, 内含 json+h5+png 三件套
SUCCESS_DIR = os.path.join(FACTOR_LIBRARY_DIR, "success")


# =============================================================================
# 文件读写
# =============================================================================

def _safe_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]", "", name or "factor")[:40] or "factor"


def series_dir(series_id: str, name: str = "") -> str:
    """返回系列专属文件夹: success/<series_id>_<name>/(文件夹不存在时由 save 创建)"""
    return os.path.join(SUCCESS_DIR, f"{series_id}_{_safe_name(name)}")


def _series_glob(series_id: str) -> list:
    return sorted(glob.glob(os.path.join(SUCCESS_DIR, f"{series_id}_*",
                                         f"{series_id}_*.json")))


def series_path(series_id: str, name: str = "") -> str:
    """返回系列文件路径。若已存在则沿用(保持ID前缀), 否则按 ID_name 新建。"""
    existing = _series_glob(series_id)
    if existing:
        return existing[0]
    safe = _safe_name(name)
    return os.path.join(series_dir(series_id, safe), f"{series_id}_{safe}.json")


def load_series(series_id: str) -> dict | None:
    existing = _series_glob(series_id)
    if not existing:
        return None
    try:
        with open(existing[0], "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_library() -> list:
    """加载全部成功因子系列(F日频 + M分钟), 按 series_id 排序"""
    ensure_workspace()
    series_list = []
    for path in sorted(glob.glob(os.path.join(SUCCESS_DIR, "*", "[FM]*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as f:
                series_list.append(json.load(f))
        except Exception:
            continue
    series_list.sort(key=lambda s: s.get("series_id", ""))
    return series_list


def save_series(series: dict):
    """原子写入系列文件, 并清理同ID的旧名文件(改名后避免残留)"""
    ensure_workspace()
    series["updated_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = series_path(series["series_id"], series.get("name", ""))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(series, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    # 清理同ID其它残留文件
    for old in _series_glob(series["series_id"]):
        if os.path.abspath(old) != os.path.abspath(path):
            try:
                os.remove(old)
            except Exception:
                pass


def next_series_id(prefix: str = "F") -> str:
    """扫描因子库, 返回下一个可用的系列ID。
    prefix: F=日频因子(F001/F002/...), M=分钟因子(M001/M002/...)。两个前缀独立计数、互不冲突。
    库文件同时含 F/M 时, 相关性检查统一对全库计算(见 load_library/check_library_correlation)。"""
    ids = []
    for path in glob.glob(os.path.join(SUCCESS_DIR, "*", f"{prefix}*.json")):
        m = re.match(prefix + r"(\d+)", os.path.basename(path))
        if m:
            ids.append(int(m.group(1)))
    return f"{prefix}{max(ids) + 1:03d}" if ids else f"{prefix}001"


def count_series(prefix: str) -> int:
    """统计因子库中指定前缀(如 M/F)的已入库系列数(按系列ID去重, 同ID改名残留不重复计)"""
    return len({s.get("series_id") for s in load_library()
                if str(s.get("series_id", "")).startswith(prefix)})


# =============================================================================
# 系列构建与更新
# =============================================================================

def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _factor_lesson(entry: dict) -> tuple:
    """
    从单个因子评价归纳一条经验。返回 (类型, 文本); 类型∈{success, failure, None}
    """
    ev = entry.get("eval") or {}
    name = entry.get("name", "")
    expr = entry.get("expr", "")
    if ev.get("error"):
        return "failure", f"[{name}] 公式 {expr} 计算/评价失败: {str(ev['error'])[:80]}"
    if ev.get("qualified"):
        return "success", (
            f"[{name}] 公式 {expr} 合格: IC均值={ev.get('ic_mean', 0)*100:+.3f}%, "
            f"多头年化={ev.get('long_annual', 0)*100:.2f}%, "
            f"月度正占比={ev.get('monthly_pos_ratio', 0)*100:.0f}%")
    if not ev.get("direction_ok"):
        return "failure", (
            f"[{name}] 公式 {expr} 方向无效(IC与多头收益矛盾或为负): "
            f"IC={ev.get('ic_mean', 0)*100:+.3f}%, 多头累计={ev.get('long_total', 0)*100:+.2f}%, "
            "该逻辑应彻底放弃而非微调")
    reasons = []
    if not ev.get("monthly_ok"):
        reasons.append(f"月度IC正占比仅{ev.get('monthly_pos_ratio', 0)*100:.0f}%不足")
    if not ev.get("yearly_ok"):
        bad = ev.get("bad_hist_years") or {}
        reasons.append("历史完整年份多头超额存在为负(" +
                       ", ".join(f"{y}年" for y in sorted(bad)) + ")")
    return "failure", (
        f"[{name}] 公式 {expr} 方向正确但未达标: " + "; ".join(reasons) +
        f" (IC={ev.get('ic_mean', 0)*100:+.3f}%, 多头累计={ev.get('long_total', 0)*100:+.2f}%)")


def new_series(series_id: str, best_entry: dict, hypothesis: str, style: str,
               diagnostics: dict | None = None) -> dict:
    """以首个合格因子创建新系列文件"""
    best = dict(best_entry)
    best["hypothesis"] = hypothesis
    best["style"] = style
    best["diagnostics"] = diagnostics
    path_entry = _make_path_entry(best_entry, hypothesis, style)
    lessons_s, lessons_f = [], []
    kind, text = _factor_lesson(best_entry)
    if kind == "success":
        lessons_s.append(text)
    return {
        "series_id": series_id,
        "name": best_entry.get("name", ""),
        "status": "qualified",
        "style": style,
        "desc": best_entry.get("desc", ""),
        "created_at": _now(),
        "updated_at": _now(),
        "best": best,
        "path": [path_entry],
        "hypotheses_tried": [hypothesis] if hypothesis else [],
        "lessons_success": lessons_s,
        "lessons_failure": lessons_f,
    }


def create_series_from_history(series_id: str, history: list, best_entry: dict,
                               hypothesis: str, style: str,
                               diagnostics: dict | None = None) -> dict:
    """新因子挖掘模式合格时: 用整条历史轮次构建系列(路径含所有失败尝试), 并设定最佳因子"""
    series = {
        "series_id": series_id,
        "name": best_entry.get("name", ""),
        "status": "qualified",
        "style": style,
        "desc": best_entry.get("desc", ""),
        "created_at": _now(),
        "updated_at": _now(),
        "best": None,
        "path": [],
        "hypotheses_tried": [],
        "lessons_success": [],
        "lessons_failure": [],
    }
    for rec in history:
        append_round(series, rec.get("round"), rec.get("factors", []),
                     rec.get("hypothesis", ""))
    update_best(series, best_entry, hypothesis, style, diagnostics)
    return series


def _make_path_entry(entry: dict, hypothesis: str, style: str) -> dict:
    ev = entry.get("eval") or {}
    return {
        "round": entry.get("round"),
        "name": entry.get("name", ""),
        "expr": entry.get("expr", ""),
        "style": style,
        "hypothesis": hypothesis,
        "qualified": bool(ev.get("qualified")),
        "eval": ev,
    }


def append_round(series: dict, round_no: int, factors: list, hypothesis: str,
                 style: str = ""):
    """把一轮的所有因子追加进迭代路径, 并归纳假设与成败经验(去重限长)"""
    if hypothesis and hypothesis not in series["hypotheses_tried"]:
        series["hypotheses_tried"].append(hypothesis)
    for entry in factors:
        entry = dict(entry)
        entry.setdefault("round", round_no)
        series["path"].append(_make_path_entry(entry, hypothesis,
                                               style or entry.get("style", "")))
        kind, text = _factor_lesson(entry)
        bucket = series["lessons_success"] if kind == "success" else \
            (series["lessons_failure"] if kind == "failure" else None)
        if bucket is not None and text not in bucket:
            bucket.append(text)
    # 限长, 保留最近的经验
    series["hypotheses_tried"] = series["hypotheses_tried"][-30:]
    series["lessons_success"] = series["lessons_success"][-30:]
    series["lessons_failure"] = series["lessons_failure"][-40:]


def update_best(series: dict, best_entry: dict, hypothesis: str, style: str,
                diagnostics: dict | None = None):
    """更新系列的最佳因子(含诊断), 并刷新名称/简介/风格"""
    best = dict(best_entry)
    best["hypothesis"] = hypothesis
    best["style"] = style
    best["diagnostics"] = diagnostics
    series["best"] = best
    series["status"] = "qualified"
    if best_entry.get("name"):
        series["name"] = best_entry["name"]
    if best_entry.get("desc"):
        series["desc"] = best_entry["desc"]
    if style:
        series["style"] = style


# =============================================================================
# 文本组装(注入提示词)
# =============================================================================

def _eval_brief(ev: dict) -> str:
    if not ev or ev.get("error"):
        return f"评价失败: {str(ev.get('error', ''))[:60]}"
    return (
        f"IC均值={ev.get('ic_mean', 0)*100:+.3f}%, ICIR={ev.get('icir', 0):.3f}, "
        f"多头累计超额={ev.get('long_total', 0)*100:+.2f}%, "
        f"多头年化={ev.get('long_annual', 0)*100:.2f}%, "
        f"月度正占比={ev.get('monthly_pos_ratio', 0)*100:.0f}%, "
        f"{'合格' if ev.get('qualified') else '不合格'}")


def _stability_brief(stab: dict) -> str:
    if not stab or stab.get("score") is None:
        return "(暂无稳定性数据)"
    cv_txt = f"{stab.get('cv'):.2f}" if stab.get("cv") is not None else "NA"
    hhi_txt = f"{stab.get('hhi'):.3f}" if stab.get("hhi") is not None else "NA"
    return (
        f"年度信息比率={stab.get('yearly_ir'):.2f}"
        f"(年均{stab.get('mean', 0)*100:+.2f}%/年标准差{stab.get('std', 0)*100:.2f}%), "
        f"变异系数={cv_txt}, 集中度HHI={hhi_txt}, "
        f"最差年{stab.get('min_year_year')}({stab.get('min_year', 0)*100:+.2f}%) "
        "[比率越高=每年多头收益越平均稳定, 是迭代优劣核心标准]")


def _func_name_counts(expr: str) -> dict:
    """统计表达式AST中各函数名的出现次数(用于识别同函数多参数堆叠)。"""
    from . import expr_engine
    try:
        ast = expr_engine.Parser(expr_engine.tokenize(expr)).parse()
    except Exception:
        return {}
    counts = {}

    def walk(n):
        if isinstance(n, expr_engine.Call):
            counts[n.name] = counts.get(n.name, 0) + 1
            for a in n.args:
                walk(a)
        elif isinstance(n, expr_engine.Bin):
            walk(n.left); walk(n.right)
        elif isinstance(n, expr_engine.Unary):
            walk(n.x)
        elif isinstance(n, expr_engine.Ternary):
            walk(n.cond); walk(n.true); walk(n.false)
    walk(ast)
    return counts


def detect_ensemble_tendency(series: dict, last_n: int = 5) -> str:
    """
    检测优化路径近期是否反复采用"同函数不同参数相加/平均"的参数堆叠套路。
    这是思路枯竭的信号(边际增益小、易过拟合、因子间高度相关)。
    检测到则返回告诫文本(注入优化提示词), 否则返回空串。
    判据(满足其一, 且近期有效样本>=3):
      a) 近半数及以上因子在单个表达式内对同一函数调用>=2次(不同参数变体相加/平均/平滑包装);
      b) 近期因子由同一个主导函数家族反复构造(占比>=80%)。
    """
    path = series.get("path") or []
    recent = [p for p in path[-last_n:] if p.get("expr")]
    n = len(recent)
    if n < 3:
        return ""
    within = 0                      # 单表达式内同函数调用>=2次的因子数
    dominant_counter = {}
    examples = []
    for p in recent:
        counts = _func_name_counts(p["expr"])
        if not counts:
            continue
        top_func, top_cnt = max(counts.items(), key=lambda kv: kv[1])
        dominant_counter[top_func] = dominant_counter.get(top_func, 0) + 1
        if top_cnt >= 2:
            within += 1
            examples.append(f"{p.get('name', '')}({top_func}×{top_cnt})")
    if not dominant_counter:
        return ""
    dom_func, dom_n = max(dominant_counter.items(), key=lambda kv: kv[1])
    within_ratio = within / n
    dom_ratio = dom_n / n
    if not (within_ratio >= 0.5 or dom_ratio >= 0.8):
        return ""
    lines = [
        "【重要告诫·禁止参数堆叠】检测到你近期反复采用'同一函数更换参数后相加/平均'的微调方式"
        "(如 TS_CORR(...,10)+TS_CORR(...,20) 取平均、价量相关与价额相关简单叠加、对同一核心项做平滑/中值包装)。"]
    if examples:
        lines.append("  近期堆叠样例: " + "; ".join(examples[:4]) + "。")
    lines.append(
        f"  近{n}轮中主导函数 {dom_func} 出现于 {dom_n}/{n} 个因子, 单式多参数堆叠 {within}/{n} 个。"
        "这是思路枯竭的信号: 边际增益极小、易对历史区间过拟合、且因子间高度相关。"
        "【本轮硬性要求】不得再产出'同函数不同参数的加权和/平均'或'对同一核心项的平滑包装'。"
        "请跳出当前函数族, 从经济逻辑层面做实质性结构创新, 例如: "
        "①引入新的价量关系(成交分布、量价背离的加速度/动量、量价协同的波动状态); "
        "②非线性/状态条件变换(按波动或趋势状态切换逻辑, 用IF实现); "
        "③融合换手率、振幅、相对强弱等其他维度; "
        "④不同经济含义项的逻辑组合(而非同一项的参数平均)。")
    return "\n".join(lines)


def series_path_text(series: dict, max_entries: int = 30) -> str:
    """整条迭代路径: 每条含表达式 + 结构化评价摘要"""
    path = series.get("path", [])
    lines = [f"共 {len(path)} 步迭代路径(显示最近 {min(len(path), max_entries)} 步):"]
    for p in path[-max_entries:]:
        lines.append(
            f"  第{p.get('round')}轮 | {p.get('name')} | 公式: {p.get('expr')} | "
            f"假设: {(p.get('hypothesis') or '')[:60]}")
        lines.append(f"      评价: {_eval_brief(p.get('eval') or {})}")
    return "\n".join(lines)


def _diagnostics_brief(diag: dict) -> str:
    if not diag:
        return "(暂无诊断)"
    lines = []
    tc = diag.get("turnover_cost", {})
    lines.append(f"  - 换手成本: {'⚠过高' if tc.get('flag') else '达标'}"
                 f" (扣费净收益为负年份 {tc.get('n_neg_years', 0)}/{tc.get('n_years', 0)})")
    mono = diag.get("monotonicity", {})
    if mono.get("flag"):
        bad = ", ".join(f"{y}年(多头排第{d['top_rank']}, 被{('/'.join('Q'+str(q) for q in d['beat_top_groups'])) or '无'}超越)"
                        for y, d in sorted(mono.get("bad_years", {}).items()))
        lines.append(f"  - 单调性: ⚠不足, {bad}")
    else:
        lines.append("  - 单调性: 达标(每年多头组均第一)")
    stress = diag.get("stress", {})
    if stress.get("flag"):
        sp = ", ".join(f"{p['name']}(回撤{p['max_drawdown']*100:.1f}%)"
                       for p in stress.get("periods", [])
                       if p.get("max_drawdown") is not None and p["max_drawdown"] < -0.03)
        lines.append(f"  - 压力期: ⚠脆弱, {sp}")
    else:
        lines.append("  - 压力期: 达标")
    return "\n".join(lines)


def series_current_text(series: dict) -> str:
    """优化模式注入: 当前因子全况(简介+风格+评价+诊断+整条路径)"""
    best = series.get("best") or {}
    ev = best.get("eval") or {}
    parts = [
        f"【系列 {series.get('series_id')} · {series.get('name')}】",
        f"风格: {series.get('style', '')}",
        f"简介: {series.get('desc', '')}",
        f"当前最佳公式: {best.get('expr', '')}"
        + (" [评价时已取相反数翻转方向]" if best.get("flipped") else ""),
        f"当前最佳假设: {best.get('hypothesis', '')}",
        f"当前评价: {_eval_brief(ev)}",
        f"当前多头稳定性: {_stability_brief(ev.get('long_stability') or {})}",
        "当前诊断:",
        _diagnostics_brief(best.get("diagnostics") or {}),
        "",
        "【历史迭代路径】",
        series_path_text(series),
    ]
    return "\n".join(parts)


def library_summary_text(max_lessons: int = 8) -> str:
    """
    新因子挖掘防撞车注入: 已有因子库的简要信息(公式/简介/风格/假设经验/路径概览)。
    无因子时返回提示语。
    """
    library = load_library()
    if not library:
        return "(当前因子库为空, 这是首个因子系列, 可自由选择任意量价方向。)"
    blocks = [f"当前因子库已有 {len(library)} 个合格因子系列, 新因子必须与它们在题材和数值上显著区分:"]
    for s in library:
        best = s.get("best") or {}
        blocks.append("")
        blocks.append(f"━━ 系列 {s.get('series_id')} · {s.get('name')} ━━")
        blocks.append(f"风格: {s.get('style', '')}")
        blocks.append(f"简介: {s.get('desc', '')}")
        blocks.append(f"最佳公式: {best.get('expr', '')}"
                      + (" [已翻转方向]" if best.get("flipped") else ""))
        blocks.append(f"评价: {_eval_brief(best.get('eval') or {})}")
        hyps = s.get("hypotheses_tried", [])
        if hyps:
            blocks.append("曾尝试的假设:")
            for h in hyps[-max_lessons:]:
                blocks.append(f"    · {h[:100]}")
        ls = s.get("lessons_success", [])
        if ls:
            blocks.append("成功经验:")
            for t in ls[-max_lessons:]:
                blocks.append(f"    + {t[:140]}")
        lf = s.get("lessons_failure", [])
        if lf:
            blocks.append("失败教训:")
            for t in lf[-max_lessons:]:
                blocks.append(f"    - {t[:140]}")
        blocks.append("迭代路径概览:")
        for p in s.get("path", [])[-6:]:
            blocks.append(f"    第{p.get('round')}轮 {p.get('expr')} -> "
                          f"{'合格' if p.get('qualified') else '不合格'}")
    blocks.append("")
    blocks.append("避让要求: 新因子不得复用上述核心逻辑/公式结构, 应选择不同的风格大类"
                  "(如已有量价背离/反转, 可尝试动量/波动率/流动性/振幅/趋势等), "
                  "且最终合格因子会与库内因子做截面相关性检查, 相关系数过高将被拒绝。")
    return "\n".join(blocks)


# =============================================================================
# 库相关性检查(防撞车)
# =============================================================================

def check_library_correlation(factor_wide: pd.DataFrame, data, cfg,
                              exclude_id: str = "") -> dict:
    """
    新合格因子与库内各系列最佳因子做截面Spearman相关(抽样若干交易日取均值)。
    |相关| 超过 cfg.max_library_corr 判定撞车。返回 {flag, max_abs_corr, details}
    """
    from .expr_engine import compute_factor  # 延迟导入避免循环依赖

    library = load_library()
    targets = [s for s in library if s.get("series_id") != exclude_id
               and (s.get("best") or {}).get("expr")]
    if not targets:
        return {"flag": False, "max_abs_corr": 0.0, "details": [],
                "n_compared": 0}

    dates = factor_wide.index
    n = cfg.corr_sample_dates
    if len(dates) > n:
        step = max(1, len(dates) // n)
        sample_dates = dates[::step][:n]
    else:
        sample_dates = dates

    new_sub = factor_wide.loc[sample_dates]
    details = []
    for s in targets:
        best = s["best"]
        expr = best["expr"]
        try:
            lib_wide = compute_factor(expr, data, cfg)
            if best.get("flipped"):
                lib_wide = -lib_wide
            lib_sub = lib_wide.reindex(sample_dates)
            corrs = []
            for d in sample_dates:
                a = new_sub.loc[d].dropna()
                b = lib_sub.loc[d].dropna()
                common = a.index.intersection(b.index)
                if len(common) < 30:
                    continue
                c = a.loc[common].rank().corr(b.loc[common].rank())
                if np.isfinite(c):
                    corrs.append(c)
            avg = float(np.mean(corrs)) if corrs else np.nan
        except Exception as e:
            avg = np.nan
            details.append({"series_id": s.get("series_id"),
                            "name": s.get("name"), "corr": None,
                            "error": str(e)[:60]})
            continue
        details.append({"series_id": s.get("series_id"),
                        "name": s.get("name"), "corr": avg})

    abs_corrs = [abs(d["corr"]) for d in details
                 if d.get("corr") is not None and np.isfinite(d["corr"])]
    max_abs = float(max(abs_corrs)) if abs_corrs else 0.0
    return {
        "flag": bool(max_abs > cfg.max_library_corr),
        "max_abs_corr": max_abs,
        "threshold": cfg.max_library_corr,
        "n_compared": len(targets),
        "details": details,
    }
