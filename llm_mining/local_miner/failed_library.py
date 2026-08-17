"""
本地化因子挖掘项目 - 失败因子库管理

设计目标(最少储存占用存储最多信息, 便于用户与大模型了解挖掘历程):
- 每次挖掘的每一轮, 把该轮所有不合格因子的表达式与失败理由持久化(体积精简, 不存完整eval);
- 每100轮战役结束后, 用"规则统计 + LLM概括"生成战役级总结(尝试了哪些方向、哪些方面不行);
- 下次挖掘首轮注入: 成功因子库摘要(factor_library) + 历史战役失败经验(本模块)。

目录结构:
    factor_library/failed/<campaign_key>/
        round_XXX.json            每轮失败因子明细(表达式+失败理由+关键指标)
        round_XXX_summary.json    该轮失败总结(规则生成一句话)
        campaign_summary.json     战役总结(规则统计文本 + LLM概括, LLM失败自动降级)
        index.json                全部战役的注入文本索引(仅保留最近 N 个战役的摘要)
"""

import datetime
import glob
import json
import os
import time
from collections import Counter

from .config import FACTOR_LIBRARY_DIR, ensure_workspace

FAILED_DIR = os.path.join(FACTOR_LIBRARY_DIR, "failed")
INDEX_FILE = os.path.join(FAILED_DIR, "index.json")
MAX_INDEX_CAMPAIGNS = 4        # index.json 最多保留最近几个战役的注入摘要
LLM_SAMPLE_PER_REASON = 4      # 给 LLM 概括时, 每种失败原因取的代表样本数


# =============================================================================
# 路径与基础工具
# =============================================================================

def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def campaign_dir(campaign_key: str) -> str:
    return os.path.join(FAILED_DIR, campaign_key)


def load_index() -> dict:
    """读取失败库注入索引 {campaign_key: {ts, frequency, summary_text}}"""
    if not os.path.exists(INDEX_FILE):
        return {"campaigns": []}
    try:
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"campaigns": []}


def save_index(index: dict):
    ensure_workspace()
    os.makedirs(FAILED_DIR, exist_ok=True)
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)


# =============================================================================
# 失败原因分类(从 eval 提取, 保持与 factor_eval/run_mining 判定一致)
# =============================================================================

def failure_reason(ev: dict) -> tuple:
    """从评价结果分类失败原因。返回 (原因, 细节描述)。合格/评价失败同样归类。"""
    if not ev:
        return "无评价", ""
    if ev.get("error"):
        return "计算失败", str(ev["error"])[:100]
    if ev.get("qualified"):
        return "合格", ""
    if ev.get("review_rejected"):
        return "语义评审拒绝", str(ev.get("review_reason") or "")[:100]
    if ev.get("library_rejected"):
        return "相关性撞车", (f"max|ρ|={ev.get('library_max_corr'):.2f}"
                             if ev.get("library_max_corr") is not None else "")
    mg = ev.get("monotonicity_grade") or {}
    if not ev.get("direction_ok"):
        return "方向无效", (f"IC={ev.get('ic_mean', 0)*100:+.2f}%, "
                           f"多头={ev.get('long_total', 0)*100:+.1f}%(IC与多头矛盾或为负)")
    if mg.get("grade") == "C":
        neg = mg.get("neg_years") or []
        peak = (mg.get("yearly_peak") or 0) * 100
        return "单调性C级", (f"秩相关={mg.get('drc', 0):.3f}, 年度越序峰值={peak:.1f}%"
                            + (f", 负年={neg}" if neg else ""))
    if not ev.get("monthly_ok"):
        return "月度不足", f"月度正占比={ev.get('monthly_pos_ratio', 0)*100:.0f}%(<60%)"
    if not ev.get("yearly_ok"):
        bad = sorted((ev.get("bad_hist_years") or {}))
        return "年度为负", f"历史完整年份多头为负: {bad}"
    return "其他未达标", (f"IC={ev.get('ic_mean', 0)*100:+.2f}%, "
                         f"月度占比={ev.get('monthly_pos_ratio', 0)*100:.0f}%, "
                         f"多头累计={ev.get('long_total', 0)*100:+.1f}%")


def _brief_factor(f: dict) -> dict:
    """单个因子的精简记录(表达式 + 失败理由 + 关键指标), 不存完整 eval 以控制体积"""
    ev = f.get("eval") or {}
    reason, detail = failure_reason(ev)
    mg = ev.get("monotonicity_grade") or {}
    return {
        "name": f.get("name", ""),
        "expr": f.get("expr", ""),
        "style": f.get("style", ""),
        "reason": reason,
        "detail": detail,
        "ic_mean": round(ev.get("ic_mean", 0), 5) if ev.get("ic_mean") is not None else None,
        "long_total": round(ev.get("long_total", 0), 5) if ev.get("long_total") is not None else None,
        "monthly_pos_ratio": round(ev.get("monthly_pos_ratio", 0), 4) if ev.get("monthly_pos_ratio") is not None else None,
        "grade": mg.get("grade"),
        "neg_years": mg.get("neg_years") or [],
    }


# =============================================================================
# 每轮沉淀
# =============================================================================

def append_round(campaign_key: str, round_no: int, hypothesis: str,
                 reflection: str, factors: list) -> dict | None:
    """把一轮的因子明细沉淀进失败库(仅不合格因子), 并生成该轮一句话总结。
    返回轮次记录; 无因子或全部合格时仍写轮次文件(便于了解历程)。"""
    cdir = campaign_dir(campaign_key)
    os.makedirs(cdir, exist_ok=True)
    rec = {
        "round": round_no,
        "hypothesis": (hypothesis or "")[:200],
        "reflection": (reflection or "")[:200],
        "factors": [_brief_factor(f) for f in (factors or []) if f.get("eval")],
    }
    # 该轮失败总结(规则一句话)
    reasons = {}
    for bf in rec["factors"]:
        reasons[bf["reason"]] = reasons.get(bf["reason"], 0) + 1
    n_total = len(rec["factors"])
    n_fail = n_total - reasons.get("合格", 0)
    summary = {
        "round": round_no,
        "n_factors": n_total,
        "n_failed": n_fail,
        "reasons": reasons,
        "text": (f"第{round_no}轮共{n_total}个因子" + (f"，{n_fail}个不合格" if n_fail else "")
                 + (f"，失败原因分布: {', '.join(f'{k}×{v}' for k, v in reasons.items() if k != '合格')}"
                    if reasons else "") + "。"),
    }
    with open(os.path.join(cdir, f"round_{round_no:03d}.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    with open(os.path.join(cdir, f"round_{round_no:03d}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    return rec


# =============================================================================
# 战役级总结: 规则统计
# =============================================================================

def _rule_statistics(cdir: str, adopted: list = None) -> dict:
    """扫描战役目录内全部轮次文件, 统计失败模式/高频结构/有效方向"""
    reason_cnt = Counter()
    struct_cnt = Counter()
    valid_dirs = []           # IC>0.02 且多头>0.15 但未合格的探索方向
    n_factors = n_failed = 0
    for path in sorted(glob.glob(os.path.join(cdir, "round_*.json"))):
        if "_summary" in os.path.basename(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except Exception:
            continue
        for bf in rec.get("factors", []):
            n_factors += 1
            if bf["reason"] != "合格":
                n_failed += 1
                reason_cnt[bf["reason"]] += 1
                # 公式核心结构(去掉参数与窗口, 近似去重)
                expr = str(bf.get("expr") or "")
                core = expr.split(",")[0].replace("(", "").replace(")", "").strip()
                if len(core) > 4:
                    struct_cnt[core] += 1
            else:
                reason_cnt["合格"] += 1
            ic, lt = bf.get("ic_mean"), bf.get("long_total")
            if bf["reason"] not in ("合格", "计算失败", "无评价") \
                    and ic is not None and lt is not None \
                    and ic > 0.02 and lt > 0.15:
                valid_dirs.append(f"{bf['name']}(IC{ic*100:+.1f}% 多头{lt*100:+.0f}% "
                                  f"→{bf['reason']})")
    return {
        "n_factors": n_factors,
        "n_failed": n_failed,
        "reason_dist": dict(reason_cnt),
        "top_structures": [f"{k}({v}次)" for k, v in struct_cnt.most_common(8)],
        "valid_directions": valid_dirs[:12],
        "adopted": adopted or [],
    }


def _rule_summary_text(stats: dict, frequency: str, rounds: int) -> str:
    """把统计结果组装为规则摘要文本(LLM 概括失败时的降级文本, 也是 LLM 输入素材)"""
    lines = [
        f"【失败因子库·{frequency}战役 {rounds}轮统计】累计 {stats['n_factors']} 个因子, "
        f"其中不合格 {stats['n_failed']} 个。",
        f"1. 失败原因分布: {', '.join(f'{k}×{v}' for k, v in stats['reason_dist'].items()) if stats['reason_dist'] else '无'}。",
    ]
    if stats["top_structures"]:
        lines.append(f"2. 高频被拒结构(勿再重复探索): {', '.join(stats['top_structures'])}。")
    if stats["valid_directions"]:
        lines.append(f"3. 已探索但未达合格的有效方向(IC与多头为正, 卡在{'; '.join(set(f.split('→')[-1].rstrip(')') for f in stats['valid_directions']))}): "
                     f"{'、'.join(stats['valid_directions'])}。")
    lines.append("4. 经验教训: 换字段/换窗口/加包装的等价因子会被语义评审与相关性检验拦截, "
                 "新因子必须来自不同的经济逻辑; 历史完整年份秩相关为负会直接判C级。")
    return "\n".join(lines)


# =============================================================================
# 战役级总结: LLM 概括(失败自动降级)
# =============================================================================

def _collect_samples(cdir: str, per_reason: int = LLM_SAMPLE_PER_REASON) -> list:
    """从战役目录抽样代表失败因子(每类失败原因取前 N 条), 供 LLM 概括。"""
    samples = []   # [(expr, reason, detail)]
    seen = Counter()
    for path in sorted(glob.glob(os.path.join(cdir, "round_*.json"))):
        if "_summary" in os.path.basename(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except Exception:
            continue
        for bf in rec.get("factors", []):
            if bf["reason"] in ("合格", "计算失败", "无评价"):
                continue
            if seen[bf["reason"]] >= per_reason:
                continue
            seen[bf["reason"]] += 1
            samples.append((bf.get("expr", ""), bf["reason"], bf.get("detail", "")))
    return samples


def _llm_summarize(cfg, rule_text: str, frequency: str, samples: list) -> tuple:
    """调用 LLM 做战役概括。返回 (summary_text, ok)。失败时 ok=False。"""
    try:
        from . import llm_client
        sample_block = ""
        if samples:
            sample_lines = [f"- {e}  ({r}: {d})" for e, r, d in samples]
            sample_block = "代表失败因子样本(勿再重复同类公式):\n" + "\n".join(sample_lines)
        prompt = (
            f"你是量化因子挖掘的战役复盘助手。以下是最近一次频率={frequency}的因子挖掘战役的规则统计:"
            f"\n{rule_text}\n"
            f"\n{sample_block}\n"
            f"\n请用简洁中文概括该战役: ①主要尝试了哪些方向的因子; "
            f"②这些方向分别在哪些方面不行(IC、单调性、月度稳定性、年度、语义重复等); "
            f"③哪些方向仍有潜力、下一步建议优先探索什么。控制在300字以内, 分条列出。"
        )
        resp = llm_client.call_llm([{"role": "user", "content": prompt}], cfg)
        content = ((resp or {}).get("content") or "").strip()
        if not content:
            return None, False
        return content, True
    except Exception as e:
        return f"(LLM概括失败: {type(e).__name__}: {e})", False


def cfg_campaign_key(cfg, round_no: int = 0) -> str:
    """根据配置生成战役标识: 频率_起始日期(如 minute_20260808)"""
    freq = getattr(cfg, "data_frequency", "daily")
    start = getattr(cfg, "eval_start_date", "2018-01-01")[:10].replace("-", "")
    return f"{freq}_{start}"


# =============================================================================
# 战役完成: 生成 campaign_summary + 更新 index
# =============================================================================

def finalize_campaign(cfg, rounds: int, adopted: list = None) -> dict | None:
    """战役结束(跑满轮数)时: 汇总失败库统计, 生成战役总结(规则+LLM)并更新注入索引。
    返回 campaign_summary dict; 战役目录不存在(无失败沉淀)时返回 None。"""
    key = cfg_campaign_key(cfg)
    cdir = campaign_dir(key)
    if not os.path.isdir(cdir):
        return None
    stats = _rule_statistics(cdir, adopted or [])
    rule_text = _rule_summary_text(stats, getattr(cfg, "data_frequency", "daily"), rounds)
    samples = _collect_samples(cdir)
    llm_text, llm_ok = _llm_summarize(cfg, rule_text,
                                      getattr(cfg, "data_frequency", "daily"), samples)
    summary = {
        "campaign": key,
        "frequency": getattr(cfg, "data_frequency", "daily"),
        "rounds": rounds,
        "adopted": adopted or [],
        "stats": stats,
        "rule_text": rule_text,
        "llm_summary": llm_text,
        "llm_ok": llm_ok,
        "created_at": _now(),
    }
    with open(os.path.join(cdir, "campaign_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    # 更新注入索引(只保留最近 N 个)
    index = load_index()
    index["campaigns"] = [c for c in index.get("campaigns", [])
                          if c.get("campaign") != key]
    index["campaigns"].append({
        "campaign": key,
        "ts": _now(),
        "frequency": summary["frequency"],
        "summary_text": (summary.get("llm_summary") or "") + "\n" + rule_text,
    })
    index["campaigns"] = index["campaigns"][-MAX_INDEX_CAMPAIGNS:]
    save_index(index)
    return summary


# =============================================================================
# 注入文本(下次挖掘首轮)
# =============================================================================

def injection_text(max_campaigns: int = MAX_INDEX_CAMPAIGNS) -> str:
    """历史战役失败经验注入文本(供新挖掘首轮提示词)。无记录时返回空串。"""
    index = load_index()
    camps = index.get("campaigns", [])[-max_campaigns:]
    if not camps:
        return ""
    blocks = ["【历史挖掘失败经验(防止重复探索, 务必先读)】"]
    for c in camps:
        blocks.append(f"━━ 战役@{c.get('ts', '')[:16]} ({c.get('frequency', '')}) ━━")
        blocks.append(c.get("summary_text", ""))
    return "\n\n".join(blocks)


# =============================================================================
# 战役信息压缩 —— 每100轮战役结束且有因子入库后, 将整体历史压缩为概括性摘要,
# 供下一个战役首轮注入(避免长期逐轮记忆导致上下文过长、稀释模型注意力)。
# 原属 prompts.py, 2026-08 重构时并入本模块(与 finalize_campaign/injection_text 同为战役级记忆)。
# =============================================================================

CAMPAIGN_SUMMARY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "workspace", "campaign_summaries.json")


def summarize_campaign(state: dict, cfg, library: list) -> str:
    """把已完成战役(history 全部轮次 + 黑名单)压缩为概括性摘要文本。
    library: factor_library.load_library() 结果, 用于列出本次战役入库成果。"""
    history = state.get("history") or []
    adopted = state.get("campaign_adopted") or []
    rounds = len(history)
    # 1) 入库成果
    adopted_lines = []
    for sid in adopted:
        s = next((x for x in library if x.get("series_id") == sid), None)
        if s:
            best = s.get("best") or {}
            adopted_lines.append(
                f"  - {sid} {s.get('name')}: {str(best.get('expr') or '')[:80]}")
    adopted_txt = "\n".join(adopted_lines) if adopted_lines else "  (本战役无入库因子)"
    # 2) 失败模式统计(从 history 全量 eval 统计)
    reason_cnt = Counter()          # 失败原因分类计数
    struct_cnt = Counter()          # 被拒因子核心结构(公式)高频重复
    for rec in history:
        for f in rec.get("factors", []):
            ev = f.get("eval") or {}
            if not ev:
                continue
            if ev.get("error"):
                reason_cnt["计算失败"] += 1
            elif ev.get("review_rejected"):
                reason_cnt["语义评审拒绝"] += 1
            elif ev.get("library_rejected"):
                reason_cnt["相关性撞车"] += 1
            else:
                mg = ev.get("monotonicity_grade") or {}
                if mg.get("grade") == "C":
                    reason_cnt["单调性C级"] += 1
                elif not ev.get("qualified"):
                    reason_cnt["其他未达标"] += 1
            # 公式核心结构: 去掉参数与窗口, 近似去重
            expr = str(f.get("expr") or "")
            core = expr.split(",")[0].replace("(", "").replace(")", "").strip()
            struct_cnt[core] += 1
    top_struct = "、".join(f"{k}({v}次)" for k, v in struct_cnt.most_common(8))
    # 3) 有效方向: IC>0 且多头为正 但未达合格的探索
    ok_dir = []
    for rec in history:
        for f in rec.get("factors", []):
            ev = f.get("eval") or {}
            if ev and not ev.get("qualified") and not ev.get("error") \
                    and (ev.get("ic_mean") or 0) > 0.02 and (ev.get("long_total") or 0) > 0.15:
                ok_dir.append(f"{f.get('name')}(IC{(ev['ic_mean']*100):+.1f}% 多头{(ev['long_total']*100):+.0f}%)")
    ok_dir_txt = "、".join(ok_dir[:12]) or "无"
    summary = (
        f"【战役信息压缩摘要·频率={cfg.data_frequency}】本轮战役共 {rounds} 轮, "
        f"入库 {len(adopted)} 个新因子系列。\n"
        f"1. 入库成果:\n{adopted_txt}\n"
        f"2. 失败模式分布(累计): {', '.join(f'{k}{v}次' for k, v in reason_cnt.items()) or '无'}。\n"
        f"   其中高频被拒结构(勿再探索): {top_struct}。\n"
        f"3. 已探索但未达合格的有效方向(IC与多头为正, 卡在单调性/年度): {ok_dir_txt}。\n"
        f"4. 经验教训: 换字段/换窗口的包装会被语义评审与相关性检验拦截, 新因子必须来自不同的经济逻辑; "
        f"最新不完整年份不参与单调性负年判定, 历史完整年份秩相关为负仍会直接判C级。"
    )
    return summary


def save_campaign_summary(summary_text: str):
    """把战役压缩摘要追加保存(供下个战役首轮注入)"""
    try:
        recs = []
        if os.path.exists(CAMPAIGN_SUMMARY_FILE):
            with open(CAMPAIGN_SUMMARY_FILE, "r", encoding="utf-8") as f:
                recs = json.load(f)
        recs.append({"ts": time.strftime("%Y-%m-%d %H:%M"), "summary": summary_text})
        # 只保留最近 3 次战役摘要, 防止无限增长
        recs = recs[-3:]
        os.makedirs(os.path.dirname(CAMPAIGN_SUMMARY_FILE), exist_ok=True)
        with open(CAMPAIGN_SUMMARY_FILE, "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=1)
        return True
    except Exception:
        return False


def load_campaign_summaries() -> str:
    """读取历史战役压缩摘要(最近3次), 拼接为注入文本"""
    if not os.path.exists(CAMPAIGN_SUMMARY_FILE):
        return ""
    try:
        with open(CAMPAIGN_SUMMARY_FILE, "r", encoding="utf-8") as f:
            recs = json.load(f)
        blocks = [f"── 战役@{r.get('ts')} ──\n{r.get('summary')}" for r in recs if r.get("summary")]
        return "\n\n".join(blocks)
    except Exception:
        return ""
