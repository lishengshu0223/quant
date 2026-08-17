"""
本地化因子挖掘项目 - 因子语义相关性评审(事后防撞车)

设计(对应需求: 一个管挖掘, 一个做评判):
- 新因子在数值截面相关性检验之前, 先经一个独立的 LLM 对话窗口做语义评审,
  判断它与库内已有因子的经济逻辑/公式结构是否本质相同。
- 评审重点抓"换皮造因子": 同一量价关系换价格字段($close/$open/$high/$low/$vwap)、
  同一核心逻辑改窗口/做平滑包装、同一逻辑的线性组合 —— 这些数值相关可能不高,
  但语义上就是同一个因子。
- 语义评审通过后才进入数值截面相关性检验(ρ≤cfg.max_library_corr), 两道关都过才入库。
- 评审调用失败时降级: 不阻塞流程, 仅警告并交由数值检验把关(数值检验是硬门槛)。
"""

from . import console
from .llm_client import call_llm, extract_json

REVIEW_SYSTEM = """你是一位严格的量化因子评审专家, 专职判断两个A股量价因子的"语义相关性"。
你的职责是防止"换皮造因子": 挖掘模型可能通过微小改动伪装成新因子, 你在语义/经济逻辑层面识别它们。
识别重点(视为本质相同):
1. 同一量价关系, 仅互换价格字段: 如 TS_CORR($close,$volume,n) 与 TS_CORR($low/$open/$high/$vwap,$volume,n);
2. 同一核心逻辑, 仅改变回看窗口 n 或做平滑/中值/加权/取反包装(如 TS_MEAN(核心项,3)、EMA(核心项,5));
3. 同一逻辑的线性组合/平均(如 两个同族相关性取均值);
4. 对同一信号换一个等价表达(如用 amount 代替 volume 表达同一资金流逻辑)。
只有换到不同的经济逻辑大类(动量/反转/波动率/流动性/资金流/微观结构/趋势等)且核心函数与字段组合
本质上不同, 才算是真正的新因子。"""


def build_review_prompt(candidate: dict, library: list) -> str:
    """构建评审提示词: 候选因子 vs 库内各系列最佳因子"""
    lines = [
        "请评审以下候选因子是否与因子库中已有因子在语义/经济逻辑上本质相同。",
        "",
        "【候选因子】",
        f"名称: {candidate.get('name', '')}",
        f"风格: {candidate.get('style', '')}",
        f"公式: {candidate.get('expr', '')}",
        f"简介: {candidate.get('desc', '')}",
        "",
        f"【因子库已有 {len(library)} 个系列】",
    ]
    for s in library:
        best = s.get("best") or {}
        lines.append("")
        lines.append(f"系列 {s.get('series_id')} · {s.get('name')}")
        lines.append(f"  风格: {s.get('style', '')}")
        lines.append(f"  公式: {best.get('expr', '')}")
        lines.append(f"  简介: {(best.get('desc') or s.get('desc') or '')[:200]}")
    lines.append("")
    lines.append(
        "请输出一个JSON对象(不要解释性文字), 结构如下:\n"
        "{\n"
        '  "判断": "不相关" 或 "相似" 或 "重复",\n'
        '  "最相似因子": "最相似的库内系列ID, 如 D001; 若都不相关则为空",\n'
        '  "相似度评分": 0到1之间的小数(1=完全相同, 0=完全不同),\n'
        '  "理由": "一句话说明判定依据, 明确指出命中了哪类换皮模式(若有)"\n'
        "}\n"
        "判定标准: 只要命中上述任一换皮模式(换价格字段/换窗口/包装/线性组合/等价表达), "
        "就应判为'重复'或'相似'; 只有经济逻辑大类与核心函数结构都不同才可判'不相关'。"
    )
    return "\n".join(lines)


def review_factor(cfg, candidate: dict, library: list) -> dict:
    """
    对候选因子做语义相关性评审(独立 LLM 对话窗口)。
    返回 {"reject": bool, "match_series": str, "similarity": float, "reason": str, "error": str|None}
    """
    if not library:
        return {"reject": False, "match_series": "", "similarity": 0.0,
                "reason": "因子库为空", "error": None}
    prompt = build_review_prompt(candidate, library)
    messages = [
        {"role": "system", "content": REVIEW_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    try:
        console.log("    [语义评审] 独立对话窗口评审中(判断与库内因子是否语义重复)...")
        resp = call_llm(messages, cfg)
        parsed = extract_json(resp["content"])
    except Exception as e:
        console.log(f"    [警告] 语义评审调用失败({e}), 降级跳过语义评审, 仅靠数值相关性检验把关。")
        return {"reject": False, "match_series": "", "similarity": None,
                "reason": f"评审失败: {e}", "error": str(e)}

    verdict = str(parsed.get("判断") or "")
    sim = parsed.get("相似度评分")
    try:
        sim = float(sim) if sim is not None else None
    except (TypeError, ValueError):
        sim = None
    match = str(parsed.get("最相似因子") or "")
    reason = str(parsed.get("理由") or "")

    # 判定: "重复" 一律拒绝; "相似" 且评分达到阈值也拒绝
    reject = (verdict == "重复") or (
        verdict == "相似" and sim is not None and sim >= cfg.review_similar_threshold)
    return {"reject": bool(reject), "match_series": match, "similarity": sim,
            "reason": reason, "error": None}
