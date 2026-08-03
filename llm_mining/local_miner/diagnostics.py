"""
本地化因子挖掘项目 - 合格因子诊断(用于优化模式与因子库结构化记录)

三项诊断(对应需求三 1.3):
1. 换手成本检查: 年化净收益 = 年化收益 - 年化双边换手率(倍) × 成本系数;
   若超过阈值比例的年份净收益为负, 判定换手率相对收益过高 -> 建议降换手。
2. 分组单调性检查: 期望因子值越高分组年收益越高(严格单调递增);
   记录多头(最高)组年收益排名不是第一的年份, 以及超越多头组的分组 -> 建议增强单调性。
3. 特殊时期压力测试: 计算多头超额收益在指定窗口(小盘股崩盘/科创板抱团等)内的
   累计收益与最大回撤 -> 提示模型该因子在极端风格下的脆弱性。

诊断结果既写入因子库系列文件(静态留存), 也由 format_diagnostics_advice 转成
条件判断式的优化建议文本注入优化模式提示词(动态判断)。
"""

import warnings

import numpy as np
import pandas as pd

import factor_analysis as fa


def _build_clean(factor_series: pd.DataFrame, data, cfg):
    """从因子长表构建 factor_analysis 的 clean 数据结构"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clean = fa.get_clean_factor_and_forward_returns(
            factor=factor_series,
            prices=data.close_long,
            periods=(cfg.ic_period,),
            quantiles=cfg.n_quantiles,
            min_stocks_per_day=cfg.min_stocks_per_day,
        )
    return clean


def turnover_cost_check(clean, cfg) -> dict:
    """
    换手成本检查。
    年化净收益 = 多头组年化收益 - 多头组年化双边换手率(倍) × turnover_cost_coef
    若净收益为负的年份占比 >= turnover_cost_neg_ratio, 判定换手率相对收益过高。
    """
    period_str = f"period_{cfg.ic_period}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group_ret = fa.calc_group_returns(clean, period=period_str, excess=True,
                                          normalize=True)
        turnover = fa.calc_group_turnover(clean, double_sided=True,
                                          period=cfg.ic_period, normalize=True)
        top = int(group_ret.columns.max())
        yearly = fa.calc_yearly_stats(group_ret, turnover, long_quantile=top,
                                      periods_per_year=252)

    top_col = f"Q{top}"
    coef = cfg.turnover_cost_coef
    yearly_detail = {}
    n_neg = 0
    for year, row in yearly.iterrows():
        annual_return = float(row[top_col]) if pd.notna(row[top_col]) else np.nan
        annual_turnover = float(row["多头年化换手率"]) if pd.notna(row["多头年化换手率"]) else np.nan
        net = annual_return - annual_turnover * coef
        if np.isfinite(net) and net < 0:
            n_neg += 1
        yearly_detail[int(year)] = {
            "年化收益": annual_return,
            "年化双边换手率_倍": annual_turnover,
            "扣费净收益": net,
        }
    n_years = len(yearly_detail)
    neg_ratio = n_neg / n_years if n_years else 0.0
    return {
        "flag": bool(neg_ratio >= cfg.turnover_cost_neg_ratio),
        "cost_coef": coef,
        "neg_ratio": neg_ratio,
        "n_neg_years": n_neg,
        "n_years": n_years,
        "threshold": cfg.turnover_cost_neg_ratio,
        "yearly": yearly_detail,
    }


def monotonicity_check(clean, cfg) -> dict:
    """
    分组单调性检查。
    按年累积分组收益(日度等效), 期望分组编号越大年收益越高(严格单调递增)。
    记录多头(最高)组年收益排名不是第一的年份, 及超越多头组的分组编号。
    """
    period_str = f"period_{cfg.ic_period}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group_ret = fa.calc_group_returns(clean, period=period_str, excess=True,
                                          normalize=True)

    top = int(group_ret.columns.max())
    n_groups = len(group_ret.columns)
    yearly_group = group_ret.groupby(group_ret.index.year).sum()

    yearly_detail = {}
    bad_years = {}
    latest_year = int(yearly_group.index.max())
    for year, row in yearly_group.iterrows():
        year = int(year)
        vals = {int(q): float(v) for q, v in row.items() if pd.notna(v)}
        if not vals:
            continue
        # 按收益从高到低排名, 1 为最高
        sorted_qs = sorted(vals.items(), key=lambda kv: kv[1], reverse=True)
        rank_of = {q: i + 1 for i, (q, _) in enumerate(sorted_qs)}
        top_rank = rank_of.get(top, n_groups)
        beat_top = sorted([q for q, v in vals.items()
                           if q != top and v > vals.get(top, -np.inf)])
        # 分组编号 与 收益排名 的 Spearman 相关(单调性强度, 越接近1越好)
        qs = pd.Series(vals)
        rank_corr = float(qs.index.to_series().astype(float).corr(qs.rank()))
        yearly_detail[year] = {
            "top_rank": top_rank,
            "n_groups": n_groups,
            "beat_top_groups": beat_top,
            "rank_corr": rank_corr,
            "group_returns": vals,
        }
        if top_rank != 1:
            bad_years[year] = {
                "top_rank": top_rank,
                "n_groups": n_groups,
                "beat_top_groups": beat_top,
                "rank_corr": rank_corr,
                "is_latest": year == latest_year,
            }
    return {
        "flag": bool(len(bad_years) > 0),
        "n_bad_years": len(bad_years),
        "bad_years": bad_years,
        "hist_bad_years": [y for y, d in bad_years.items() if not d["is_latest"]],
        "yearly": yearly_detail,
    }


def stress_period_check(clean, cfg) -> dict:
    """
    特殊时期压力测试。
    对每个配置窗口, 计算多头超额收益(日度等效)的窗口累计收益与最大回撤。
    """
    period_str = f"period_{cfg.ic_period}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        long_ret = fa.calc_long_returns(clean, period=period_str, excess=True,
                                        normalize=True)

    periods = []
    for item in cfg.stress_periods:
        name, start, end = item
        window = long_ret.loc[start:end] if end else long_ret.loc[start:]
        if len(window) == 0:
            periods.append({"name": name, "start": start, "end": end,
                            "n_days": 0, "cum_return": None, "max_drawdown": None})
            continue
        cum = float(window.sum())
        nav = (1 + window).cumprod()
        dd = nav / nav.cummax() - 1
        max_dd = float(dd.min())
        periods.append({
            "name": name, "start": start, "end": end,
            "n_days": int(len(window)),
            "cum_return": cum,
            "max_drawdown": max_dd,
        })
    # 任一窗口回撤明显(如 < -3%)即提示
    flag = any((p["max_drawdown"] is not None and p["max_drawdown"] < -0.03)
               for p in periods)
    return {"flag": bool(flag), "periods": periods}


def compute_diagnostics(factor_series: pd.DataFrame, data, cfg) -> dict:
    """对合格因子计算全部三项诊断(内部构建一次clean, 约数十秒)"""
    clean = _build_clean(factor_series, data, cfg)
    return {
        "turnover_cost": turnover_cost_check(clean, cfg),
        "monotonicity": monotonicity_check(clean, cfg),
        "stress": stress_period_check(clean, cfg),
    }


def format_diagnostics_advice(diag: dict, cfg) -> str:
    """
    把诊断结果转成条件判断式的优化建议文本(注入优化模式提示词)。
    仅对触发的问题给出针对性建议, 未触发则说明该项已达标。
    """
    lines = []

    # ---- 1. 换手成本 ----
    tc = diag.get("turnover_cost", {})
    if tc.get("flag"):
        lines.append(
            f"【换手率相对收益过高】有 {tc['n_neg_years']}/{tc['n_years']} 年"
            f"({tc['neg_ratio']*100:.0f}%, ≥{tc['threshold']*100:.0f}%阈值)的扣费净收益为负"
            f"(扣费口径: 年化收益 - 年化双边换手率×{tc['cost_coef']:.4f})。")
        worst = sorted(tc["yearly"].items(),
                       key=lambda kv: kv[1].get("扣费净收益", 0))[:3]
        for y, d in worst:
            lines.append(
                f"    {y}年: 年化收益={d['年化收益']*100:+.2f}%, "
                f"年化双边换手率={d['年化双边换手率_倍']:.1f}倍, "
                f"扣费净收益={d['扣费净收益']*100:+.2f}%")
        lines.append("    优化方向: 在保留核心信号的前提下降低换手率——延长平滑/衰减窗口、"
                     "用更慢的时序函数包裹信号、减少高频翻转, 使年化换手率与收益相匹配。")
    else:
        lines.append(f"【换手成本】达标: 仅 {tc.get('n_neg_years', 0)}/{tc.get('n_years', 0)} "
                     f"年扣费净收益为负, 换手率与收益匹配良好, 无需刻意降换手。")

    # ---- 2. 单调性 ----
    mono = diag.get("monotonicity", {})
    if mono.get("flag"):
        lines.append(
            f"【分组单调性不足】有 {mono['n_bad_years']} 个年份多头(最高)组年收益不是全场第一, "
            f"理想情况应是分组编号越大收益越高(严格单调递增)。")
        for y, d in sorted(mono["bad_years"].items()):
            beat = ", ".join(f"Q{q}" for q in d["beat_top_groups"]) or "无"
            tag = "(最新不完整年份)" if d.get("is_latest") else ""
            lines.append(
                f"    {y}年{tag}: 多头组(最高组)年收益仅排第 {d['top_rank']}/{d['n_groups']}, "
                f"被 {beat} 超越, 分组-收益秩相关={d['rank_corr']:+.2f}")
        lines.append("    优化方向: 增强因子打分的单调性——让高因子值更稳定地对应高未来收益, "
                     "可考虑对信号做更稳健的时序平滑、剔除噪声区间、或叠加强化单调性的辅助项。")
    else:
        lines.append("【分组单调性】达标: 每一年多头(最高)组年收益均为全场第一, 单调性良好。")

    # ---- 3. 压力期 ----
    stress = diag.get("stress", {})
    if stress.get("flag"):
        lines.append("【特殊时期压力测试】多头超额在以下极端风格窗口回撤明显, 因子存在风格脆弱性:")
        for p in stress["periods"]:
            if p["max_drawdown"] is None:
                lines.append(f"    {p['name']}({p['start']}~{p['end'] or '至今'}): 无数据")
                continue
            lines.append(
                f"    {p['name']}({p['start']}~{p['end'] or '至今'}, {p['n_days']}个交易日): "
                f"窗口累计超额={p['cum_return']*100:+.2f}%, 最大回撤={p['max_drawdown']*100:.2f}%")
        lines.append("    优化方向: 针对上述脆弱期(小盘股流动性危机/科创板抱团等极端风格), "
                     "考虑增强因子对市值/风格切换的鲁棒性, 或叠加风险状态过滤项。")
    else:
        lines.append("【特殊时期压力测试】达标: 各极端风格窗口多头超额回撤均在可接受范围内。")

    return "\n".join(lines)
