"""
本地化因子挖掘项目 - 因子评价(复用项目自有的 factor_analysis 模块)

合格标准(对应需求3.1):
1. IC = 5日 RankIC; 分组数默认10; 多头 = 因子值最高组; 多头收益 = 多头组相对全市场均值的超额
2. IC均值与多头累计收益必须同向为正(双负自动取反翻转; 一正一负直接剔除)
3. 月度IC均值 > 60% 的月份为正
4. 分年度多头超额收益: 除最新不完整年份外, 每个历史完整年份都必须为正

收益量纲统一: n日前向收益在时间轴上重叠, 直接按年求和/累加会放大n倍。
本模块所有收益指标均启用 normalize=True, 即 n日前向收益÷n 转为日度等效收益,
使任意周期(ic_period取任何值)的收益都可比, 便于观察信号衰减对收益的影响。
"""

import warnings

import numpy as np
import pandas as pd

import factor_analysis as fa

from . import console


def _compute_metrics(factor_series: pd.DataFrame, data, cfg) -> dict:
    """对因子(长表Series)计算全部评价指标"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clean = fa.get_clean_factor_and_forward_returns(
            factor=factor_series,
            prices=data.close_long,
            periods=(cfg.ic_period,),
            quantiles=cfg.n_quantiles,
            min_stocks_per_day=cfg.min_stocks_per_day,
        )
        ic_df = fa.calc_information_coefficient(clean, method="spearman")
        # normalize=True: n日前向收益÷n 转日度等效, 消除重叠收益的n倍放大, 跨周期可比
        long_ret = fa.calc_long_returns(clean, period=f"period_{cfg.ic_period}",
                                        excess=True, normalize=True)
        group_ret = fa.calc_group_returns(clean, period=f"period_{cfg.ic_period}",
                                          excess=True, normalize=True)

    period_col = f"period_{cfg.ic_period}"
    ic = ic_df[period_col].dropna()
    ic_mean = float(ic.mean()) if len(ic) else np.nan
    ic_std = float(ic.std(ddof=1)) if len(ic) > 1 else np.nan
    icir = ic_mean / ic_std if ic_std and ic_std > 0 else np.nan
    long_total = float(long_ret.sum()) if len(long_ret) else np.nan
    long_annual = float(long_ret.mean() * 252) if len(long_ret) else np.nan

    # 月度IC
    monthly_ic = ic.resample("ME").mean().dropna()
    n_months = int(len(monthly_ic))
    n_pos_months = int((monthly_ic > 0).sum())
    monthly_pos_ratio = n_pos_months / n_months if n_months else np.nan
    neg_months = [str(d.date())[:7] for d in monthly_ic[monthly_ic <= 0].index]

    # 分年度多头超额收益
    yearly_long = long_ret.groupby(long_ret.index.year).sum()
    yearly_dict = {int(y): float(v) for y, v in yearly_long.items()}
    latest_year = max(yearly_dict) if yearly_dict else None
    bad_hist_years = {y: v for y, v in yearly_dict.items()
                      if y != latest_year and v <= 0}

    # 分组年化收益(观察单调性)
    group_annual = {int(q): float(group_ret[q].mean() * 252) for q in group_ret.columns}

    return {
        "ic_mean": ic_mean, "ic_std": ic_std, "icir": icir,
        "long_total": long_total, "long_annual": long_annual,
        "n_ic_days": int(len(ic)),
        "n_months": n_months, "n_pos_months": n_pos_months,
        "monthly_pos_ratio": monthly_pos_ratio, "neg_months": neg_months,
        "yearly_long": yearly_dict, "latest_year": latest_year,
        "bad_hist_years": bad_hist_years,
        "group_annual": group_annual,
    }


def evaluate_factor(factor_wide: pd.DataFrame, data, cfg, name: str = "") -> dict:
    """
    评价单个因子, 返回结构化结果(含各项达标判定)。
    factor_wide: 宽表(日期×股票)
    """
    try:
        f = factor_wide[factor_wide.index >= pd.Timestamp(cfg.eval_start_date)]
        series = f.stack(future_stack=True).dropna()
        if len(series) < 5000:
            return {"name": name, "error": f"有效样本过少({len(series)}条), 无法评价"}
        series.index.names = ["date", "code"]
        series.name = "factor"

        metrics = _compute_metrics(series, data, cfg)
        flipped = False
        ic_mean, long_total = metrics["ic_mean"], metrics["long_total"]

        # 方向处理: 双负 -> 取相反数重新计算; 一正一负 -> 差因子
        if ic_mean < 0 and long_total < 0:
            flipped = True
            metrics = _compute_metrics(-series, data, cfg)
            ic_mean, long_total = metrics["ic_mean"], metrics["long_total"]

        direction_ok = bool(ic_mean > 0 and long_total > 0)
        monthly_ok = bool(metrics["monthly_pos_ratio"] is not np.nan
                          and metrics["monthly_pos_ratio"] >= cfg.monthly_ic_pos_ratio)
        yearly_ok = bool(len(metrics["bad_hist_years"]) == 0)
        qualified = bool(direction_ok and monthly_ok and yearly_ok)

        result = {
            "name": name, "error": None, "flipped": flipped,
            "direction_ok": direction_ok,
            "monthly_ok": monthly_ok,
            "monthly_require": cfg.monthly_ic_pos_ratio,
            "yearly_ok": yearly_ok,
            "qualified": qualified,
        }
        result.update(metrics)
        result["n_neg_months"] = len(metrics["neg_months"])
        return result

    except Exception as e:
        return {"name": name, "error": f"{type(e).__name__}: {e}", "qualified": False}


def build_feedback_text(round_factors: list, cfg) -> str:
    """
    把一轮所有因子的评价结果组织成给模型的中文反馈文本。
    round_factors: [{"name","desc","expr","eval": {...}}, ...]
    """
    lines = []
    for i, f in enumerate(round_factors, 1):
        ev = f.get("eval") or {}
        lines.append(f"━━ 因子{i}: {f.get('name')} ━━")
        lines.append(f"公式: {f.get('expr')}")
        if ev.get("error"):
            lines.append(f"结果: 计算或评价失败 -> {ev['error']}")
            lines.append("建议: 检查公式语法、函数参数与除零保护, 避免使用未声明的函数。")
            lines.append("")
            continue
        lines.append(
            f"结果: IC均值={ev['ic_mean']*100:+.3f}%, ICIR={ev['icir']:.3f}, "
            f"多头累计超额={ev['long_total']*100:+.2f}%, 多头年化超额={ev['long_annual']*100:.2f}%"
            f" (收益为日度等效口径: {cfg.ic_period}日前向收益÷{cfg.ic_period}, 跨周期可比)"
            + (" [因子已自动取相反数翻转方向]" if ev["flipped"] else "")
        )
        # 标准a
        if ev["direction_ok"]:
            lines.append(f"[达标] 方向检验: IC均值与多头收益同向为正。")
        else:
            lines.append(f"[未达标] 方向检验: IC均值={ev['ic_mean']*100:+.3f}% 与多头累计收益="
                         f"{ev['long_total']*100:+.2f}% 方向矛盾或仍为负, 该因子无效。"
                         "说明因子逻辑与预期相反或根本无效, 应彻底更换假设而非微调参数。")
        # 标准b
        if ev["monthly_ok"]:
            lines.append(f"[达标] 月度IC为正占比: {ev['monthly_pos_ratio']*100:.1f}% "
                         f"(≥{cfg.monthly_ic_pos_ratio*100:.0f}%)。")
        else:
            examples = ", ".join(ev["neg_months"][:5])
            lines.append(f"[未达标] 月度IC为正占比仅 {ev['monthly_pos_ratio']*100:.1f}% "
                         f"(要求≥{cfg.monthly_ic_pos_ratio*100:.0f}%), 共{ev['n_neg_months']}个月为负"
                         f"(如 {examples})。说明因子稳定性不足, 可考虑更稳健的平滑/标准化或更换信号。")
        # 标准c
        if ev["yearly_ok"]:
            lines.append("[达标] 分年度多头超额: 历史完整年份全部为正。")
        else:
            bad = ", ".join(f"{y}年({v*100:+.2f}%)" for y, v in sorted(ev["bad_hist_years"].items()))
            lines.append(f"[未达标] 分年度多头超额存在为负的历史完整年份: {bad}。"
                         "说明因子在某些市场风格下失效, 需增强风格适应性或避开该逻辑。")
        if ev["qualified"]:
            lines.append("★★ 该因子已通过全部合格标准! 可在此基础上进一步抬高IC或验证稳健性。")
        lines.append("")

    lines.append("综合要求: 下一轮因子必须同时满足 (a)IC均值与多头收益同向为正 "
                 f"(b)月度IC为正占比≥{cfg.monthly_ic_pos_ratio*100:.0f}% "
                 "(c)历史完整年份多头超额全部为正。请基于以上反馈进行有针对性的改进, "
                 "不要简单重复已失败的公式。")
    return "\n".join(lines)


def to_serializable(obj):
    """把评价结果转成可JSON序列化的结构"""
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
