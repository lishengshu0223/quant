"""
本地化因子挖掘项目 - 因子评价(复用项目自有的 factor_analysis 模块)

合格标准(对应需求3.1):
1. IC = 5日 RankIC; 分组数默认10; 多头 = 因子值最高组; 多头收益 = 多头组相对全市场均值的超额
2. IC均值与多头累计收益必须同向为正(双负自动取反翻转; 一正一负直接剔除)
3. 月度IC均值 > 60% 的月份为正
4. 分年度多头超额收益: 除最新不完整年份外, 每个历史完整年份都必须为正
5. 分组单调性评级不得为C级(四级: S优秀/A良好/B可入库需优化/C丢弃;
   任一秩相关为负的年份直接判C, C级一律拒绝不入库不采纳)

收益量纲统一: n日前向收益在时间轴上重叠, 直接按年求和/累加会放大n倍。
本模块所有收益指标均启用 normalize=True, 即 n日前向收益÷n 转为日度等效收益,
使任意周期(ic_period取任何值)的收益都可比, 便于观察信号衰减对收益的影响。
"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats

import factor_analysis as fa

from . import console


def _compute_metrics(factor_series: pd.DataFrame, data, cfg) -> dict:
    """对因子(长表Series)计算全部评价指标"""
    # 交易资格遮盖: 剔除 ST/停牌/上市未满一年/涨跌停 股票(因子值置 NaN, 不参与IC与分组收益),
    # 避免因子收益因不可交易股票而虚高
    if getattr(data, "tradable", None) is not None and not data.tradable.empty:
        tradable_long = data.tradable.stack(future_stack=True)
        mask = tradable_long.reindex(factor_series.index).fillna(False).astype(bool)
        factor_series = factor_series[mask]

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

    # 分组收益单调性两个量化指标(每日秩相关均值 + 越序惩罚)
    group_monotonicity = calc_group_monotonicity(group_ret, cfg)

    return {
        "ic_mean": ic_mean, "ic_std": ic_std, "icir": icir,
        "long_total": long_total, "long_annual": long_annual,
        "n_ic_days": int(len(ic)),
        "n_months": n_months, "n_pos_months": n_pos_months,
        "monthly_pos_ratio": monthly_pos_ratio, "neg_months": neg_months,
        "yearly_long": yearly_dict, "latest_year": latest_year,
        "bad_hist_years": bad_hist_years,
        "group_annual": group_annual,
        "group_monotonicity": group_monotonicity,
    }


def calc_long_stability(yearly_long: dict, latest_year) -> dict:
    """
    多头收益年度稳定性指标(静态判断)。

    设计动机: 因子迭代"更好"的特征不是IC更高(IC常由空头贡献), 而是多头收益更稳——
    每个历史完整年份都有正的多头收益, 且收益在年份间分布平均, 不集中于某一两年。
    因此仅基于历史完整年份(剔除最新不完整年份, 与合格标准c一致)计算:
    - yearly_ir: 年度多头收益信息比率 = 年均值/年标准差(=1/变异系数), 越高越稳且越高(主比较量)
    - cv:        变异系数 = 年标准差/年均值, 越低越平均
    - hhi:       年度收益贡献集中度(Herfindahl指数), 越低越分散(越不集中于个别年份)
    - min_year:  最差完整年份的多头收益, 越高底部越稳(每年都有正收益的体现)
    score 取 yearly_ir(完美平均、标准差为0时封顶99), 作为优化模式采纳的主比较量。
    """
    hist = {int(y): float(v) for y, v in yearly_long.items()
            if int(y) != latest_year and v is not None and np.isfinite(v)}
    vals = np.array(list(hist.values()), dtype=float)
    n = len(vals)
    base = {"n_years": int(n), "mean": None, "std": None, "cv": None,
            "yearly_ir": None, "hhi": None, "min_year": None,
            "min_year_year": None, "score": None}
    if n < 2:
        return base
    mean_y = float(vals.mean())
    std_y = float(vals.std(ddof=1))
    cv = std_y / mean_y if mean_y > 1e-12 else None
    if std_y > 1e-9:
        yearly_ir = mean_y / std_y
    else:
        yearly_ir = 99.0 if mean_y > 0 else 0.0   # 各年完全平均时封顶
    total = vals.sum()
    if total > 1e-12:
        shares = vals / total
        hhi = float((shares ** 2).sum())          # 均匀时≈1/n, 集中时→1
    else:
        hhi = None
    min_idx = int(np.argmin(vals))
    return {
        "n_years": int(n),
        "mean": mean_y,
        "std": float(std_y),
        "cv": float(cv) if cv is not None else None,
        "yearly_ir": float(yearly_ir),
        "hhi": hhi,
        "min_year": float(vals[min_idx]),
        "min_year_year": int(list(hist.keys())[min_idx]),
        "score": float(yearly_ir),
    }


def calc_group_monotonicity(group_ret: pd.DataFrame, cfg) -> dict:
    """
    分组收益单调性的两个量化指标(静态判断)。

    背景: 强制要求"每年分组收益随组别严格单调递增"太难实现, 导致因子挖不出来;
    但完全放宽(只要求每年收益为正)又会得到 R14 那样收益为正、分组却毫无单调性的因子。
    因此用两个连续指标替代二元判断, 度量单调性的强弱:

    指标A daily_rank_corr: 每天把"分组编号1..n"与"当日各组收益"做Spearman秩相关,
        得到日度时序, 汇总为总均值与分年度均值。1=完全单调递增, 0=与组别无关, <0=反向。
        反映"因子值越高→当日未来收益越高"的逐日排序质量, 比只看年度顶组更细腻。

    指标B out_of_order: 越序惩罚 Σ_{i=1}^{n-1} max(0, R_i - R_{i+1}),
        度量更低组收益高于更高组的程度, 0=完全无越序(单调递增)。
        分别给出日度序列的总均值/年度均值, 以及年度聚合收益的越序值。

    收益口径与评价一致(日度等效: n日前向收益÷n)。秩相关不受逐日市场均值平移影响,
    与 excess/raw 无关; 越序惩罚为日度等效收益率量纲, 数值很小(零点几个百分点级)。
    """
    n_groups = len(group_ret.columns)
    if n_groups < 5:
        return {"n_groups": n_groups, "n_days": 0,
                "daily_rank_corr": None, "daily_out_of_order": None,
                "yearly_out_of_order": {}}

    # A. 每日分组编号 与 当日各组收益 的秩相关
    def _row_rank_corr(row):
        vals = row.dropna()
        if len(vals) < 5:
            return np.nan
        x = vals.index.to_numpy(dtype=float)
        y = vals.to_numpy(dtype=float)
        return float(stats.spearmanr(x, y)[0])

    daily_corr = group_ret.apply(_row_rank_corr, axis=1)

    # B. 每日越序惩罚 Σ max(0, R_i - R_{i+1})  (向量化)
    diff = np.diff(group_ret.values, axis=1)              # R_{i+1} - R_i
    daily_oos = pd.Series(np.maximum(0.0, -diff).sum(axis=1),
                          index=group_ret.index)

    # 年度聚合收益的越序惩罚
    yearly_oos = {}
    yearly_group = group_ret.groupby(group_ret.index.year).sum()
    for year, row in yearly_group.iterrows():
        vals = row.dropna()
        if len(vals) < 5:
            continue
        yv = vals.to_numpy(dtype=float)
        yearly_oos[int(year)] = float(np.maximum(0.0, -np.diff(yv)).sum())

    def _yearly_mean(series: pd.Series) -> dict:
        g = series.groupby(series.index.year).mean().dropna()
        return {int(y): float(v) for y, v in g.items()}

    corr_mean = float(daily_corr.mean()) if daily_corr.notna().any() else np.nan
    oos_mean = float(daily_oos.mean()) if len(daily_oos) else np.nan
    return {
        "n_groups": n_groups,
        "n_days": int(len(group_ret)),
        "daily_rank_corr": {
            "overall_mean": corr_mean,
            "yearly": _yearly_mean(daily_corr),
        },
        "daily_out_of_order": {
            "overall_mean": oos_mean,
            "yearly": _yearly_mean(daily_oos),
        },
        "yearly_out_of_order": yearly_oos,
    }


def classify_group_monotonicity(gm: dict, cfg) -> dict:
    """
    分组单调性四级评级(S优秀 / A良好 / B可入库需优化 / C丢弃)。

    依据三个指标各自评级, 综合取最低档(短板决定):
    - 指标A 每日秩相关(drc, 越大越好): S≥mono_rank_s, A≥mono_rank_a, B≥mono_rank_b, 否则C
    - 指标B-1 越序惩罚日均(doo, 越小越好): S<mono_oos_s, A<mono_oos_a, B<mono_oos_b, 否则C
    - 指标B-2 年度聚合越序峰值(peak, 越小越好): S<mono_peak_s, A<mono_peak_a, B<mono_peak_b, 否则C
    任一秩相关为负的年份 -> 直接判C级(丢弃)。

    处置规则: C级一律拒绝(合格标准追加条件, qualified=False, 不入库不采纳);
    B级可入库但须针对性优化; A级良好可接受; S级优秀应保持。
    """
    drc = (gm.get("daily_rank_corr") or {}).get("overall_mean")
    doo = (gm.get("daily_out_of_order") or {}).get("overall_mean")
    corr_y = (gm.get("daily_rank_corr") or {}).get("yearly") or {}
    y_oos = gm.get("yearly_out_of_order") or {}
    # 最新不完整年份豁免: 与合格标准(c)及长期稳定性一致, 未满一年的最新年份
    # 不参与负年拒绝/坏年份/年度越序峰值判定(其样本不完整, 统计量不可与完整年份直接比较)
    all_years = sorted(set(corr_y) | set(y_oos))
    latest_year = all_years[-1] if all_years else None
    hist_years = [y for y in all_years if y != latest_year]
    peak = float(max([y_oos[y] for y in hist_years if y in y_oos])) if hist_years else 0.0
    neg_years = sorted(int(y) for y, v in corr_y.items() if v < 0 and y != latest_year)
    # 坏年份: 秩相关为负 或 年度聚合越序>10%(B级上界), 均仅统计历史完整年份
    bad_years = sorted(set(neg_years +
                            [int(y) for y, v in y_oos.items()
                             if v > 0.10 and y != latest_year]))
    order = {"S": 0, "A": 1, "B": 2, "C": 3}

    def _grade_rank(d):
        if d is None or not np.isfinite(d):
            return "C"
        if d >= cfg.mono_rank_s:
            return "S"
        if d >= cfg.mono_rank_a:
            return "A"
        if d >= cfg.mono_rank_b:
            return "B"
        return "C"

    def _grade_oos(d):
        if d is None or not np.isfinite(d):
            return "C"
        if d < cfg.mono_oos_s:
            return "S"
        if d < cfg.mono_oos_a:
            return "A"
        if d < cfg.mono_oos_b:
            return "B"
        return "C"

    def _grade_peak(p):
        if p < cfg.mono_peak_s:
            return "S"
        if p < cfg.mono_peak_a:
            return "A"
        if p < cfg.mono_peak_b:
            return "B"
        return "C"

    g_rank, g_oos, g_peak = _grade_rank(drc), _grade_oos(doo), _grade_peak(peak)
    # 综合取最低档(短板决定): order值最大的即最低档
    grade = max([g_rank, g_oos, g_peak], key=lambda g: order[g])
    if neg_years and cfg.mono_neg_year_reject:
        grade = "C"
    return {
        "grade": grade,
        "drc": drc,
        "doo": doo,
        "yearly_peak": peak,
        "grade_rank": g_rank,
        "grade_oos": g_oos,
        "grade_peak": g_peak,
        "neg_years": neg_years,
        "bad_years": bad_years,
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

        # 分组单调性评级(四级), C级一律拒绝(合格标准追加条件)
        mono_grade = classify_group_monotonicity(metrics["group_monotonicity"], cfg)
        monotonicity_ok = bool(mono_grade["grade"] != "C")

        qualified = bool(direction_ok and monthly_ok and yearly_ok and monotonicity_ok)

        result = {
            "name": name, "error": None, "flipped": flipped,
            "direction_ok": direction_ok,
            "monthly_ok": monthly_ok,
            "monthly_require": cfg.monthly_ic_pos_ratio,
            "yearly_ok": yearly_ok,
            "qualified": qualified,
        }
        result.update(metrics)
        result["long_stability"] = calc_long_stability(
            metrics["yearly_long"], metrics["latest_year"])
        result["n_neg_months"] = len(metrics["neg_months"])
        result["monotonicity_grade"] = mono_grade
        result["monotonicity_ok"] = monotonicity_ok
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
        # 多头收益年度稳定性(静态判断: 每年为正且分布平均, 不集中于个别年份)
        stab = ev.get("long_stability") or {}
        if stab.get("score") is not None:
            # 负收益因子: 年均≤0时变异系数无定义、总收益≤0时HHI无定义, 需None安全格式化
            cv_txt = f"{stab['cv']:.2f}" if stab.get("cv") is not None else "NA(年均≤0)"
            hhi_txt = f"{stab['hhi']:.3f}" if stab.get("hhi") is not None else "NA(总收益≤0)"
            lines.append(
                f"[多头稳定性] 年度信息比率={stab['yearly_ir']:.2f}"
                f"(年均{stab['mean']*100:+.2f}% / 年标准差{stab['std']*100:.2f}%), "
                f"变异系数={cv_txt}, 集中度HHI={hhi_txt}, "
                f"最差年{stab['min_year_year']}({stab['min_year']*100:+.2f}%)。"
                "该比率越高=每年多头收益越平均稳定(不集中于个别年份), 是判断因子迭代优劣的核心标准, 优先于单纯抬高IC。")
        # 分组收益单调性(每日秩相关均值 + 越序惩罚, 分年度完整展示)
        gm = ev.get("group_monotonicity") or {}
        drc = (gm.get("daily_rank_corr") or {}).get("overall_mean")
        doo = (gm.get("daily_out_of_order") or {}).get("overall_mean")
        if drc is not None and np.isfinite(drc):
            corr_y = (gm.get("daily_rank_corr") or {}).get("yearly") or {}
            oos_y = (gm.get("daily_out_of_order") or {}).get("yearly") or {}
            y_oos = gm.get("yearly_out_of_order") or {}
            years = sorted(set(corr_y) | set(oos_y) | set(y_oos))
            lines.append(
                f"[分组单调性] 每日秩相关均值={drc:+.3f}(1=完全单调递增, 0=无关), "
                f"越序惩罚日均={doo*100:.4f}%(Σmax(0,R_i-R_(i+1)), 0=完全无越序), "
                f"分组数={gm.get('n_groups')}。分年度值(年: 秩相关 / 越序日均 / 年度聚合越序):")
            for y in years:
                c = corr_y.get(y)
                o = oos_y.get(y)
                v = y_oos.get(y)
                c_txt = f"{c:+.3f}" if c is not None else "NA"
                o_txt = f"{o*100:.3f}%" if o is not None else "NA"
                v_txt = f"{v*100:.2f}%" if v is not None else "NA"
                lines.append(f"    {y}: {c_txt} / {o_txt} / {v_txt}")
            lines.append(
                "    以上分年度值供针对性挖掘: 找出秩相关偏低(<0.15)或年度聚合越序偏高(>10%)的年份, "
                "诊断其市场风格共性并针对性修正, 同时保持高分年份不退化; "
                "理想目标: 秩相关逐年尽量接近1, 越序逐年尽量接近0。")
        # 单调性四级评级, 按等级给不同处置措辞
        mg = ev.get("monotonicity_grade") or {}
        if mg.get("grade"):
            g = mg["grade"]
            gtxt = {"S": "优秀·保持", "A": "良好·可接受", "B": "一般·可入库但需优化",
                    "C": "差·丢弃"}.get(g, g)
            peak_txt = (f"{mg['yearly_peak']*100:.2f}%" if mg.get("yearly_peak") is not None
                        else "NA")
            drc_txt = f"{mg['drc']:+.3f}" if mg.get("drc") is not None else "NA"
            doo_txt = (f"{mg['doo']*100:.3f}%" if mg.get("doo") is not None else "NA")
            lines.append(
                f"[单调性评级] {g}级·{gtxt} (秩相关{drc_txt} / 越序日均{doo_txt} / "
                f"年度峰值{peak_txt}, 三项取最低档)。")
            if g == "C":
                bad_txt = ", ".join(str(y) for y in mg.get("bad_years") or []) or "无"
                lines.append(
                    "    ⚠ 该因子因分组单调性C级被【直接判为不合格】: 排序结构近乎无效"
                    f"(坏年份: {bad_txt}), 按规则一律拒绝——不入库、不采纳, "
                    "必须更换核心逻辑重新设计, 不得再围绕该结构微调。")
            elif g == "B":
                if mg.get("bad_years"):
                    bad_txt = ", ".join(str(y) for y in mg["bad_years"])
                    lines.append(
                        f"    该因子单调性一般: 可入库但须针对性优化——重点改善坏年份({bad_txt})的排序质量, "
                        "分析其风格共性后做结构性修正, 而非仅调参数。")
                else:
                    lines.append(
                        "    该因子单调性一般: 可入库但须针对性优化——无突出坏年份, "
                        "但整体排序质量未达良好线(秩相关/越序日均未到A档), "
                        "需系统性提升各组逐日排序质量, 而非仅调参数。")
            elif g == "A":
                lines.append(
                    "    该因子单调性良好: 在保持核心结构下可小幅提升排序质量(秩相关/越序继续向优秀线靠拢)。")
            else:
                lines.append(
                    "    该因子单调性优秀: 请保持当前结构, 仅允许微调窗口/门控以增强收益, 不得破坏现有排序质量。")
        # 语义评审拒绝 / 相关性撞车 —— 换皮因子的具体被拒原因必须回传给模型, 否则模型永远不知道自己为何被拦
        if ev.get("review_rejected"):
            lines.append(
                f"[语义评审拒绝] 该因子因与库内已有因子语义重复被拒入库: "
                f"{str(ev.get('review_reason') or '')[:200]}")
        if ev.get("library_rejected"):
            lines.append(
                f"[相关性撞车] 该因子与库内因子截面相关性 max|ρ|="
                f"{float(ev.get('library_max_corr') or 0):.2f} 超上限({cfg.max_library_corr}), "
                f"被拒入库——仅换字段/换窗口/换包装构造的'新'因子无法通过此关, 必须换经济逻辑。")
        if ev["qualified"]:
            lines.append("★★ 该因子已通过全部合格标准! 可在此基础上进一步抬高IC或验证稳健性。")
        lines.append("")

    lines.append("综合要求: 下一轮因子必须同时满足 (a)IC均值与多头收益同向为正 "
                 f"(b)月度IC为正占比≥{cfg.monthly_ic_pos_ratio*100:.0f}% "
                 "(c)历史完整年份多头超额全部为正 "
                 "(d)分组单调性评级不得为C级(秩相关为负或年度越序超限直接拒绝)。"
                 "请基于以上反馈进行有针对性的改进, 不要简单重复已失败的公式。")
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


# ====================== Barra 行业/市值中性化 ======================

# Barra v1 暴露数据中的风格/元数据列, 其余中文列即为申万一级行业哑变量
BARRA_NON_INDUSTRY_COLS = {
    "beta", "book_to_price", "earnings_yield", "growth", "leverage",
    "liquidity", "momentum", "non_linear_size", "residual_volatility",
    "size", "specific_return", "specific_risk",
}


def load_barra_exposure(start_date=None, model: str = "v1"):
    """
    通过 local_api 读取 Barra 因子暴露(全市场)。

    Args:
        start_date: 只保留该日期起的暴露(可选, str 或 Timestamp)
        model: Barra 模型版本(本地数据目录 barra/{model}/exposure)

    Returns:
        (exp, ind_cols): exp 为 (date, code) MultiIndex 的暴露 DataFrame(含 size 列);
        ind_cols 为行业哑变量列名列表(申万一级中文名)。
    """
    import local_api as la
    exp = la.get_factor_exposure([], start_date=start_date, end_date=None, model=model)
    if exp is None or exp.empty:
        raise RuntimeError(f"Barra 暴露数据为空, 请检查本地数据目录 barra/{model}/exposure")
    if start_date is not None:
        exp = exp.loc[exp.index.get_level_values(0) >= pd.Timestamp(start_date)]
    ind_cols = [c for c in exp.columns if c not in BARRA_NON_INDUSTRY_COLS]
    return exp, ind_cols


def neutralize_factor(factor_wide: pd.DataFrame, exp: pd.DataFrame,
                      ind_cols: list) -> pd.DataFrame:
    """
    每日截面中性化: factor ~ 行业哑变量 + size, 返回残差宽表(日期×股票)。

    FWL 等价向量化实现: 先在 日期×行业 内对 factor 与 size 各自 demean
    (等价于对行业哑变量回归取残差), 再无截距回归
    beta = Σ(f_dm·s_dm)/Σ(s_dm²), resid = f_dm - beta·s_dm。

    Args:
        factor_wide: 因子宽表(日期×股票)
        exp: Barra 暴露(需含 size 列与行业哑变量列, MultiIndex=(date, code))
        ind_cols: 行业哑变量列名列表
    """
    f_long = factor_wide.stack(future_stack=True).dropna()
    f_long.index.names = ["date", "code"]
    f_long = f_long.rename("factor")

    ind_cols = list(ind_cols)
    need = exp[["size"] + ind_cols]
    merged = f_long.to_frame().join(need, how="inner").sort_index()
    merged = merged.dropna(subset=["size"])

    ind_vals = merged[ind_cols].to_numpy(dtype="float32")
    has_ind = ind_vals.sum(axis=1) > 0.5        # 行业哑变量全为0(缺失行业)的样本剔除
    merged = merged[has_ind]
    ind_vals = ind_vals[has_ind]
    merged["ind"] = ind_vals.argmax(axis=1)

    # 第一步: 行业内 demean (等价于回归行业哑变量取残差)
    grp = merged.groupby([merged.index.get_level_values(0), "ind"], observed=True)
    f_dm = merged["factor"] - grp["factor"].transform("mean")
    s_dm = merged["size"] - grp["size"].transform("mean")

    # 第二步: 每日无截距回归 f_dm ~ s_dm, 取残差
    dates = merged.index.get_level_values(0)
    num = (f_dm * s_dm).groupby(dates).sum()
    den = (s_dm * s_dm).groupby(dates).sum()
    beta = num / den.replace(0.0, np.nan)
    resid = f_dm - beta.loc[dates].to_numpy() * s_dm
    resid.name = "factor"
    return resid.unstack("code").astype("float64")


def neutralize_barra(factor_wide: pd.DataFrame, start_date=None,
                     model: str = "v1") -> pd.DataFrame:
    """一站式 Barra 行业/市值中性化: 加载暴露并每日截面回归, 返回残差宽表(日期×股票)"""
    exp, ind_cols = load_barra_exposure(start_date=start_date, model=model)
    return neutralize_factor(factor_wide, exp, ind_cols)
