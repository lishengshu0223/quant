"""
本地化因子挖掘项目 - 合格因子全面评价与单张图片报告

对最佳因子重新计算全区间因子值, 用 factor_analysis 做完整 tear sheet
(1/5/10/20日 RankIC、分组净值、多头多基准、回撤、换手、自相关、分年度),
汇总为一张图片。
"""

import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

import factor_analysis as fa

from . import console
from .config import REPORT_PNG_PATH
from .expr_engine import compute_factor

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = {
    "primary": "#2C3E50", "accent": "#E74C3C", "success": "#27AE60",
    "warning": "#F39C12", "info": "#3498DB", "light": "#ECF0F1",
    # 分组配色: Q1(最差,红) 渐变到 Q10(多头,深紫), 兼容5组/10组
    "q_colors": ["#E74C3C", "#E67E22", "#F39C12", "#F1C40F", "#2ECC71",
                 "#1ABC9C", "#3498DB", "#2980B9", "#9B59B6", "#8E44AD"],
}


def generate_report(best: dict, data, cfg, png_path: str | None = None,
                    factor_wide: pd.DataFrame | None = None) -> str:
    if png_path is None:
        png_path = REPORT_PNG_PATH
    name = best.get("name", "factor")
    expr = best.get("expr", "")
    desc = best.get("描述", "") or best.get("desc", "")
    hypothesis = best.get("hypothesis", "")
    ev = best.get("eval") or {}
    qualified = ev.get("qualified", False)

    if factor_wide is None:
        console.log(f"    重新计算最佳因子全区间值: {name}")
        factor_wide = compute_factor(expr, data, cfg)
    else:
        console.log(f"    使用预计算因子值(如中性化因子): {name}")
    if ev.get("flipped"):
        factor_wide = -factor_wide
        console.log("    该因子评价时已翻转方向, 报告按翻转后方向展示。")
    series = factor_wide.stack(future_stack=True).dropna()
    series.index.names = ["date", "code"]
    series.name = "factor"

    # 交易资格遮盖: 剔除 ST/停牌/次新/涨跌停 股票(与评价口径一致, 避免收益虚高)
    if getattr(data, "tradable", None) is not None and not data.tradable.empty:
        tradable_long = data.tradable.stack(future_stack=True)
        mask = tradable_long.reindex(series.index).fillna(False).astype(bool)
        series = series[mask]
        console.log(f"    已应用交易资格遮盖, 有效观测 {len(series)} 条。")

    console.log("    完整因子分析(1/5/10/20日 RankIC + 分组 + 多头多基准)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clean = fa.get_clean_factor_and_forward_returns(
            factor=series, prices=data.close_long,
            periods=(1, 5, 10, 20), quantiles=cfg.n_quantiles,
            min_stocks_per_day=cfg.min_stocks_per_day,
        )
        results = fa.create_full_tear_sheet(
            clean, method="spearman", period=f"period_{cfg.ic_period}",
            periods_per_year=252, benchmark_returns=None, excess=True, verbose=False,
            normalize=True,  # n日前向收益÷n 转日度等效, 跨周期可比
        )

    console.log("    绘制图片报告...")
    fig = plt.figure(figsize=(20, 34), facecolor="white")
    gs = gridspec.GridSpec(9, 4, figure=fig, hspace=0.55, wspace=0.3,
                           left=0.06, right=0.96, top=0.98, bottom=0.02)

    # ---- 标题 ----
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    verdict = "合格因子 (通过全部标准)" if qualified else "最佳候选因子 (未完全达标)"
    ax_title.text(0.5, 0.88, f"本地化LLM因子挖掘报告 · {verdict}", ha="center", va="top",
                  fontsize=26, fontweight="bold", color=COLORS["primary"])
    ax_title.text(0.5, 0.58, name, ha="center", va="top",
                  fontsize=20, color=COLORS["accent"], fontweight="bold")
    ax_title.text(0.5, 0.30, f"公式: {expr}", ha="center", va="top", fontsize=11,
                  color=COLORS["primary"], parse_math=False,
                  bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS["light"],
                            edgecolor=COLORS["info"], alpha=0.8))
    sub = (desc or hypothesis)[:150]
    if sub:
        ax_title.text(0.5, 0.08, sub, ha="center", va="top", fontsize=9,
                      color="#7F8C8D", style="italic")

    # ---- 图1-4: 月度IC柱状图(1/5/10/20日) ----
    ic = results["ic"]
    period_cols = [c for c in ic.columns if c.startswith("period_")]
    for idx, period_col in enumerate(period_cols[:4]):
        row_idx = 1 + idx // 2
        col_idx = (idx % 2) * 2
        ax = fig.add_subplot(gs[row_idx, col_idx:col_idx + 2])
        ic_series = ic[period_col].dropna()
        ic_monthly = ic_series.resample("ME").mean()
        bar_colors = [COLORS["success"] if v >= 0 else COLORS["accent"]
                      for v in ic_monthly.values]
        ax.bar(ic_monthly.index, ic_monthly.values, color=bar_colors, alpha=0.8, width=20)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", labelsize=9)
        pos_ratio = (ic_monthly > 0).mean()
        ax.text(0.98, 0.95,
                f"整体均值: {ic_series.mean()*100:.2f}%  月正比例: {pos_ratio*100:.0f}%",
                transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["light"],
                          edgecolor=COLORS["info"], alpha=0.9))
        label = period_col.replace("period_", "")
        ax.set_title(f"月度 RankIC 均值 ({label}日, Spearman)", fontsize=12, fontweight="bold")
        ax.set_ylabel("IC 均值")
        ax.grid(True, alpha=0.3, axis="y")

    # ---- 图5: 分组累计净值 ----
    ax5 = fig.add_subplot(gs[3, :2])
    cum_returns = results["cumulative_returns"]
    for i, col in enumerate(cum_returns.columns):
        ax5.plot(cum_returns.index, cum_returns[col].values,
                 color=COLORS["q_colors"][i % 10], linewidth=1.2, label=f"Q{int(col)}")
    ax5.set_title(f"分组累计净值 (单利, {cfg.ic_period}日前向收益÷{cfg.ic_period}日度等效, 超额 vs 全市场等权)",
                  fontsize=12, fontweight="bold")
    ax5.set_ylabel("净值")
    ax5.legend(loc="upper left", fontsize=9, ncol=5)
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=1.0, color="black", linewidth=0.5, linestyle="--")

    # ---- 图6: 多头累计净值 vs 各基准 ----
    ax6 = fig.add_subplot(gs[3, 2:])
    from factor_analysis.data import get_benchmark_returns
    from factor_analysis.performance import calc_long_returns
    date_min = str(clean.index.get_level_values(0).min().date())
    date_max = str(clean.index.get_level_values(0).max().date())
    benchmarks = {"全市场": None, "沪深300": "000300.XSHG",
                  "中证500": "000905.XSHG", "中证1000": "000852.XSHG"}
    bench_colors = ["#2980B9", "#E74C3C", "#27AE60", "#F39C12"]
    period_col = f"period_{cfg.ic_period}"
    for i, (bname, code) in enumerate(benchmarks.items()):
        bench_ret = None
        if code is not None:
            try:
                bench_ret = get_benchmark_returns(code, date_min, date_max,
                                                  [cfg.ic_period])[period_col]
            except Exception:
                continue
        long_ret = calc_long_returns(clean, period=period_col,
                                     benchmark_returns=bench_ret, excess=True,
                                     normalize=True)
        cum_long = 1 + long_ret.cumsum()
        ax6.plot(cum_long.index, cum_long.values, color=bench_colors[i],
                 linewidth=1.8, label=f"vs {bname}")
    ax6.set_title(f"多头累计净值 (单利, {cfg.ic_period}日÷{cfg.ic_period}日度等效, 超额 vs 各基准)",
                  fontsize=12, fontweight="bold")
    ax6.set_ylabel("净值")
    ax6.legend(loc="upper left", fontsize=9)
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=1.0, color="black", linewidth=0.5, linestyle="--")

    # ---- 图7: 多头回撤 ----
    ax7 = fig.add_subplot(gs[4, :2])
    long_returns = results["long_returns"]
    cum_long_main = long_returns.cumsum()
    drawdown = cum_long_main.cummax() - cum_long_main
    ax7.fill_between(drawdown.index, drawdown.values, 0, color=COLORS["accent"], alpha=0.4)
    ax7.plot(drawdown.index, drawdown.values, color=COLORS["accent"], linewidth=0.8)
    ax7.set_title(f"多头回撤 (单利, {cfg.ic_period}日÷{cfg.ic_period}日度等效, 超额 vs 全市场)",
                  fontsize=12, fontweight="bold")
    ax7.set_ylabel("回撤")
    ax7.grid(True, alpha=0.3)

    # ---- 图8: 分组换手率 ----
    ax8 = fig.add_subplot(gs[4, 2:])
    turnover = results["turnover"]
    for i, col in enumerate(turnover.columns):
        t_ma = turnover[col].rolling(60, min_periods=20).mean()
        ax8.plot(t_ma.index, t_ma.values, color=COLORS["q_colors"][i % 10],
                 linewidth=1.2, label=f"Q{int(col)}")
    ax8.set_title(f"分组换手率 (双边, {cfg.ic_period}日调仓间隔÷{cfg.ic_period}日度等效, 60日滚动均值)",
                  fontsize=12, fontweight="bold")
    ax8.set_ylabel("双边换手率")
    ax8.legend(loc="upper right", fontsize=9, ncol=5)
    ax8.grid(True, alpha=0.3)

    # ---- 图9: 因子自相关 ----
    ax9 = fig.add_subplot(gs[5, :2])
    autocorr = results["autocorr"].dropna()
    ac_ma = autocorr.rolling(60, min_periods=20).mean()
    ax9.plot(autocorr.index, autocorr.values, color=COLORS["info"], alpha=0.3, linewidth=0.5)
    ax9.plot(ac_ma.index, ac_ma.values, color=COLORS["primary"], linewidth=1.5,
             label="60日滚动均值")
    ax9.axhline(y=0, color="black", linewidth=0.5)
    ax9.set_title("因子自相关系数 (稳定性)", fontsize=13, fontweight="bold")
    ax9.set_ylabel("自相关系数")
    ax9.legend(loc="upper right", fontsize=9)
    ax9.grid(True, alpha=0.3)

    # ---- 图10: 分年度分组收益 ----
    ax10 = fig.add_subplot(gs[5, 2:])
    yearly = results["yearly_stats"]
    years = yearly.index.tolist()
    n_groups = sum(1 for c in yearly.columns if str(c).startswith("Q"))
    x = np.arange(len(years))
    width = 0.8 / max(n_groups, 1)
    for i in range(n_groups):
        q_col = f"Q{i+1}"
        if q_col not in yearly.columns:
            continue
        vals = yearly[q_col].values * 100
        offset = (i - (n_groups - 1) / 2) * width
        ax10.bar(x + offset, vals, width, color=COLORS["q_colors"][i % 10],
                 alpha=0.85, label=f"Q{i+1}", edgecolor="white", linewidth=0.3)
    ax10.set_xticks(x)
    ax10.set_xticklabels([str(y) for y in years], fontsize=9)
    ax10.axhline(y=0, color="black", linewidth=0.5)
    latest_year = max(years)
    ax10.set_title(f"分年度分组年化超额收益 (日度等效, vs 全市场; {latest_year}为不完整年份)",
                   fontsize=12, fontweight="bold")
    ax10.set_ylabel("年化超额收益 (%)")
    # 图例放到图片下方, 避免遮挡多头组(最高组)每年的收益柱
    ax10.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
                ncol=min(n_groups, 10), fontsize=8, frameon=False)
    ax10.grid(True, alpha=0.3, axis="y")

    # ---- 表格1: IC统计量 ----
    ax_t1 = fig.add_subplot(gs[6, :2])
    ax_t1.axis("off")
    ic_fmt = results["ic_stats_formatted"]
    t1 = ax_t1.table(cellText=ic_fmt.values, rowLabels=ic_fmt.index,
                     colLabels=ic_fmt.columns, cellLoc="center", loc="center")
    t1.auto_set_font_size(False)
    t1.set_fontsize(10)
    t1.scale(1.0, 1.8)
    for j in range(len(ic_fmt.columns)):
        t1[0, j].set_facecolor(COLORS["primary"])
        t1[0, j].set_text_props(color="white", fontweight="bold")
    ax_t1.set_title("RankIC 统计量 (1/5/10/20日)", fontsize=13, fontweight="bold", y=1.15)

    # ---- 表格2: 多头多基准 ----
    ax_t2 = fig.add_subplot(gs[6, 2:])
    ax_t2.axis("off")
    lmb_fmt = results["long_multi_benchmark_formatted"]
    key_cols = ["年化收益", "夏普比率", "最大回撤", "胜率"]
    lmb_key = lmb_fmt[key_cols] if all(c in lmb_fmt.columns for c in key_cols) else lmb_fmt
    t2 = ax_t2.table(cellText=lmb_key.values, rowLabels=lmb_key.index,
                     colLabels=lmb_key.columns, cellLoc="center", loc="center")
    t2.auto_set_font_size(False)
    t2.set_fontsize(10)
    t2.scale(1.0, 1.8)
    for j in range(len(lmb_key.columns)):
        t2[0, j].set_facecolor(COLORS["accent"])
        t2[0, j].set_text_props(color="white", fontweight="bold")
    ax_t2.set_title(f"多头多基准统计 ({cfg.ic_period}日超额)", fontsize=13,
                    fontweight="bold", y=1.15)

    # ---- 表格3: 分年度统计 ----
    ax_t3 = fig.add_subplot(gs[7, :])
    ax_t3.axis("off")
    ys_fmt = results["yearly_stats_formatted"]
    q5_col = f"Q{cfg.n_quantiles}"
    table_cols = [q5_col, "多头年化换手率"]
    ys_table = ys_fmt[[c for c in table_cols if c in ys_fmt.columns]].copy()
    t3 = ax_t3.table(cellText=ys_table.values, rowLabels=ys_table.index,
                     colLabels=ys_table.columns, cellLoc="center", loc="center")
    t3.auto_set_font_size(False)
    t3.set_fontsize(10)
    t3.scale(1.0, 1.6)
    for j in range(len(ys_table.columns)):
        t3[0, j].set_facecolor(COLORS["success"])
        t3[0, j].set_text_props(color="white", fontweight="bold")
    ax_t3.set_title("分年度统计 (多头年化超额收益 + 多头年化双边换手率)",
                    fontsize=13, fontweight="bold", y=1.15)

    # ---- 表格4: 关键指标与合格标准核对 ----
    ax_t4 = fig.add_subplot(gs[8, :])
    ax_t4.axis("off")
    summary = results["summary"]
    km = []
    if "ic" in summary:
        km.append([f"RankIC 均值 ({cfg.ic_period}日)", f"{summary['ic']['IC均值']*100:.2f}%"])
        km.append(["ICIR", f"{summary['ic']['ICIR']:.2f}"])
    if "多头" in summary:
        km.append(["多头年化超额(vs全市场)", f"{summary['多头']['年化收益']*100:.2f}%"])
        km.append(["多头夏普比率", f"{summary['多头']['夏普比率']:.4f}"])
        km.append(["多头最大回撤(单利)", f"{summary['多头']['最大回撤']*100:.2f}%"])
    km.append(["多头平均双边换手率(日度等效)",
               f"{summary['换手率']['平均双边换手率']*100:.2f}%" if '换手率' in summary else "-"])
    if "多头多基准" in summary:
        for bname, s in summary["多头多基准"].items():
            km.append([f"多头 vs {bname}", f"{s['年化收益']*100:.2f}%"])
    km.append(["─" * 25, "─" * 15])
    km.append(["[合格标准核对]", ""])
    km.append(["(a) IC与多头收益同向为正",
               "通过" if ev.get("direction_ok") else "未通过"])
    km.append(["(b) 月度IC为正占比≥60%",
               f"{ev.get('monthly_pos_ratio', 0)*100:.1f}% "
               + ("通过" if ev.get("monthly_ok") else "未通过")])
    bad_years = ev.get("bad_hist_years") or {}
    km.append(["(c) 历史完整年份多头超额为正",
               "通过" if ev.get("yearly_ok") else f"未通过({len(bad_years)}年)"])

    n = len(km)
    half = (n + 1) // 2
    left = km[:half]
    right = km[half:]
    while len(right) < len(left):
        right.append(["", ""])
    combined = [[l[0], l[1], r[0], r[1]] for l, r in zip(left, right)]
    t4 = ax_t4.table(cellText=combined, colLabels=["指标", "值", "指标", "值"],
                     cellLoc="center", loc="center",
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    t4.auto_set_font_size(False)
    t4.set_fontsize(11)
    t4.scale(1.0, 1.5)
    for j in range(4):
        t4[0, j].set_facecolor(COLORS["warning"])
        t4[0, j].set_text_props(color="white", fontweight="bold", fontsize=12)
    ax_t4.set_title("关键指标汇总", fontsize=14, fontweight="bold", y=1.12)

    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 文本摘要
    console.log("")
    console.log("    " + "=" * 60)
    console.log(f"    因子报告摘要: {name}")
    if "ic" in summary:
        console.log(f"    RankIC({cfg.ic_period}日): {summary['ic']['IC均值']*100:.2f}%, "
                    f"ICIR: {summary['ic']['ICIR']:.2f}")
    if "多头" in summary:
        console.log(f"    多头(vs全市场): 年化 {summary['多头']['年化收益']*100:.2f}%, "
                    f"夏普 {summary['多头']['夏普比率']:.4f}, "
                    f"回撤 {summary['多头']['最大回撤']*100:.2f}%")
    console.log(f"    合格判定: {'全部通过' if qualified else '未完全达标'}")
    console.log("    " + "=" * 60)
    return png_path
