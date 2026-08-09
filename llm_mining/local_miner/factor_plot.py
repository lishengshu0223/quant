"""
本地化因子挖掘项目 - 每轮因子评价图 / 全库相关性 / 静态HTML报告

1) plot_round_factor : 为单个因子生成紧凑评价图(matplotlib, 公式用 mathtext 数学化渲染)
   - 月度IC柱状 + 十分位分组年化收益 + 多头累计净值 + 关键指标表 + LaTeX公式
2) plot_library_corr  : 全库合格因子两两截面Spearman相关矩阵 -> 热图PNG + CSV
3) build_html_report  : 汇总所有轮次的评价图/公式(KaTeX渲染)/全库相关性 -> 静态 index.html
"""

import datetime
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import factor_analysis as fa

from . import factor_library
from .formula_tex import to_mathtext, to_tex

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = {
    "primary": "#2C3E50", "accent": "#E74C3C", "success": "#27AE60",
    "warning": "#F39C12", "info": "#3498DB", "light": "#ECF0F1",
    "q_colors": ["#E74C3C", "#E67E22", "#F39C12", "#F1C40F", "#2ECC71",
                 "#1ABC9C", "#3498DB", "#2980B9", "#9B59B6", "#8E44AD"],
}


# =============================================================================
# 单因子评价图
# =============================================================================

def _prep_clean(factor_wide: pd.DataFrame, data, cfg):
    """与 factor_eval 相同口径: 交易资格遮盖 -> clean 对象 + 完整分析结果。
    返回 (clean, results): clean 为多周期前向收益清洗对象,
    results 为 fa.create_full_tear_sheet 完整分析(月度IC/分组/多头/换手率/分年度统计)。"""
    series = factor_wide.stack(future_stack=True).dropna()
    series.index.names = ["date", "code"]
    series.name = "factor"
    if getattr(data, "tradable", None) is not None and not data.tradable.empty:
        tradable_long = data.tradable.stack(future_stack=True)
        mask = tradable_long.reindex(series.index).fillna(False).astype(bool)
        series = series[mask]
    # IC 展示多周期(1/5/10/20日), 收益与换手率主口径用 cfg.ic_period
    periods = sorted(set((1, 5, 10, 20, cfg.ic_period)))
    with np.errstate(all="ignore"):
        clean = fa.get_clean_factor_and_forward_returns(
            factor=series, prices=data.close_long,
            periods=tuple(periods), quantiles=cfg.n_quantiles,
            min_stocks_per_day=cfg.min_stocks_per_day,
        )
        results = fa.create_full_tear_sheet(
            clean, method="spearman", period=f"period_{cfg.ic_period}",
            periods_per_year=252, benchmark_returns=None, excess=True,
            verbose=False, normalize=True,
        )
    return clean, results


def _verdict(ev: dict) -> str:
    if ev.get("error"):
        return "计算失败"
    if ev.get("qualified"):
        return "合格"
    if ev.get("review_rejected"):
        return "语义评审拒绝"
    if ev.get("library_rejected"):
        return "相关性撞车拒绝"
    if ev.get("direction_ok"):
        return "方向正确 · 未达标"
    return "方向无效/失败"


def plot_round_factor(entry: dict, factor_wide: pd.DataFrame, data, cfg,
                      out_path: str, round_no: int = 0, idx: int = 0) -> str | None:
    """生成单因子评价图(日频标准格式: 月度IC 2x2 / 分组累计净值 / 多头多基准净值 /
    回撤 / 分组换手率 / 自相关 / 分年度分组收益柱状 / IC统计 / 多头多基准统计 /
    分年度统计(多头年化收益+换手率) / 关键指标汇总)。
    factor_wide 为全区间因子宽表(调用方已计算)。失败返回 None, 不抛异常。"""
    try:
        name = entry.get("name", "factor")
        expr = entry.get("expr", "")
        ev = entry.get("eval") or {}
        flipped = ev.get("flipped")
        if flipped:
            factor_wide = -factor_wide
        clean, results = _prep_clean(factor_wide, data, cfg)
        ic = results.get("ic")
        if ic is None or len(ic) == 0:
            fig = plt.figure(figsize=(16, 8), facecolor="white")
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.text(0.5, 0.5, f"{name}  ·  评价数据不足", ha="center", va="center",
                    fontsize=20, fontweight="bold", color=COLORS["primary"])
            fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            return out_path

        n_q = len(COLORS["q_colors"])
        fig = plt.figure(figsize=(20, 32), facecolor="white")
        gs = fig.add_gridspec(9, 4, hspace=0.55, wspace=0.3,
                              left=0.06, right=0.96, top=0.98, bottom=0.02)

        # ---- 标题区 ----
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis("off")
        tag = f"第{round_no}轮" if round_no else "评价"
        ax_title.text(0.5, 0.88, f"{tag}因子评价报告", ha="center", va="top",
                      fontsize=24, fontweight="bold", color=COLORS["primary"])
        ax_title.text(0.5, 0.62, name, ha="center", va="top",
                      fontsize=18, color=COLORS["accent"], fontweight="bold")
        if expr:
            ax_title.text(0.5, 0.32, to_mathtext(expr), ha="center", va="top",
                          fontsize=13, color=COLORS["primary"],
                          bbox=dict(boxstyle="round,pad=0.45", facecolor=COLORS["light"],
                                    edgecolor=COLORS["info"], alpha=0.9))
        vcol = COLORS["success"] if ev.get("qualified") else \
            (COLORS["warning"] if ev.get("direction_ok") else COLORS["accent"])
        ax_title.text(0.5, 0.10, _verdict(ev), ha="center", va="top", fontsize=13,
                      fontweight="bold", color=vcol)

        # ---- 图1-4: 月度 IC 均值柱状图 (2x2, 按 1/5/10/20 日排序) ----
        period_cols = sorted(
            [c for c in ic.columns if str(c).startswith("period_")],
            key=lambda c: int(str(c).replace("period_", "")))[:4]
        for i, pc in enumerate(period_cols):
            ax_ic = fig.add_subplot(gs[1 + i // 2, (i % 2) * 2:(i % 2) * 2 + 2])
            s = ic[pc].dropna()
            m = s.resample("ME").mean().dropna()
            bar_colors = [COLORS["success"] if v >= 0 else COLORS["accent"]
                          for v in m.values]
            ax_ic.bar(m.index, m.values, color=bar_colors, alpha=0.8, width=20)
            ax_ic.axhline(y=0, color="black", linewidth=0.5)
            ax_ic.xaxis.set_major_locator(mdates.YearLocator())
            ax_ic.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax_ic.tick_params(axis="x", labelsize=8)
            ax_ic.text(0.98, 0.95, f"整体均值: {s.mean()*100:.2f}%",
                       transform=ax_ic.transAxes, fontsize=10, fontweight="bold",
                       ha="right", va="top",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["light"],
                                 edgecolor=COLORS["info"], alpha=0.9))
            ax_ic.set_title(f"月度 IC 均值 ({pc.replace('period_', '')}日, Spearman Rank IC)",
                            fontsize=12, fontweight="bold")
            ax_ic.set_ylabel("IC 均值")
            ax_ic.grid(True, alpha=0.3, axis="y")

        # ---- 图5: 分组累计净值 (单利, 超额 vs 全市场等权均值) ----
        ax5 = fig.add_subplot(gs[3, :2])
        cum_returns = results["cumulative_returns"]
        qcols = list(cum_returns.columns)
        for i, col in enumerate(qcols):
            ax5.plot(cum_returns.index, cum_returns[col].values,
                     color=COLORS["q_colors"][i % n_q], linewidth=1.0,
                     label=f"Q{int(col)}")
        ax5.set_title(f"分组累计净值 (单利, 超额 vs 全市场等权均值, {cfg.n_quantiles}组)",
                      fontsize=13, fontweight="bold")
        ax5.set_ylabel("净值")
        ax5.legend(loc="upper left", fontsize=8, ncol=min(len(qcols), 5))
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=1.0, color="black", linewidth=0.5, linestyle="--")

        # ---- 图6: 多头累计净值 (单利, vs 各基准) ----
        ax6 = fig.add_subplot(gs[3, 2:])
        from factor_analysis.data import get_benchmark_returns
        from factor_analysis.performance import calc_long_returns
        date_min = str(clean.index.get_level_values(0).min().date())
        date_max = str(clean.index.get_level_values(0).max().date())
        benchmarks = {"全市场": None, "沪深300": "000300.XSHG",
                      "中证500": "000905.XSHG", "中证1000": "000852.XSHG"}
        bench_colors = ["#2980B9", "#E74C3C", "#27AE60", "#F39C12"]
        # 基准周期须与组合收益周期(period_{cfg.ic_period})一致, 否则 normalize 时基准贡献被缩小
        bench_days = cfg.ic_period
        for i, (bn, code) in enumerate(benchmarks.items()):
            bench_ret = None
            if code is not None:
                try:
                    bench_ret = get_benchmark_returns(
                        code, date_min, date_max, [bench_days])[f"period_{bench_days}"]
                except Exception:
                    continue
            lr = calc_long_returns(clean, period=f"period_{cfg.ic_period}",
                                   benchmark_returns=bench_ret, excess=True,
                                   normalize=True)
            cum_long = 1 + lr.cumsum()
            ax6.plot(cum_long.index, cum_long.values, color=bench_colors[i],
                     linewidth=1.6, label=f"vs {bn}")
        ax6.set_title("多头累计净值 (单利, 日度等效, vs 各基准超额)",
                      fontsize=13, fontweight="bold")
        ax6.set_ylabel("净值")
        ax6.legend(loc="upper left", fontsize=9)
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=1.0, color="black", linewidth=0.5, linestyle="--")

        # ---- 图7: 多头回撤 (单利, vs 全市场) ----
        ax7 = fig.add_subplot(gs[4, :2])
        long_ret = results["long_returns"]
        cum_long_main = long_ret.cumsum()
        running_max = cum_long_main.cummax()
        drawdown = running_max - cum_long_main
        ax7.fill_between(drawdown.index, drawdown.values, 0,
                         color=COLORS["accent"], alpha=0.4)
        ax7.plot(drawdown.index, drawdown.values, color=COLORS["accent"], linewidth=0.8)
        ax7.set_title("多头回撤 (单利, 日度等效, 超额 vs 全市场)",
                      fontsize=13, fontweight="bold")
        ax7.set_ylabel("回撤 (单利)")
        ax7.grid(True, alpha=0.3)

        # ---- 图8: 分组换手率 (双边, 日度等效, 60日滚动均值) ----
        ax8 = fig.add_subplot(gs[4, 2:])
        turnover = results["turnover"]
        for i, col in enumerate(turnover.columns):
            ma = turnover[col].rolling(60, min_periods=20).mean()
            ax8.plot(ma.index, ma.values,
                     color=COLORS["q_colors"][i % n_q], linewidth=1.0,
                     label=f"Q{int(col)}")
        ax8.set_title("分组换手率 (双边, 日度等效, 60日滚动均值)",
                      fontsize=13, fontweight="bold")
        ax8.set_ylabel("双边换手率")
        ax8.legend(loc="upper right", fontsize=8, ncol=min(len(turnover.columns), 5))
        ax8.grid(True, alpha=0.3)

        # ---- 图9: 因子自相关系数 ----
        ax9 = fig.add_subplot(gs[5, :2])
        autocorr = results["autocorr"].dropna()
        autocorr_ma = autocorr.rolling(60, min_periods=20).mean()
        ax9.plot(autocorr.index, autocorr.values, color=COLORS["info"],
                 alpha=0.3, linewidth=0.5)
        ax9.plot(autocorr_ma.index, autocorr_ma.values, color=COLORS["primary"],
                 linewidth=1.5, label="60日滚动均值")
        ax9.axhline(y=0, color="black", linewidth=0.5)
        ax9.set_title("因子自相关系数 (稳定性)", fontsize=13, fontweight="bold")
        ax9.set_ylabel("自相关系数")
        ax9.legend(loc="upper right", fontsize=9)
        ax9.grid(True, alpha=0.3)

        # ---- 图10: 分年度分组收益柱状图 (看单调性) ----
        ax10 = fig.add_subplot(gs[5, 2:])
        yearly = results["yearly_stats"]
        years = yearly.index.tolist()
        qcols10 = [c for c in yearly.columns if str(c).startswith("Q")]
        x = np.arange(len(years))
        width = 0.8 / max(len(qcols10), 1)
        for i, qc in enumerate(qcols10):
            vals = yearly[qc].values * 100
            offset = (i - (len(qcols10) - 1) / 2) * width
            ax10.bar(x + offset, vals, width,
                     color=COLORS["q_colors"][i % n_q], alpha=0.85,
                     label=qc, edgecolor="white", linewidth=0.3)
        ax10.set_xticks(x)
        ax10.set_xticklabels([str(y) for y in years], fontsize=9)
        ax10.axhline(y=0, color="black", linewidth=0.5)
        ax10.set_title("分年度分组年化超额收益 (vs 全市场, 日度等效, 看单调性)",
                       fontsize=12, fontweight="bold")
        ax10.set_ylabel("年化超额收益 (%)")
        ax10.legend(loc="upper right", fontsize=7, ncol=min(len(qcols10), 5))
        ax10.grid(True, alpha=0.3, axis="y")

        # ---- 表格1: IC 统计量 ----
        ax_t1 = fig.add_subplot(gs[6, :2])
        ax_t1.axis("off")
        ic_fmt = results["ic_stats_formatted"]
        table1 = ax_t1.table(cellText=ic_fmt.values, rowLabels=ic_fmt.index,
                             colLabels=ic_fmt.columns, cellLoc="center", loc="center")
        table1.auto_set_font_size(False)
        table1.set_fontsize(10)
        table1.scale(1.0, 1.8)
        for j in range(len(ic_fmt.columns)):
            table1[0, j].set_facecolor(COLORS["primary"])
            table1[0, j].set_text_props(color="white", fontweight="bold")
        for i in range(1, len(ic_fmt) + 1):
            table1[i, -1].set_facecolor(COLORS["light"])
            table1[i, -1].set_text_props(fontweight="bold")
        ax_t1.set_title("IC 统计量 (IC均值百分号, ICIR为比率)",
                        fontsize=13, fontweight="bold", y=1.15)

        # ---- 表格2: 多头多基准统计 ----
        ax_t2 = fig.add_subplot(gs[6, 2:])
        ax_t2.axis("off")
        lmb_fmt = results["long_multi_benchmark_formatted"]
        key_cols = ["年化收益", "夏普比率", "最大回撤", "胜率"]
        lmb_key = lmb_fmt[key_cols].copy() if all(c in lmb_fmt.columns for c in key_cols) else lmb_fmt
        table2 = ax_t2.table(cellText=lmb_key.values, rowLabels=lmb_key.index,
                             colLabels=lmb_key.columns, cellLoc="center", loc="center")
        table2.auto_set_font_size(False)
        table2.set_fontsize(10)
        table2.scale(1.0, 1.8)
        for j in range(len(lmb_key.columns)):
            table2[0, j].set_facecolor(COLORS["accent"])
            table2[0, j].set_text_props(color="white", fontweight="bold")
        for i in range(1, len(lmb_key) + 1):
            table2[i, -1].set_facecolor(COLORS["light"])
            table2[i, -1].set_text_props(fontweight="bold")
        ax_t2.set_title("多头多基准统计 (超额收益)", fontsize=13, fontweight="bold", y=1.15)

        # ---- 表格3: 分年度统计 (多头年化收益 + 多头年化换手率) ----
        ax_t3 = fig.add_subplot(gs[7, :])
        ax_t3.axis("off")
        ys_fmt = results["yearly_stats_formatted"]
        q_last = f"Q{cfg.n_quantiles}"
        q_col = q_last if q_last in ys_fmt.columns else ys_fmt.columns[-2]
        table_cols = [q_col, "多头年化换手率"]
        ys_table = ys_fmt[table_cols].copy()
        ys_table.columns = ["多头年化收益", "多头年化换手率"]
        table3 = ax_t3.table(cellText=ys_table.values, rowLabels=ys_table.index,
                             colLabels=ys_table.columns, cellLoc="center", loc="center")
        table3.auto_set_font_size(False)
        table3.set_fontsize(10)
        table3.scale(1.0, 1.6)
        for j in range(len(ys_table.columns)):
            table3[0, j].set_facecolor(COLORS["success"])
            table3[0, j].set_text_props(color="white", fontweight="bold")
        for i in range(1, len(ys_table) + 1):
            table3[i, -1].set_facecolor(COLORS["light"])
            table3[i, -1].set_text_props(fontweight="bold")
        ax_t3.set_title("分年度统计 (多头年化超额收益% + 多头年化双边换手率)",
                        fontsize=13, fontweight="bold", y=1.15)

        # ---- 表格4: 关键指标汇总 ----
        ax_t4 = fig.add_subplot(gs[8, :])
        ax_t4.axis("off")
        summary = results["summary"]
        key_metrics = []
        if "ic" in summary:
            key_metrics.append(["IC 均值", f"{summary['ic']['IC均值']*100:.2f}%"])
            key_metrics.append(["ICIR", f"{summary['ic']['ICIR']:.2f}"])
        if "多头" in summary:
            key_metrics.append(["多头年化收益(vs全市场)", f"{summary['多头']['年化收益']*100:.2f}%"])
            key_metrics.append(["多头夏普比率", f"{summary['多头']['夏普比率']:.4f}"])
            key_metrics.append(["多头最大回撤(单利)", f"{summary['多头']['最大回撤']*100:.2f}%"])
        if "换手率" in summary:
            key_metrics.append(["多头平均双边换手率", f"{summary['换手率']['平均双边换手率']*100:.2f}%"])
        if "多头多基准" in summary:
            for bench, s in summary["多头多基准"].items():
                key_metrics.append([f"多头 vs {bench}", f"{s['年化收益']*100:.2f}%"])
        # 追加挖掘过程判定信息(与 factor_eval 口径一致)
        key_metrics.append(["挖掘判定", _verdict(ev)])
        if ev.get("ic_mean") is not None:
            key_metrics.append(["挖掘IC均值", f"{ev['ic_mean']*100:+.3f}%"])
        if ev.get("monotonicity_grade"):
            key_metrics.append(["单调性评级", str(ev.get("monotonicity_grade"))])
        n = len(key_metrics)
        half = (n + 1) // 2
        left = key_metrics[:half]
        right = key_metrics[half:]
        while len(right) < len(left):
            right.append(["", ""])
        combined = [[l[0], l[1], r[0], r[1]] for l, r in zip(left, right)]
        table4 = ax_t4.table(cellText=combined, colLabels=["指标", "值", "指标", "值"],
                             cellLoc="center", loc="center",
                             colWidths=[0.25, 0.25, 0.25, 0.25])
        table4.auto_set_font_size(False)
        table4.set_fontsize(10)
        table4.scale(1.0, 1.5)
        for j in range(4):
            table4[0, j].set_facecolor(COLORS["warning"])
            table4[0, j].set_text_props(color="white", fontweight="bold", fontsize=11)
        ax_t4.set_title("关键指标汇总", fontsize=13, fontweight="bold", y=1.12)

        fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return out_path
    except Exception as e:
        try:
            plt.close("all")
        except Exception:
            pass
        print(f"    [评价图失败] {type(e).__name__}: {e}")
        return None


# =============================================================================
# 全库相关性矩阵
# =============================================================================

def compute_library_corr_matrix(data, cfg, exclude_id: str = "") -> dict:
    """全库合格因子两两截面Spearman相关(抽样日期取均值)。
    返回 {matrix, series_ids, names, max_abs, flag}。matrix: DataFrame(对角NaN)。"""
    library = factor_library.load_library()
    targets = [(s.get("series_id"), (s.get("best") or {}).get("expr"),
                bool((s.get("best") or {}).get("flipped")), s.get("name"))
               for s in library if s.get("series_id") != exclude_id
               and (s.get("best") or {}).get("expr")]
    if len(targets) < 2:
        return {"matrix": None, "series_ids": [t[0] for t in targets],
                "names": {t[0]: t[3] for t in targets}, "max_abs": 0.0, "flag": False}

    from .expr_engine import compute_factor
    # 抽样日期: 以第一个因子的宽表日期为准
    wide = compute_factor(targets[0][1], data, cfg)
    if targets[0][2]:
        wide = -wide
    dates = wide.index
    n = cfg.corr_sample_dates
    sample_dates = dates[::max(1, len(dates) // n)][:n] if len(dates) > n else dates

    subs = {}
    names = {}
    for sid, expr, flipped, nm in targets:
        try:
            w = compute_factor(expr, data, cfg)
            if flipped:
                w = -w
            subs[sid] = w.reindex(sample_dates)
        except Exception:
            subs[sid] = None
        names[sid] = nm or sid

    sids = [t[0] for t in targets]
    mat = pd.DataFrame(np.nan, index=sids, columns=sids, dtype=float)
    for i, a in enumerate(sids):
        for j in range(i + 1, len(sids)):
            b = sids[j]
            wa, wb = subs.get(a), subs.get(b)
            if wa is None or wb is None:
                continue
            corrs = []
            for d in sample_dates:
                sa = wa.loc[d].dropna()
                sb = wb.loc[d].dropna()
                common = sa.index.intersection(sb.index)
                if len(common) < 30:
                    continue
                c = sa.loc[common].rank().corr(sb.loc[common].rank())
                if np.isfinite(c):
                    corrs.append(c)
            if corrs:
                mat.loc[a, b] = mat.loc[b, a] = float(np.mean(corrs))
    absv = mat.abs().to_numpy(dtype=float)
    max_abs = float(np.nanmax(absv)) if absv.size else 0.0
    return {"matrix": mat, "series_ids": sids, "names": names,
            "max_abs": max_abs, "flag": bool(max_abs > cfg.max_library_corr),
            "threshold": cfg.max_library_corr, "n": len(sids)}


def plot_library_corr(res: dict, out_png: str, out_csv: str | None = None) -> str | None:
    """相关性矩阵 -> 热图PNG(+CSV)。res 来自 compute_library_corr_matrix。"""
    mat = res.get("matrix")
    if mat is None or mat.shape[0] < 2:
        return None
    try:
        fig, ax = plt.subplots(figsize=(max(6, mat.shape[0] * 0.8),
                                        max(5.5, mat.shape[0] * 0.7)), facecolor="white")
        data = mat.to_numpy(dtype=float)
        im = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(mat.shape[1]))
        ax.set_yticks(range(mat.shape[0]))
        ax.set_xticklabels(list(mat.columns), fontsize=9, rotation=45, ha="right")
        ax.set_yticklabels(list(mat.index), fontsize=9)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if abs(v) > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"因子库全库截面相关矩阵 (|ρ|max={res.get('max_abs', 0):.2f}, "
                     f"上限{res.get('threshold', 0.5)})", fontsize=12, fontweight="bold")
        fig.savefig(out_png, dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        if out_csv:
            mat.to_csv(out_csv, encoding="utf-8-sig")
        return out_png
    except Exception as e:
        print(f"    [相关性热图失败] {e}")
        return None


# =============================================================================
# 静态 HTML 报告(KaTeX 公式, 优先本地资源离线渲染, 缺失时回退 CDN)
# =============================================================================

KATEX_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "katex")
KATEX_CDN = "https://cdn.jsdelivr.net/npm/katex/dist"


def _katex_base(out_dir: str) -> str:
    """返回 KaTeX 资源 URL 前缀: 本地相对路径(离线可用) 或 CDN 回退"""
    if os.path.isdir(KATEX_ASSETS):
        rel = os.path.relpath(KATEX_ASSETS, out_dir).replace("\\", "/")
        return rel
    return KATEX_CDN


HTML_HEAD = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>分钟因子挖掘进度报告</title>
<link rel="stylesheet" href="{katex}/katex.min.css">
<script defer src="{katex}/katex.min.js"></script>
<script defer src="{katex}/contrib/auto-render.min.js"
        onload="renderMathInElement(document.body,{{delimiters:[{{left:'$$',right:'$$',display:true}}]}});">
</script>
<style>
body{{font-family:"Microsoft YaHei",sans-serif;margin:24px auto;max-width:1180px;color:#2C3E50;}}
h1{{color:#2C3E50;border-bottom:3px solid #3498DB;padding-bottom:8px;}}
h2{{color:#3498DB;margin-top:36px;}}
h3{{color:#2C3E50;}}
.round{{border:1px solid #D5DBDB;border-radius:8px;padding:14px 18px;margin:14px 0;background:#FAFBFC;}}
.formula{{background:#F4F6F7;border-left:4px solid #3498DB;padding:10px 14px;margin:8px 0;
         font-size:1.05em;border-radius:4px;}}
.eval{{color:#7F8C8D;font-size:0.92em;}}
img{{max-width:100%;border:1px solid #D5DBDB;border-radius:6px;margin-top:6px;}}
table{{border-collapse:collapse;margin:10px 0;}}
th,td{{border:1px solid #BDC3C7;padding:5px 10px;font-size:0.92em;}}
th{{background:#3498DB;color:#fff;}}
.tag{{display:inline-block;padding:2px 10px;border-radius:10px;color:#fff;font-size:0.85em;font-weight:bold;}}
.ok{{background:#27AE60;}} .fail{{background:#E74C3C;}} .warn{{background:#F39C12;}}
.note{{color:#95A5A6;font-size:0.9em;}}
</style>
</head>
<body>
"""


def _ev_tag(ev: dict) -> str:
    if ev.get("error"):
        return '<span class="tag fail">计算失败</span>'
    if ev.get("qualified"):
        return '<span class="tag ok">合格</span>'
    if ev.get("review_rejected") or ev.get("library_rejected"):
        return '<span class="tag warn">拒绝入库</span>'
    if ev.get("direction_ok"):
        return '<span class="tag warn">方向正确未达标</span>'
    return '<span class="tag fail">方向无效</span>'


def _round_section(rec: dict, out_dir: str) -> str:
    round_no = rec.get("round")
    hyp = rec.get("hypothesis", "")
    parts = [f'<div class="round"><h3>第 {round_no} 轮</h3>']
    if hyp:
        parts.append(f'<p class="eval">假设: {hyp[:200]}</p>')
    for i, f in enumerate(rec.get("factors", []), 1):
        ev = f.get("eval") or {}
        parts.append(f"<h4>因子{i}: {f.get('name', '')} {_ev_tag(ev)}</h4>")
        expr = f.get("expr", "")
        if expr:
            parts.append(f'<div class="formula">$${to_tex(expr)}$$</div>')
        if ev.get("error"):
            parts.append(f'<p class="eval">错误: {ev["error"][:160]}</p>')
        else:
            stab = ev.get("long_stability") or {}
            parts.append(
                f'<p class="eval">IC均值 {ev.get("ic_mean", 0)*100:+.3f}% · '
                f'ICIR {ev.get("icir", 0):.3f} · 多头年化 '
                f'{ev.get("long_annual", 0)*100:+.2f}% · '
                f'月度正占比 {ev.get("monthly_pos_ratio", 0)*100:.0f}% · '
                f'年度信息比率 {stab.get("score") if stab.get("score") is not None else "-"}</p>')
        png = f.get("_png_rel")
        if png and os.path.exists(os.path.join(out_dir, png)):
            parts.append(f'<img src="{png}" alt="评价图">')
    parts.append("</div>")
    return "\n".join(parts)


def build_html_report(out_dir: str, rounds_meta: list, cfg,
                      corr_res: dict | None = None, library: list | None = None,
                      title: str = "分钟因子挖掘进度报告") -> str:
    """生成/覆盖 index.html: 标题 + 全库相关性 + 每轮因子(KaTeX公式+评价图)"""
    parts = [HTML_HEAD.format(katex=_katex_base(out_dir)), f"<h1>{title}</h1>"]
    parts.append(f'<p class="note">更新于 {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} · '
                 f'共 {len(rounds_meta)} 轮</p>')

    # 全库相关性
    library = factor_library.load_library() if library is None else library
    parts.append("<h2>因子库总览与相关性</h2>")
    if library:
        rows = []
        for s in library:
            best = s.get("best") or {}
            ev = best.get("eval") or {}
            rows.append(
                f"<tr><td>{s.get('series_id')}</td><td>{s.get('name','')}</td>"
                f"<td>{best.get('expr','')}</td>"
                f"<td>{ev.get('ic_mean',0)*100:+.3f}%</td>"
                f"<td>{ev.get('long_annual',0)*100:+.2f}%</td></tr>")
        parts.append("<table><tr><th>系列</th><th>名称</th><th>最佳公式</th>"
                     "<th>IC均值</th><th>多头年化</th></tr>" + "".join(rows) + "</table>")
    if corr_res and corr_res.get("matrix") is not None:
        mat = corr_res["matrix"]
        png_rel = "library_corr.png"
        if plot_library_corr(corr_res, os.path.join(out_dir, png_rel)):
            parts.append(f'<img src="{png_rel}" alt="全库相关性热图" '
                         f'style="max-width:640px;">')
        parts.append(f'<p class="eval">|ρ| 最大 {corr_res.get("max_abs", 0):.2f} '
                     f'(上限 {corr_res.get("threshold", 0.5)}) · '
                     f'{"撞车超标" if corr_res.get("flag") else "未超标"}</p>')
        corr_csv = os.path.join(out_dir, "library_corr.csv")
        mat.to_csv(corr_csv, encoding="utf-8-sig")

    # 各轮次
    parts.append("<h2>各轮因子评价</h2>")
    for rec in rounds_meta:
        parts.append(_round_section(rec, out_dir))

    parts.append("</body></html>")
    html = "\n".join(parts)
    out_html = os.path.join(out_dir, "index.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html
