"""
生成最佳因子的图片分析报告 (v2: 多头为核心, 单利, 分年度)

因子: VolumeZoneBreakout_20D
输出: f:\\quant\\factor_analysis\\factor_report.png
"""

import sys
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, r"f:\quant")
import factor_analysis as fa

# =============================================================================
# 1. 加载因子数据
# =============================================================================
print("[1/4] 加载因子数据...")

lib_path = r"f:\quant\llm_mining\QuantaAlpha\data\factorlib\all_factors_library.json"
with open(lib_path, "r", encoding="utf-8") as f:
    lib = json.load(f)

factor_id = "6b7a3955873bdb63"
factor_data = lib["factors"][factor_id]
factor_name = factor_data["factor_name"]
factor_expr = factor_data["factor_expression"]
factor_desc = factor_data.get("factor_description", "")
bt = factor_data.get("backtest_results", {})
qlib_annual = bt.get("1day.excess_return_with_cost.annualized_return")
qlib_ir = bt.get("1day.excess_return_with_cost.information_ratio")

h5_path = factor_data["cache_location"]["result_h5_path"]
df = pd.read_hdf(h5_path, key="data")
factor = df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
factor.index.names = ["date", "code"]

print(f"  因子: {factor_name}")

# =============================================================================
# 2. 数据清洗与分析
# =============================================================================
print("[2/4] 数据清洗与因子分析...")

clean_data = fa.get_clean_factor_and_forward_returns(
    factor=factor, periods=[1, 5, 10, 20], quantiles=5,
    stock_pool="csi300", min_stocks_per_day=50,
)

results = fa.create_full_tear_sheet(
    clean_data, method="spearman", period="period_1",
    periods_per_year=252, benchmark_returns=None, excess=True,
    verbose=False,
)

print("  分析完成")

# =============================================================================
# 3. 生成图片报告
# =============================================================================
print("[3/4] 生成图片报告...")

COLORS = {
    "primary": "#2C3E50",
    "accent": "#E74C3C",
    "success": "#27AE60",
    "warning": "#F39C12",
    "info": "#3498DB",
    "light": "#ECF0F1",
    "q_colors": ["#E74C3C", "#E67E22", "#F1C40F", "#27AE60", "#2980B9"],
    "long_color": "#2980B9",
}

fig = plt.figure(figsize=(20, 34), facecolor="white")
gs = gridspec.GridSpec(9, 4, figure=fig, hspace=0.55, wspace=0.3,
                       left=0.06, right=0.96, top=0.98, bottom=0.02)

# -----------------------------------------------------------------------------
# 标题区
# -----------------------------------------------------------------------------
ax_title = fig.add_subplot(gs[0, :])
ax_title.axis("off")
ax_title.text(0.5, 0.85, "因子分析报告", ha="center", va="top",
              fontsize=28, fontweight="bold", color=COLORS["primary"])
ax_title.text(0.5, 0.55, factor_name, ha="center", va="top",
              fontsize=20, color=COLORS["accent"], fontweight="bold")
ax_title.text(0.5, 0.28, f"公式: {factor_expr}", ha="center", va="top",
              fontsize=11, color=COLORS["primary"],
              bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS["light"],
                        edgecolor=COLORS["info"], alpha=0.8))
if factor_desc:
    desc_short = factor_desc[:150] + ("..." if len(factor_desc) > 150 else "")
    ax_title.text(0.5, 0.08, desc_short, ha="center", va="top",
                  fontsize=9, color="#7F8C8D", style="italic")

# -----------------------------------------------------------------------------
# 图1-4: 月度 IC 均值柱状图 (按年月分组, 每月一个柱子)
# 替换原来的 IC 时序和累计 IC
# -----------------------------------------------------------------------------
ic = results["ic"]
period_cols = [c for c in ic.columns if c.startswith("period_")]
# 最多展示 4 个周期, 2x2 网格
ic_axes = []
for idx, period_col in enumerate(period_cols[:4]):
    row_idx = 1 + idx // 2
    col_idx = (idx % 2) * 2
    ax_ic = fig.add_subplot(gs[row_idx, col_idx:col_idx + 2])

    ic_series = ic[period_col].dropna()
    # 按年月分组求平均 (resample 'ME' = 月末), 每月一个值
    ic_monthly = ic_series.resample("ME").mean()

    # 柱子颜色: 正绿负红
    bar_colors = [COLORS["success"] if v >= 0 else COLORS["accent"]
                  for v in ic_monthly.values]
    ax_ic.bar(ic_monthly.index, ic_monthly.values, color=bar_colors,
              alpha=0.8, width=20)  # width=20 天, 让柱子有合适宽度
    ax_ic.axhline(y=0, color="black", linewidth=0.5)

    # x 轴: 每年一个标签
    import matplotlib.dates as mdates
    ax_ic.xaxis.set_major_locator(mdates.YearLocator())
    ax_ic.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_ic.tick_params(axis="x", labelsize=9)

    # 整体平均 IC (text 显示)
    overall_mean = ic_series.mean()
    ax_ic.text(0.98, 0.95, f"整体均值: {overall_mean*100:.2f}%",
               transform=ax_ic.transAxes, fontsize=10, fontweight="bold",
               ha="right", va="top",
               bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["light"],
                         edgecolor=COLORS["info"], alpha=0.9))

    # 周期标签
    period_label = period_col.replace("period_", "")
    ax_ic.set_title(f"月度 IC 均值 ({period_label}日, Spearman Rank IC)",
                    fontsize=12, fontweight="bold")
    ax_ic.set_ylabel("IC 均值")
    ax_ic.grid(True, alpha=0.3, axis="y")
    ic_axes.append(ax_ic)

# -----------------------------------------------------------------------------
# 图5: 分组累计净值 (单利, 超额收益 vs 全市场均值)
# -----------------------------------------------------------------------------
ax5 = fig.add_subplot(gs[3, :2])
cum_returns = results["cumulative_returns"]
for i, col in enumerate(cum_returns.columns):
    ax5.plot(cum_returns.index, cum_returns[col].values,
             color=COLORS["q_colors"][i % 5], linewidth=1.2, label=f"Q{int(col)}")
ax5.set_title("分组累计净值 (单利, 超额 vs 全市场等权均值)", fontsize=13, fontweight="bold")
ax5.set_ylabel("净值")
ax5.legend(loc="upper left", fontsize=9, ncol=5)
ax5.grid(True, alpha=0.3)
ax5.axhline(y=1.0, color="black", linewidth=0.5, linestyle="--")

# -----------------------------------------------------------------------------
# 图6: 多头累计净值 (单利, vs 各基准)
# -----------------------------------------------------------------------------
ax6 = fig.add_subplot(gs[3, 2:])
from factor_analysis.data import get_benchmark_returns
from factor_analysis.performance import calc_long_returns

date_min = str(clean_data.index.get_level_values(0).min().date())
date_max = str(clean_data.index.get_level_values(0).max().date())

benchmarks = {
    "全市场": None,
    "沪深300": "000300.XSHG",
    "中证500": "000905.XSHG",
    "中证1000": "000852.XSHG",
}
bench_colors = ["#2980B9", "#E74C3C", "#27AE60", "#F39C12"]

for i, (name, code) in enumerate(benchmarks.items()):
    if code is None:
        bench_ret = None
    else:
        try:
            bench_ret = get_benchmark_returns(code, date_min, date_max, [1])["period_1"]
        except Exception:
            continue
    long_ret = calc_long_returns(clean_data, period="period_1",
                                 benchmark_returns=bench_ret, excess=True)
    cum_long = 1 + long_ret.cumsum()  # 单利: 净值 = 1 + 累计收益
    ax6.plot(cum_long.index, cum_long.values, color=bench_colors[i],
             linewidth=1.8, label=f"vs {name}")

ax6.set_title("多头累计净值 (单利, vs 各基准超额)", fontsize=13, fontweight="bold")
ax6.set_ylabel("净值")
ax6.legend(loc="upper left", fontsize=9)
ax6.grid(True, alpha=0.3)
ax6.axhline(y=1.0, color="black", linewidth=0.5, linestyle="--")

# -----------------------------------------------------------------------------
# 图7: 多头回撤 (单利, vs 全市场)
# -----------------------------------------------------------------------------
ax7 = fig.add_subplot(gs[4, :2])
long_returns = results["long_returns"]
cum_long_main = long_returns.cumsum()
running_max = cum_long_main.cummax()
drawdown = running_max - cum_long_main  # 单利回撤
ax7.fill_between(drawdown.index, drawdown.values, 0,
                 color=COLORS["accent"], alpha=0.4)
ax7.plot(drawdown.index, drawdown.values, color=COLORS["accent"], linewidth=0.8)
ax7.set_title("多头回撤 (单利, 超额 vs 全市场)", fontsize=13, fontweight="bold")
ax7.set_ylabel("回撤 (单利)")
ax7.grid(True, alpha=0.3)

# -----------------------------------------------------------------------------
# 图8: 分组换手率 (双边)
# -----------------------------------------------------------------------------
ax8 = fig.add_subplot(gs[4, 2:])
turnover = results["turnover"]
for i, col in enumerate(turnover.columns):
    turnover_ma = turnover[col].rolling(60, min_periods=20).mean()
    ax8.plot(turnover_ma.index, turnover_ma.values,
             color=COLORS["q_colors"][i % 5], linewidth=1.2, label=f"Q{int(col)}")
ax8.set_title("分组换手率 (双边, 60日滚动均值)", fontsize=13, fontweight="bold")
ax8.set_ylabel("双边换手率")
ax8.legend(loc="upper right", fontsize=9, ncol=5)
ax8.grid(True, alpha=0.3)

# -----------------------------------------------------------------------------
# 图9: 因子自相关系数
# -----------------------------------------------------------------------------
ax9 = fig.add_subplot(gs[5, :2])
autocorr = results["autocorr"].dropna()
autocorr_ma = autocorr.rolling(60, min_periods=20).mean()
ax9.plot(autocorr.index, autocorr.values, color=COLORS["info"], alpha=0.3, linewidth=0.5)
ax9.plot(autocorr_ma.index, autocorr_ma.values, color=COLORS["primary"], linewidth=1.5,
         label="60日滚动均值")
ax9.axhline(y=0, color="black", linewidth=0.5)
ax9.set_title("因子自相关系数 (稳定性)", fontsize=13, fontweight="bold")
ax9.set_ylabel("自相关系数")
ax9.legend(loc="upper right", fontsize=9)
ax9.grid(True, alpha=0.3)

# -----------------------------------------------------------------------------
# 图10: 分年度分组收益柱状图 (5组对比, 看单调性)
# -----------------------------------------------------------------------------
ax10 = fig.add_subplot(gs[5, 2:])
yearly = results["yearly_stats"]
years = yearly.index.tolist()
n_groups = sum(1 for c in yearly.columns if c.startswith("Q"))
x = np.arange(len(years))
width = 0.8 / n_groups  # 每组柱子宽度

for i in range(n_groups):
    q_col = f"Q{i+1}"
    if q_col not in yearly.columns:
        continue
    vals = yearly[q_col].values * 100  # 转百分号
    offset = (i - (n_groups - 1) / 2) * width
    ax10.bar(x + offset, vals, width, color=COLORS["q_colors"][i % 5],
             alpha=0.85, label=f"Q{i+1}", edgecolor="white", linewidth=0.3)

ax10.set_xticks(x)
ax10.set_xticklabels([str(y) for y in years], fontsize=9)
ax10.axhline(y=0, color="black", linewidth=0.5)
ax10.set_title("分年度分组年化超额收益 (vs 全市场, 看单调性)",
               fontsize=12, fontweight="bold")
ax10.set_ylabel("年化超额收益 (%)")
ax10.legend(loc="upper right", fontsize=8, ncol=n_groups)
ax10.grid(True, alpha=0.3, axis="y")

# -----------------------------------------------------------------------------
# 表格1: IC 统计量 (IC, ICIR)
# -----------------------------------------------------------------------------
ax_t1 = fig.add_subplot(gs[6, :2])
ax_t1.axis("off")
ic_fmt = results["ic_stats_formatted"]
table1 = ax_t1.table(
    cellText=ic_fmt.values,
    rowLabels=ic_fmt.index,
    colLabels=ic_fmt.columns,
    cellLoc="center", loc="center",
)
table1.auto_set_font_size(False)
table1.set_fontsize(10)
table1.scale(1.0, 1.8)
for j in range(len(ic_fmt.columns)):
    table1[0, j].set_facecolor(COLORS["primary"])
    table1[0, j].set_text_props(color="white", fontweight="bold")
for i in range(1, len(ic_fmt) + 1):
    table1[i, -1].set_facecolor(COLORS["light"])
    table1[i, -1].set_text_props(fontweight="bold")
ax_t1.set_title("IC 统计量 (IC均值百分号, ICIR为比率)", fontsize=13, fontweight="bold", y=1.15)

# -----------------------------------------------------------------------------
# 表格2: 多头多基准统计
# -----------------------------------------------------------------------------
ax_t2 = fig.add_subplot(gs[6, 2:])
ax_t2.axis("off")
lmb_fmt = results["long_multi_benchmark_formatted"]
key_cols = ["年化收益", "夏普比率", "最大回撤", "胜率"]
lmb_key = lmb_fmt[key_cols].copy() if all(c in lmb_fmt.columns for c in key_cols) else lmb_fmt

table2 = ax_t2.table(
    cellText=lmb_key.values,
    rowLabels=lmb_key.index,
    colLabels=lmb_key.columns,
    cellLoc="center", loc="center",
)
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

# -----------------------------------------------------------------------------
# 表格3: 分年度统计 (多头年化收益 + 多头年化换手率)
# 表格只显示多头(Q5), 分组对比在柱状图中看
# -----------------------------------------------------------------------------
ax_t3 = fig.add_subplot(gs[7, :])
ax_t3.axis("off")
ys_fmt = results["yearly_stats_formatted"]
# 表格只显示多头(Q5)年化收益 + 多头年化换手率
q5_col = "Q5" if "Q5" in ys_fmt.columns else ys_fmt.columns[-2]
table_cols = [q5_col, "多头年化换手率"]
ys_table = ys_fmt[table_cols].copy()
ys_table.columns = ["多头年化收益", "多头年化换手率"]

table3 = ax_t3.table(
    cellText=ys_table.values,
    rowLabels=ys_table.index,
    colLabels=ys_table.columns,
    cellLoc="center", loc="center",
)
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

# -----------------------------------------------------------------------------
# 表格4: 关键指标汇总
# -----------------------------------------------------------------------------
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
if qlib_annual is not None:
    key_metrics.append(["─" * 25, "─" * 15])
    key_metrics.append(["[QuantaAlpha Qlib回测]", ""])
    key_metrics.append(["QP 年化超额收益", f"{qlib_annual*100:.2f}%"])
    key_metrics.append(["QP 信息比率 IR", f"{qlib_ir:.4f}"])

# 分两列
n = len(key_metrics)
half = (n + 1) // 2
left = key_metrics[:half]
right = key_metrics[half:]
while len(right) < len(left):
    right.append(["", ""])
combined = [[l[0], l[1], r[0], r[1]] for l, r in zip(left, right)]

table4 = ax_t4.table(
    cellText=combined,
    colLabels=["指标", "值", "指标", "值"],
    cellLoc="center", loc="center",
    colWidths=[0.25, 0.25, 0.25, 0.25],
)
table4.auto_set_font_size(False)
table4.set_fontsize(11)
table4.scale(1.0, 1.5)
for j in range(4):
    table4[0, j].set_facecolor(COLORS["warning"])
    table4[0, j].set_text_props(color="white", fontweight="bold", fontsize=12)
for i in range(1, len(combined) + 1):
    for j in range(4):
        if combined[i-1][0] and "─" in str(combined[i-1][0]):
            table4[i, j].set_facecolor(COLORS["light"])
        elif combined[i-1][0] and "QuantaAlpha" in str(combined[i-1][0]):
            table4[i, j].set_facecolor(COLORS["warning"])
            table4[i, j].set_text_props(fontweight="bold", color="white")
ax_t4.set_title("关键指标汇总", fontsize=14, fontweight="bold", y=1.12)

# -----------------------------------------------------------------------------
# 保存
# -----------------------------------------------------------------------------
output_path = r"f:\quant\factor_analysis\factor_report.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"[4/4] 图片报告已保存: {output_path}")
print(f"  文件大小: {os.path.getsize(output_path) / 1024:.1f} KB")

# 文本摘要
print("\n" + "=" * 70)
print("因子报告摘要:")
print(f"  因子: {factor_name}")
print(f"  IC(1日): {summary['ic']['IC均值']*100:.2f}%, ICIR: {summary['ic']['ICIR']:.2f}")
print(f"  多头(vs全市场): 年化 {summary['多头']['年化收益']*100:.2f}%, "
      f"夏普 {summary['多头']['夏普比率']:.4f}, "
      f"回撤 {summary['多头']['最大回撤']*100:.2f}%")
print(f"  多头(vs沪深300): 年化 {summary['多头多基准']['沪深300']['年化收益']*100:.2f}%")
print(f"  多头双边换手率: {summary['换手率']['平均双边换手率']*100:.2f}%")
print("=" * 70)
