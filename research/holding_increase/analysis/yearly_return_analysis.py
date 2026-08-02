"""
按年度分析增持事件收益率（0-120日）
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

EVENT_PATH = r"F:\quant\research\holding_increase\increase_events"
OUTPUT_PATH = r"F:\quant\research\holding_increase\analysis_output"

# 加载事件
files = [f for f in os.listdir(EVENT_PATH) if f.endswith('.json')]
records = []
for f in files:
    with open(os.path.join(EVENT_PATH, f), 'r', encoding='utf-8') as fp:
        records.append(json.load(fp))

df = pd.DataFrame(records)
df['公告日期'] = pd.to_datetime(df['公告日期'])
df['year'] = df['公告日期'].dt.year

# 筛选（同之前）
def is_active_purpose(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    s = str(val)
    return not ('被动' in s or '触发' in s or '稳定股价' in s)

def funding_only_other(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    parts = [s.strip() for s in str(val).split('|') if s.strip()]
    if not parts:
        return False
    for p in parts:
        if '自有' in p or '自筹' in p or '金融' in p or '专项贷' in p or '贷款' in p:
            return False
    return True

def method_only_excluded(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    parts = [s.strip() for s in str(val).split('|') if s.strip()]
    if not parts:
        return False
    excluded_kw = ['协议转让', '认购', '定增', '定向增发']
    for p in parts:
        if not any(kw in p for kw in excluded_kw):
            return False
    return True

df = df[df['增持目的'].apply(is_active_purpose)].copy()
df = df[~df['资金来源'].apply(funding_only_other)].copy()
df = df[~df['增持方式'].apply(method_only_excluded)].copy()
print(f"筛选后事件数: {len(df)}")

# ============================================================================
# 按年度统计收益率序列
# ============================================================================
PERIODS = [1, 3, 5, 10, 22, 60, 90, 120]
period_cols = [f'{p}日收益率' for p in PERIODS]

print("\n" + "=" * 70)
print("按年度平均收益率")
print("=" * 70)
print(f"{'年份':<6} {'事件数':<8}", end="")
for p in PERIODS:
    print(f"{str(p)+'日':<8}", end="")
print()
print("-" * 70)

yearly_stats = {}
for year in sorted(df['year'].unique()):
    year_df = df[df['year'] == year]
    n = len(year_df)
    print(f"{year:<6} {n:<8}", end="")
    stats = {}
    for p, col in zip(PERIODS, period_cols):
        if col in year_df.columns:
            vals = year_df[col].dropna()
            mean_val = vals.mean() if len(vals) > 0 else np.nan
            stats[f'{p}日'] = mean_val
            print(f"{mean_val:<8.2%}", end="")
        else:
            print(f"{'N/A':<8}", end="")
    print()
    yearly_stats[year] = stats

# 全样本
print(f"{'全部':<6} {len(df):<8}", end="")
for p, col in zip(PERIODS, period_cols):
    if col in df.columns:
        vals = df[col].dropna()
        print(f"{vals.mean():<8.2%}", end="")
print()

# ============================================================================
# 按年度胜率
# ============================================================================
print("\n" + "=" * 70)
print("按年度胜率")
print("=" * 70)
print(f"{'年份':<6} {'事件数':<8}", end="")
for p in PERIODS:
    print(f"{str(p)+'日':<8}", end="")
print()
print("-" * 70)

for year in sorted(df['year'].unique()):
    year_df = df[df['year'] == year]
    n = len(year_df)
    print(f"{year:<6} {n:<8}", end="")
    for p, col in zip(PERIODS, period_cols):
        if col in year_df.columns:
            vals = year_df[col].dropna()
            win_rate = (vals > 0).sum() / len(vals) if len(vals) > 0 else np.nan
            print(f"{win_rate:<8.1%}", end="")
        else:
            print(f"{'N/A':<8}", end="")
    print()

# ============================================================================
# 绘图：分年度收益率曲线
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

years = sorted(df['year'].unique())
colors_year = {2023: 'steelblue', 2024: 'coral', 2025: 'green', 2026: 'purple'}

# 图1: 分年度平均收益率折线
ax = axes[0]
for year in years:
    year_df = df[df['year'] == year]
    means = [year_df[col].dropna().mean() for col in period_cols if col in year_df.columns]
    valid_periods = [p for p, col in zip(PERIODS, period_cols) if col in year_df.columns]
    ax.plot(valid_periods, [m*100 for m in means], 'o-', color=colors_year[year],
            linewidth=1.5, markersize=5, label=f'{year}年 (n={len(year_df)})')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('持有交易日')
ax.set_ylabel('平均收益率 (%)')
ax.set_title('分年度平均收益率')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(PERIODS)

# 图2: 分年度中位数收益率
ax = axes[1]
for year in years:
    year_df = df[df['year'] == year]
    medians = [year_df[col].dropna().median() for col in period_cols if col in year_df.columns]
    valid_periods = [p for p, col in zip(PERIODS, period_cols) if col in year_df.columns]
    ax.plot(valid_periods, [m*100 for m in medians], 's-', color=colors_year[year],
            linewidth=1.5, markersize=5, label=f'{year}年')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('持有交易日')
ax.set_ylabel('中位数收益率 (%)')
ax.set_title('分年度中位数收益率')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(PERIODS)

# 图3: 分年度胜率
ax = axes[2]
for year in years:
    year_df = df[df['year'] == year]
    win_rates = []
    for col in period_cols:
        if col in year_df.columns:
            vals = year_df[col].dropna()
            win_rates.append((vals > 0).sum() / len(vals) * 100 if len(vals) > 0 else np.nan)
    valid_periods = [p for p, col in zip(PERIODS, period_cols) if col in year_df.columns]
    ax.plot(valid_periods, win_rates, '^-', color=colors_year[year],
            linewidth=1.5, markersize=5, label=f'{year}年')
ax.axhline(y=50, color='black', linestyle='--', linewidth=0.5)
ax.set_xlabel('持有交易日')
ax.set_ylabel('胜率 (%)')
ax.set_title('分年度胜率')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(PERIODS)

# 图4: 分年度10%和90%分位数（22日）
ax = axes[3]
x_pos = np.arange(len(years))
width = 0.35
means_22 = []
q10_22 = []
q90_22 = []
for year in years:
    year_df = df[df['year'] == year]
    vals = year_df['22日收益率'].dropna()
    means_22.append(vals.mean() * 100)
    q10_22.append(vals.quantile(0.1) * 100)
    q90_22.append(vals.quantile(0.9) * 100)

bars = ax.bar(x_pos, means_22, width, color=[colors_year[y] for y in years], alpha=0.7, label='均值')
ax.errorbar(x_pos, means_22, yerr=[np.array(means_22)-np.array(q10_22), np.array(q90_22)-np.array(means_22)],
            fmt='none', color='black', capsize=5)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{y}年' for y in years])
ax.set_ylabel('22日收益率 (%)')
ax.set_title('22日收益率: 均值 + 10%/90%分位数')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('增持事件分年度收益分析（筛选后）', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '13_yearly_return_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图已保存: 13_yearly_return_analysis.png")

# ============================================================================
# 补充：2025年按月细分
# ============================================================================
print("\n" + "=" * 70)
print("2025年按月细分（22日收益率）")
print("=" * 70)
df_2025 = df[df['year'] == 2025].copy()
df_2025['month'] = df_2025['公告日期'].dt.month
for month, grp in df_2025.groupby('month'):
    vals = grp['22日收益率'].dropna()
    if len(vals) > 0:
        print(f"  {month:2d}月: n={len(vals):3d}, 均值={vals.mean():.2%}, 中位数={vals.median():.2%}, 胜率={((vals>0).sum()/len(vals)):.1%}")
