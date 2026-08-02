"""
高管增持细分分析：金额>1000万 或 占比>1‰
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_PATH = r"F:\quant\research\holding_increase\increase_events"
OUTPUT_PATH = r"F:\quant\research\holding_increase\analysis_output"
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("=" * 80)
print("高管增持细分分析（金额>1000万 或 占比>1‰）")
print("=" * 80)

# 1. 加载数据
json_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.json')]
all_data = []
for file in json_files:
    try:
        with open(os.path.join(DATA_PATH, file), 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data.append(data)
    except Exception as e:
        pass

df = pd.DataFrame(all_data)
print(f"\n总事件数: {len(df)}")

# 添加日期字段
df['公告日期_dt'] = pd.to_datetime(df['公告日期'])
df['年份'] = df['公告日期_dt'].dt.year

# 收益率列
return_cols = ['1日收益率', '3日收益率', '5日收益率', '10日收益率', 
               '22日收益率', '60日收益率', '90日收益率', '120日收益率']
available_returns = [c for c in return_cols if c in df.columns]

# 数值化
df['增持目标金额_numeric'] = pd.to_numeric(df['增持目标金额'], errors='coerce')
df['增持占比_numeric'] = pd.to_numeric(df['增持占总市值比例'], errors='coerce')

# ============================================================================
# 2. 筛选高管增持
# ============================================================================
df_exec = df[df['增持主体'] == '高管'].copy()
print(f"\n高管增持事件数: {len(df_exec)} ({len(df_exec)/len(df)*100:.1f}%)")

# ============================================================================
# 3. 在高管增持中筛选：金额>1000万 或 占比>1‰
# ============================================================================
mask_amount = df_exec['增持目标金额_numeric'] > 10000000  # 1000万
mask_ratio = df_exec['增持占比_numeric'] > 0.001  # 1‰
mask_filter = mask_amount | mask_ratio

df_exec_filtered = df_exec[mask_filter].copy()
df_exec_unfiltered = df_exec[~mask_filter].copy()

print(f"\n高管增持筛选结果:")
print(f"  满足条件(金额>1000万 或 占比>1‰): {len(df_exec_filtered)} 个事件 ({len(df_exec_filtered)/len(df_exec)*100:.1f}%)")
print(f"  不满足条件: {len(df_exec_unfiltered)} 个事件")
print(f"    - 仅金额>1000万: {(mask_amount & ~mask_ratio).sum()}")
print(f"    - 仅占比>1‰: {(~mask_amount & mask_ratio).sum()}")
print(f"    - 双条件都满足: {(mask_amount & mask_ratio).sum()}")

# ============================================================================
# 4. 收益对比
# ============================================================================
print("\n" + "=" * 80)
print("收益对比：满足筛选条件 vs 不满足 vs 全部高管")
print("=" * 80)

def print_stats(label, data_df):
    if len(data_df) == 0:
        return
    print(f"\n{label} ({len(data_df)}个事件):")
    for col in available_returns:
        mean_ret = data_df[col].mean()
        median_ret = data_df[col].median()
        win_rate = (data_df[col] > 0).mean() * 100
        print(f"  {col}: 均值={mean_ret:+.4f}, 中位数={median_ret:+.4f}, 胜率={win_rate:.1f}%")

print_stats("全部高管增持", df_exec)
print_stats("满足条件（金额>1000万或占比>1‰）", df_exec_filtered)
print_stats("不满足条件", df_exec_unfiltered)

# ============================================================================
# 5. 高管满足条件 - 分年度
# ============================================================================
print("\n" + "=" * 80)
print("满足条件的高管增持 - 分年度分析")
print("=" * 80)

for year in sorted(df_exec_filtered['年份'].unique()):
    year_data = df_exec_filtered[df_exec_filtered['年份'] == year]
    print(f"\n{year}年 ({len(year_data)}个事件):")
    for col in available_returns:
        mean_ret = year_data[col].mean()
        median_ret = year_data[col].median()
        win_rate = (year_data[col] > 0).mean() * 100
        print(f"  {col}: 均值={mean_ret:+.4f}, 中位数={median_ret:+.4f}, 胜率={win_rate:.1f}%")

# ============================================================================
# 6. 高管满足条件 vs 大股东满足条件 对比
# ============================================================================
print("\n" + "=" * 80)
print("高管 vs 大股东（满足金额>1000万或占比>1‰）收益对比")
print("=" * 80)

df_major = df[df['增持主体'] == '大股东'].copy()
mask_amount_m = df_major['增持目标金额_numeric'] > 10000000
mask_ratio_m = df_major['增持占比_numeric'] > 0.001
df_major_filtered = df_major[mask_amount_m | mask_ratio_m].copy()

print(f"\n高管满足条件: {len(df_exec_filtered)} 个事件")
print(f"大股东满足条件: {len(df_major_filtered)} 个事件")

for col in available_returns:
    exec_mean = df_exec_filtered[col].mean() if len(df_exec_filtered) > 0 else 0
    major_mean = df_major_filtered[col].mean() if len(df_major_filtered) > 0 else 0
    exec_win = (df_exec_filtered[col] > 0).mean() * 100 if len(df_exec_filtered) > 0 else 0
    major_win = (df_major_filtered[col] > 0).mean() * 100 if len(df_major_filtered) > 0 else 0
    print(f"\n{col}:")
    print(f"  高管满足条件:  均值={exec_mean:+.4f}, 胜率={exec_win:.1f}%")
    print(f"  大股东满足条件: 均值={major_mean:+.4f}, 胜率={major_win:.1f}%")
    print(f"  差异(高管-大):  {exec_mean-major_mean:+.4f}")

# ============================================================================
# 7. 高管满足条件 - 特征分析
# ============================================================================
print("\n" + "=" * 80)
print("满足条件的高管增持 - 特征分布")
print("=" * 80)

if len(df_exec_filtered) > 0:
    print(f"\n增持目的分布:")
    print(df_exec_filtered['增持目的'].value_counts().to_string())

    print(f"\n增持方式分布 (Top 5):")
    print(df_exec_filtered['增持方式'].value_counts().head(5).to_string())

    print(f"\n资金来源分布:")
    print(df_exec_filtered['资金来源'].value_counts().to_string())

    print(f"\n数值特征:")
    print(f"  平均增持金额: {df_exec_filtered['增持目标金额_numeric'].mean()/10000:,.1f} 万元")
    print(f"  中位数金额:   {df_exec_filtered['增持目标金额_numeric'].median()/10000:,.1f} 万元")
    print(f"  平均增持占比: {df_exec_filtered['增持占比_numeric'].mean()*100:.4f}%")
    print(f"  平均增持期限: {df_exec_filtered['增持期限(月)'].mean():.1f} 个月")

    # 不减持承诺分析
    print(f"\n不减持承诺分布:")
    print(df_exec_filtered['不减持承诺(月)'].value_counts().to_string())

# ============================================================================
# 8. 保存事件列表（高管+满足条件）
# ============================================================================
print("\n" + "=" * 80)
print("保存高管满足条件的事件列表")
print("=" * 80)

if len(df_exec_filtered) > 0:
    list_file = os.path.join(OUTPUT_PATH, '高管满足条件事件列表.csv')
    export_cols = ['公告日期', '股票代码', '增持目标金额', '增持占总市值比例',
                   '增持目的', '增持方式', '增持期限(月)', '资金来源',
                   '不减持承诺(月)'] + available_returns
    existing_export = [c for c in export_cols if c in df_exec_filtered.columns]
    df_exec_filtered[existing_export].to_csv(list_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 事件列表已保存: {os.path.basename(list_file)} ({len(df_exec_filtered)}条)")

# ============================================================================
# 9. 生成图表
# ============================================================================
print("\n生成图表...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 图1: 满足条件 vs 不满足 高管增持收益对比
ax1 = axes[0, 0]
x = np.arange(len(available_returns))
width = 0.25
if len(df_exec) > 0:
    all_exec = [df_exec[c].mean() for c in available_returns]
    ax1.bar(x - width, all_exec, width, label='全部高管', color='steelblue')
if len(df_exec_filtered) > 0:
    filtered_exec = [df_exec_filtered[c].mean() for c in available_returns]
    ax1.bar(x, filtered_exec, width, label='高管+满足条件', color='green')
if len(df_exec_unfiltered) > 0:
    unfiltered_exec = [df_exec_unfiltered[c].mean() for c in available_returns]
    ax1.bar(x + width, unfiltered_exec, width, label='高管+不满足条件', color='red', alpha=0.7)
ax1.set_xticks(x)
ax1.set_xticklabels([c.replace('收益率', '') for c in available_returns])
ax1.set_title('高管增持收益对比（筛选前后）', fontsize=14)
ax1.set_ylabel('平均收益率')
ax1.legend()
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# 图2: 高管 vs 大股东（满足条件）收益对比
ax2 = axes[0, 1]
if len(df_exec_filtered) > 0 and len(df_major_filtered) > 0:
    exec_f = [df_exec_filtered[c].mean() for c in available_returns]
    major_f = [df_major_filtered[c].mean() for c in available_returns]
    x = np.arange(len(available_returns))
    ax2.bar(x - width/2, exec_f, width, label='高管满足条件', color='coral')
    ax2.bar(x + width/2, major_f, width, label='大股东满足条件', color='teal')
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.replace('收益率', '') for c in available_returns])
    ax2.set_title('高管 vs 大股东（满足条件）收益对比', fontsize=14)
    ax2.set_ylabel('平均收益率')
    ax2.legend()
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# 图3: 满足条件的高管 - 增持目的分布
ax3 = axes[1, 0]
if len(df_exec_filtered) > 0:
    purpose_counts = df_exec_filtered['增持目的'].value_counts()
    colors_p = plt.cm.Pastel1(np.linspace(0, 1, len(purpose_counts)))
    purpose_counts.plot(kind='pie', ax=ax3, autopct='%1.1f%%', colors=colors_p, startangle=90)
    ax3.set_title('高管满足条件 - 增持目的分布', fontsize=14)
    ax3.set_ylabel('')

# 图4: 满足条件的高管 - 分年度收益路径
ax4 = axes[1, 1]
if len(df_exec_filtered) > 0:
    years = sorted(df_exec_filtered['年份'].unique())
    markers = ['o', 's', '^', 'D']
    for i, year in enumerate(years):
        year_data = df_exec_filtered[df_exec_filtered['年份'] == year]
        if len(year_data) > 0:
            path = [year_data[c].mean() for c in available_returns]
            ax4.plot([c.replace('收益率', '') for c in available_returns], path, 
                     marker=markers[i % len(markers)], label=f'{year}年({len(year_data)}个)', linewidth=2)
    ax4.set_title('高管满足条件 - 分年度收益路径', fontsize=14)
    ax4.set_ylabel('平均收益率')
    ax4.legend()
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
chart_file = os.path.join(OUTPUT_PATH, '05_exec_filtered_analysis.png')
plt.savefig(chart_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ 图表已保存: {os.path.basename(chart_file)}")

# 汇总表输出
print("\n" + "=" * 80)
print("汇总表（平均收益率）")
print("=" * 80)

rows = []
labels_list = []

if len(df_exec) > 0:
    rows.append([df_exec[c].mean() for c in available_returns])
    labels_list.append('全部高管')
if len(df_exec_filtered) > 0:
    rows.append([df_exec_filtered[c].mean() for c in available_returns])
    labels_list.append('高管+满足条件')
if len(df_exec_unfiltered) > 0:
    rows.append([df_exec_unfiltered[c].mean() for c in available_returns])
    labels_list.append('高管+不满足')
if len(df_major_filtered) > 0:
    rows.append([df_major_filtered[c].mean() for c in available_returns])
    labels_list.append('大股东+满足条件')

period_labels = [c.replace('收益率', '') for c in available_returns]
summary_df = pd.DataFrame(rows, index=labels_list, columns=period_labels)
print(summary_df.round(4).to_string())

# 保存汇总
summary_file = os.path.join(OUTPUT_PATH, '高管筛选分析汇总.csv')
summary_df.round(4).to_csv(summary_file, encoding='utf-8-sig')
print(f"\n✅ 汇总表已保存: {os.path.basename(summary_file)}")
print(f"\n分析完成！输出目录: {OUTPUT_PATH}")
