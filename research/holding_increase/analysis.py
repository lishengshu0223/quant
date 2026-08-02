"""
高管/大股东增持事件分析脚本
功能：
1. 事件统计（总数、年度、月度、时间分布）
2. 收益分析（0-120天平均收益、分位数收益）
3. 筛选分析（增持金额>1000万 或 占比>1‰）
4. 特征分析（高收益 vs 亏损事件特征）
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置
DATA_PATH = r"F:\quant\research\holding_increase\increase_events"
OUTPUT_PATH = r"F:\quant\research\holding_increase\analysis_output"
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("=" * 80)
print("高管/大股东增持事件分析")
print("=" * 80)

# ============================================================================
# 1. 加载数据
# ============================================================================
print("\n[Step 1] 加载数据...")

# 读取所有JSON文件
json_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.json')]
print(f"找到 {len(json_files)} 个事件文件")

if not json_files:
    print("没有数据，请先运行数据获取！")
    exit()

# 合并所有数据
all_data = []
for file in json_files:
    try:
        with open(os.path.join(DATA_PATH, file), 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data.append(data)
    except Exception as e:
        print(f"  读取 {file} 失败: {e}")

df = pd.DataFrame(all_data)
print(f"成功加载 {len(df)} 条事件数据")

# 添加日期相关字段
df['公告日期_dt'] = pd.to_datetime(df['公告日期'])
df['年份'] = df['公告日期_dt'].dt.year
df['月份'] = df['公告日期_dt'].dt.month
df['年月'] = df['公告日期_dt'].dt.strftime('%Y-%m')
df['季度'] = df['公告日期_dt'].dt.quarter

# ============================================================================
# 2. 事件统计
# ============================================================================
print("\n[Step 2] 事件统计分析")
print("-" * 80)

total_events = len(df)
print(f"\n📊 事件总数: {total_events}")

# 按年份统计
yearly_stats = df.groupby('年份').agg({
    '股票代码': 'count',
    '公告日期': 'nunique'
}).rename(columns={'股票代码': '事件数', '公告日期': '涉及日期数'})

print("\n📅 年度分布:")
print(yearly_stats)

# 按月份统计
monthly_stats = df.groupby('年月').size().reset_index(name='事件数')
monthly_stats = monthly_stats.sort_values('年月')

print("\n📆 月度分布:")
# 按月汇总
df['年_month'] = df['公告日期_dt'].dt.to_period('M')
monthly_summary = df.groupby('年_month').size()
print(f"  平均每月事件数: {monthly_summary.mean():.1f}")
print(f"  最多月份事件数: {monthly_summary.max()} ({monthly_summary.idxmax()})")
print(f"  最少月份事件数: {monthly_summary.min()} ({monthly_summary.idxmin()})")

# 星期分布
df['星期'] = df['公告日期_dt'].dt.dayofweek
weekday_names = ['周一', '周二', '周三', '周四', '周五']
weekday_dist = df.groupby('星期').size()
print("\n📈 星期分布:")
for i, name in enumerate(weekday_names):
    if i in weekday_dist.index:
        print(f"  {name}: {weekday_dist[i]} ({weekday_dist[i]/total_events*100:.1f}%)")

# 生成图1：事件时间分布图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 图1-1: 年度柱状图
ax1 = axes[0, 0]
yearly_events = df.groupby('年份').size()
yearly_events.plot(kind='bar', ax=ax1, color='steelblue')
ax1.set_title('年度事件分布', fontsize=14)
ax1.set_xlabel('年份')
ax1.set_ylabel('事件数')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
for i, v in enumerate(yearly_events):
    ax1.text(i, v + 5, str(v), ha='center', va='bottom')

# 图1-2: 月度趋势图
ax2 = axes[0, 1]
monthly_trend = df.groupby('年_month').size()
monthly_trend.plot(ax=ax2, color='coral', marker='o', markersize=3)
ax2.set_title('月度事件趋势', fontsize=14)
ax2.set_xlabel('年月')
ax2.set_ylabel('事件数')
ax2.tick_params(axis='x', rotation=45)

# 图1-3: 星期分布饼图
ax3 = axes[1, 0]
weekday_counts = df.groupby('星期').size()
weekday_labels = [weekday_names[i] for i in weekday_counts.index]
ax3.pie(weekday_counts, labels=weekday_labels, autopct='%1.1f%%', startangle=90)
ax3.set_title('星期分布', fontsize=14)

# 图1-4: 季度分布图
ax4 = axes[1, 1]
quarterly_events = df.groupby(['年份', '季度']).size().unstack(fill_value=0)
quarterly_events.plot(kind='bar', ax=ax4, colormap='Set2')
ax4.set_title('季度分布', fontsize=14)
ax4.set_xlabel('年份')
ax4.set_ylabel('事件数')
ax4.legend(['Q1', 'Q2', 'Q3', 'Q4'])
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '01_event_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ 图1已保存: 01_event_distribution.png")

# ============================================================================
# 3. 收益分析
# ============================================================================
print("\n[Step 3] 收益分析")
print("-" * 80)

# 提取收益率数据
return_cols = ['1日收益率', '3日收益率', '5日收益率', '10日收益率', 
               '22日收益率', '60日收益率', '90日收益率', '120日收益率']

available_returns = [c for c in return_cols if c in df.columns]
print(f"\n可用的收益指标: {available_returns}")

# 总体收益统计
print("\n📊 总体收益统计:")
return_stats = df[available_returns].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
print(return_stats.round(4))

# 分年度收益统计
print("\n📅 分年度收益统计:")
for year in sorted(df['年份'].unique()):
    year_data = df[df['年份'] == year]
    print(f"\n  {year}年 ({len(year_data)}个事件):")
    for col in available_returns:
        if col in year_data.columns:
            mean_ret = year_data[col].mean()
            median_ret = year_data[col].median()
            win_rate = (year_data[col] > 0).mean() * 100
            print(f"    {col}: 均值={mean_ret:.4f}, 中位数={median_ret:.4f}, 胜率={win_rate:.1f}%")

# 生成图2：收益分布图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 图2-1: 各周期平均收益对比
ax1 = axes[0, 0]
means = df[available_returns].mean()
means.plot(kind='bar', ax=ax1, color=['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336', '#00BCD4', '#795548', '#607D8B'])
ax1.set_title('各周期平均收益率', fontsize=14)
ax1.set_ylabel('平均收益率')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
for i, v in enumerate(means):
    ax1.text(i, v + 0.005 if v >= 0 else v - 0.01, f'{v:.4f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)

# 图2-2: 收益分位数图
ax2 = axes[0, 1]
period_names = [c.replace('收益率', '') for c in available_returns]
x = np.arange(len(period_names))
width = 0.2

q10 = [df[c].quantile(0.1) for c in available_returns]
q25 = [df[c].quantile(0.25) for c in available_returns]
q50 = [df[c].quantile(0.5) for c in available_returns]
q75 = [df[c].quantile(0.75) for c in available_returns]
q90 = [df[c].quantile(0.9) for c in available_returns]

ax2.bar(x - 2*width, q10, width, label='10%分位', color='red', alpha=0.7)
ax2.bar(x - width, q25, width, label='25%分位', color='orange', alpha=0.7)
ax2.bar(x, q50, width, label='50%分位', color='green', alpha=0.7)
ax2.bar(x + width, q75, width, label='75%分位', color='blue', alpha=0.7)
ax2.bar(x + 2*width, q90, width, label='90%分位', color='purple', alpha=0.7)
ax2.set_xticks(x)
ax2.set_xticklabels(period_names)
ax2.set_title('收益分位数分布', fontsize=14)
ax2.set_ylabel('收益率')
ax2.legend()
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# 图2-3: 收益箱线图
ax3 = axes[1, 0]
box_data = [df[c].dropna().values for c in available_returns]
bp = ax3.boxplot(box_data, labels=period_names, patch_artist=True)
colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336', '#00BCD4', '#795548', '#607D8B']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax3.set_title('收益箱线图', fontsize=14)
ax3.set_ylabel('收益率')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# 图2-4: 分年度收益热力图
ax4 = axes[1, 1]
years = sorted(df['年份'].unique())
heatmap_data = []
for year in years:
    year_data = df[df['年份'] == year]
    row = [year_data[c].mean() for c in available_returns]
    heatmap_data.append(row)

heatmap_array = np.array(heatmap_data)
im = ax4.imshow(heatmap_array, cmap='RdYlGn', aspect='auto')
ax4.set_xticks(range(len(period_names)))
ax4.set_xticklabels(period_names, rotation=45)
ax4.set_yticks(range(len(years)))
ax4.set_yticklabels(years)
ax4.set_title('分年度平均收益率热力图', fontsize=14)
plt.colorbar(im, ax=ax4, label='平均收益率')

# 添加数值标签
for i in range(len(years)):
    for j in range(len(period_names)):
        ax4.text(j, i, f'{heatmap_array[i, j]:.3f}', ha='center', va='center', fontsize=8,
                color='white' if abs(heatmap_array[i, j]) > 0.05 else 'black')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '02_return_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ 图2已保存: 02_return_analysis.png")

# ============================================================================
# 4. 筛选分析
# ============================================================================
print("\n[Step 4] 筛选条件分析")
print("-" * 80)

# 筛选条件：增持金额>1000万 或 占比>1‰
df['增持目标金额_numeric'] = pd.to_numeric(df['增持目标金额'], errors='coerce')
df['增持占比_numeric'] = pd.to_numeric(df['增持占总市值比例'], errors='coerce')

# 条件1: 增持金额 > 1000万
mask_amount = df['增持目标金额_numeric'] > 10000000  # 1000万 = 10,000,000

# 条件2: 增持占比 > 1‰ (0.1%)
mask_ratio = df['增持占比_numeric'] > 0.001  # 1‰ = 0.001

# 合并条件
mask_filter = mask_amount | mask_ratio
df_filtered = df[mask_filter].copy()

print(f"\n📊 筛选结果:")
print(f"  原始事件数: {len(df)}")
print(f"  增持金额>1000万: {mask_amount.sum()}")
print(f"  增持占比>1‰: {mask_ratio.sum()}")
print(f"  满足任一条件: {len(df_filtered)} ({len(df_filtered)/len(df)*100:.1f}%)")

# 筛选后收益分析
print("\n📈 筛选后事件收益统计:")
if len(df_filtered) > 0:
    for col in available_returns:
        if col in df_filtered.columns:
            mean_ret = df_filtered[col].mean()
            median_ret = df_filtered[col].median()
            win_rate = (df_filtered[col] > 0).mean() * 100
            # 全市场对比
            all_mean = df[col].mean()
            print(f"  {col}: 筛选均值={mean_ret:.4f}, 筛选中位数={median_ret:.4f}, 胜率={win_rate:.1f}%, 全市场均值={all_mean:.4f}")

# 筛选条件细分分析
print("\n🔍 筛选条件细分:")
# 仅金额筛选
df_amount_only = df[mask_amount & ~mask_ratio]
print(f"  仅金额>1000万: {len(df_amount_only)} 个事件")

# 仅占比筛选
df_ratio_only = df[~mask_amount & mask_ratio]
print(f"  仅占比>1‰: {len(df_ratio_only)} 个事件")

# 双条件都满足
df_both = df[mask_amount & mask_ratio]
print(f"  双条件都满足: {len(df_both)} 个事件")

# 各子组收益对比
for label, sub_df in [('仅金额筛选', df_amount_only), ('仅占比筛选', df_ratio_only), ('双条件满足', df_both)]:
    if len(sub_df) > 0:
        print(f"\n  {label} ({len(sub_df)}个):")
        for col in available_returns:
            if col in sub_df.columns:
                mean_ret = sub_df[col].mean()
                median_ret = sub_df[col].median()
                win_rate = (sub_df[col] > 0).mean() * 100
                print(f"    {col}: 均值={mean_ret:.4f}, 中位数={median_ret:.4f}, 胜率={win_rate:.1f}%")

# 生成图3：筛选对比图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 图3-1: 筛选前后收益对比
ax1 = axes[0, 0]
x = np.arange(len(available_returns))
width = 0.35
all_means = [df[c].mean() for c in available_returns]
filtered_means = [df_filtered[c].mean() for c in available_returns]
ax1.bar(x - width/2, all_means, width, label='全部事件', color='steelblue')
ax1.bar(x + width/2, filtered_means, width, label='筛选事件', color='coral')
ax1.set_xticks(x)
ax1.set_xticklabels([c.replace('收益率', '') for c in available_returns], rotation=45)
ax1.set_title('筛选前后平均收益对比', fontsize=14)
ax1.set_ylabel('平均收益率')
ax1.legend()
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# 图3-2: 筛选条件分布
ax2 = axes[0, 1]
categories = ['全部事件', '金额>1000万', '占比>1‰', '双条件满足']
counts = [len(df), mask_amount.sum(), mask_ratio.sum(), (mask_amount & mask_ratio).sum()]
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
bars = ax2.bar(categories, counts, color=colors)
ax2.set_title('筛选条件事件数分布', fontsize=14)
ax2.set_ylabel('事件数')
for bar, count in zip(bars, counts):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5, str(count), ha='center', va='bottom')
ax2.tick_params(axis='x', rotation=15)

# 图3-3: 筛选后收益箱线图
ax3 = axes[1, 0]
if len(df_filtered) > 0:
    box_data_filtered = [df_filtered[c].dropna().values for c in available_returns]
    bp2 = ax3.boxplot(box_data_filtered, labels=[c.replace('收益率', '') for c in available_returns], patch_artist=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_title('筛选后事件收益箱线图', fontsize=14)
    ax3.set_ylabel('收益率')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# 图3-4: 筛选后分年度收益
ax4 = axes[1, 1]
if len(df_filtered) > 0:
    years_filtered = sorted(df_filtered['年份'].unique())
    heatmap_data_f = []
    for year in years_filtered:
        year_data = df_filtered[df_filtered['年份'] == year]
        row = [year_data[c].mean() for c in available_returns]
        heatmap_data_f.append(row)
    
    if heatmap_data_f:
        heatmap_array_f = np.array(heatmap_data_f)
        im2 = ax4.imshow(heatmap_array_f, cmap='RdYlGn', aspect='auto')
        ax4.set_xticks(range(len(available_returns)))
        ax4.set_xticklabels([c.replace('收益率', '') for c in available_returns], rotation=45)
        ax4.set_yticks(range(len(years_filtered)))
        ax4.set_yticklabels(years_filtered)
        ax4.set_title('筛选后分年度平均收益热力图', fontsize=14)
        plt.colorbar(im2, ax=ax4, label='平均收益率')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '03_filter_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ 图3已保存: 03_filter_analysis.png")

# ============================================================================
# 5. 特征分析
# ============================================================================
print("\n[Step 5] 高收益 vs 亏损事件特征分析")
print("-" * 80)

# 使用22日收益率作为基准（约1个月）
primary_return = '22日收益率'
if primary_return in df.columns:
    # 按收益率排序
    df_sorted = df.dropna(subset=[primary_return]).sort_values(primary_return, ascending=False)
    
    # 高收益事件（前20%）
    threshold_high = df_sorted[primary_return].quantile(0.8)
    df_high = df_sorted[df_sorted[primary_return] >= threshold_high]
    
    # 亏损事件（后20%）
    threshold_loss = df_sorted[primary_return].quantile(0.2)
    df_loss = df_sorted[df_sorted[primary_return] <= threshold_loss]
    
    print(f"\n📊 分组标准（基于{primary_return}）:")
    print(f"  高收益事件: {len(df_high)} 个 (收益率 >= {threshold_high:.4f})")
    print(f"  亏损事件: {len(df_loss)} 个 (收益率 <= {threshold_loss:.4f})")
    
    # 特征对比
    print("\n🔍 特征对比分析:")
    
    # 增持主体分布
    print("\n  增持主体分布:")
    print(f"    高收益组: {df_high['增持主体'].value_counts().to_dict()}")
    print(f"    亏损组: {df_loss['增持主体'].value_counts().to_dict()}")
    
    # 增持目的分布
    print("\n  增持目的分布:")
    print(f"    高收益组: {df_high['增持目的'].value_counts().to_dict()}")
    print(f"    亏损组: {df_loss['增持目的'].value_counts().to_dict()}")
    
    # 增持方式分布
    print("\n  增持方式分布:")
    print(f"    高收益组: {df_high['增持方式'].value_counts().to_dict()}")
    print(f"    亏损组: {df_loss['增持方式'].value_counts().to_dict()}")
    
    # 资金来源分布
    print("\n  资金来源分布:")
    print(f"    高收益组: {df_high['资金来源'].value_counts().to_dict()}")
    print(f"    亏损组: {df_loss['资金来源'].value_counts().to_dict()}")
    
    # 数值特征对比
    print("\n  数值特征对比:")
    for col in ['增持目标金额_numeric', '增持占比_numeric', '增持期限(月)']:
        if col in df_high.columns:
            high_mean = df_high[col].mean()
            loss_mean = df_loss[col].mean()
            if col == '增持目标金额_numeric':
                print(f"    {col}: 高收益组均值={high_mean:,.0f}, 亏损组均值={loss_mean:,.0f}")
            elif col == '增持占比_numeric':
                print(f"    {col}: 高收益组均值={high_mean*100:.4f}%, 亏损组均值={loss_mean*100:.4f}%")
            else:
                print(f"    {col}: 高收益组均值={high_mean:.1f}, 亏损组均值={loss_mean:.1f}")
    
    # 不同周期收益对比
    print("\n  各周期收益对比:")
    for col in available_returns:
        if col in df_high.columns:
            high_mean = df_high[col].mean()
            loss_mean = df_loss[col].mean()
            diff = high_mean - loss_mean
            print(f"    {col}: 高收益组均值={high_mean:.4f}, 亏损组均值={loss_mean:.4f}, 差异={diff:.4f}")
    
    # 生成图4：特征对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 图4-1: 增持主体分布对比
    ax1 = axes[0, 0]
    high_subject = df_high['增持主体'].value_counts()
    loss_subject = df_loss['增持主体'].value_counts()
    subjects = list(set(high_subject.index.tolist() + loss_subject.index.tolist()))
    x = np.arange(len(subjects))
    width = 0.35
    high_vals = [high_subject.get(s, 0) for s in subjects]
    loss_vals = [loss_subject.get(s, 0) for s in subjects]
    ax1.bar(x - width/2, high_vals, width, label='高收益组', color='green')
    ax1.bar(x + width/2, loss_vals, width, label='亏损组', color='red')
    ax1.set_xticks(x)
    ax1.set_xticklabels(subjects)
    ax1.set_title('增持主体分布对比', fontsize=14)
    ax1.legend()
    ax1.set_ylabel('事件数')
    
    # 图4-2: 增持目的分布对比
    ax2 = axes[0, 1]
    high_purpose = df_high['增持目的'].value_counts()
    loss_purpose = df_loss['增持目的'].value_counts()
    purposes = list(set(high_purpose.index.tolist() + loss_purpose.index.tolist()))
    x = np.arange(len(purposes))
    high_vals = [high_purpose.get(p, 0) for p in purposes]
    loss_vals = [loss_purpose.get(p, 0) for p in purposes]
    ax2.bar(x - width/2, high_vals, width, label='高收益组', color='green')
    ax2.bar(x + width/2, loss_vals, width, label='亏损组', color='red')
    ax2.set_xticks(x)
    ax2.set_xticklabels(purposes, rotation=30)
    ax2.set_title('增持目的分布对比', fontsize=14)
    ax2.legend()
    ax2.set_ylabel('事件数')
    
    # 图4-3: 金额和占比对比
    ax3 = axes[1, 0]
    metrics = ['增持目标金额(万)', '增持占比(‰)']
    high_amount = df_high['增持目标金额_numeric'].mean() / 10000  # 转换为万
    loss_amount = df_loss['增持目标金额_numeric'].mean() / 10000
    high_ratio = df_high['增持占比_numeric'].mean() * 1000  # 转换为‰
    loss_ratio = df_loss['增持占比_numeric'].mean() * 1000
    
    x = np.arange(2)
    ax3.bar(x - width/2, [high_amount, high_ratio], width, label='高收益组', color='green')
    ax3.bar(x + width/2, [loss_amount, loss_ratio], width, label='亏损组', color='red')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.set_title('金额和占比对比', fontsize=14)
    ax3.legend()
    ax3.set_ylabel('数值')
    for i, (h, l) in enumerate(zip([high_amount, high_ratio], [loss_amount, loss_ratio])):
        ax3.text(i - width/2, h + 10, f'{h:.1f}', ha='center', fontsize=9)
        ax3.text(i + width/2, l + 10, f'{l:.1f}', ha='center', fontsize=9)
    
    # 图4-4: 收益路径对比
    ax4 = axes[1, 1]
    periods = [c.replace('收益率', '') for c in available_returns]
    high_path = [df_high[c].mean() for c in available_returns]
    loss_path = [df_loss[c].mean() for c in available_returns]
    all_path = [df[c].mean() for c in available_returns]
    
    ax4.plot(periods, all_path, 'b-o', label='全部事件', linewidth=2)
    ax4.plot(periods, high_path, 'g-o', label='高收益组', linewidth=2)
    ax4.plot(periods, loss_path, 'r-o', label='亏损组', linewidth=2)
    ax4.set_title('收益路径对比', fontsize=14)
    ax4.set_ylabel('平均收益率')
    ax4.legend()
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, '04_feature_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✅ 图4已保存: 04_feature_comparison.png")
    
    # 高收益/亏损事件特征总结
    print("\n" + "=" * 80)
    print("📝 特征总结")
    print("=" * 80)
    
    print("\n🔆 高收益事件特征:")
    print(f"  1. 增持主体: {df_high['增持主体'].value_counts().index[0] if len(df_high) > 0 else 'N/A'} 占比较高")
    print(f"  2. 增持目的: {df_high['增持目的'].value_counts().index[0] if len(df_high) > 0 else 'N/A'} 为主")
    print(f"  3. 平均增持金额: {df_high['增持目标金额_numeric'].mean()/10000:,.1f} 万元")
    print(f"  4. 平均增持占比: {df_high['增持占比_numeric'].mean()*100:.4f}%")
    print(f"  5. 平均增持期限: {df_high['增持期限(月)'].mean():.1f} 个月")
    print(f"  6. 主要增持方式: {df_high['增持方式'].value_counts().index[0] if len(df_high) > 0 else 'N/A'}")
    
    print("\n🔻 亏损事件特征:")
    print(f"  1. 增持主体: {df_loss['增持主体'].value_counts().index[0] if len(df_loss) > 0 else 'N/A'} 占比较高")
    print(f"  2. 增持目的: {df_loss['增持目的'].value_counts().index[0] if len(df_loss) > 0 else 'N/A'} 为主")
    print(f"  3. 平均增持金额: {df_loss['增持目标金额_numeric'].mean()/10000:,.1f} 万元")
    print(f"  4. 平均增持占比: {df_loss['增持占比_numeric'].mean()*100:.4f}%")
    print(f"  5. 平均增持期限: {df_loss['增持期限(月)'].mean():.1f} 个月")
    print(f"  6. 主要增持方式: {df_loss['增持方式'].value_counts().index[0] if len(df_loss) > 0 else 'N/A'}")

# ============================================================================
# 6. 保存分析结果
# ============================================================================
print("\n[Step 6] 保存分析结果...")

# 保存汇总统计
summary = {
    '生成时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    '数据范围': f"{df['公告日期'].min()} ~ {df['公告日期'].max()}",
    '事件总数': len(df),
    '年度统计': yearly_stats.to_dict(),
    '收益统计': return_stats.to_dict(),
}

with open(os.path.join(OUTPUT_PATH, 'analysis_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

print(f"\n✅ 分析完成！")
print(f"📁 输出目录: {OUTPUT_PATH}")
print(f"📊 生成文件:")
for f in os.listdir(OUTPUT_PATH):
    print(f"   - {f}")
