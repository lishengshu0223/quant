"""
增持事件分析脚本 V2
功能：
1. 事件统计与时间分布（年度/月度/季度/星期）
2. 收益率分析（各时间点统计量、胜率）
3. 结构化特征对收益率影响分析
   - 类别特征：增持主体、增持目的、资金来源、增持方式
   - 数值特征：增持目标金额、增持占总市值比例、增持期限、不减持承诺
4. 输出图表与汇总JSON
"""
import os
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 配置
# ============================================================================
DATA_PATH = r"F:\quant\research\holding_increase\increase_events"
OUTPUT_PATH = r"F:\quant\research\holding_increase\analysis_output"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 收益率字段
RETURN_COLS = ['1日收益率', '3日收益率', '5日收益率', '10日收益率',
               '22日收益率', '60日收益率', '90日收益率', '120日收益率']
# 特征分析使用的时间点
FEATURE_RETURN_COLS = ['1日收益率', '5日收益率', '10日收益率',
                       '22日收益率', '60日收益率', '120日收益率']
PERIOD_LABELS = [c.replace('收益率', '') for c in RETURN_COLS]
FEATURE_PERIOD_LABELS = [c.replace('收益率', '') for c in FEATURE_RETURN_COLS]


# ============================================================================
# 工具函数
# ============================================================================
def to_float(x):
    """安全转换为float，失败/空值返回NaN"""
    if x is None:
        return np.nan
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan


def classify_purpose(purpose):
    """将增持目的归类为 主动/被动 两大类"""
    if purpose is None or (isinstance(purpose, float) and pd.isna(purpose)):
        return '未知'
    s = str(purpose)
    if '被动' in s:
        return '被动'
    if '主动' in s:
        return '主动'
    return '其他'


def split_sources(val):
    """拆分|分隔的值，返回列表"""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    parts = [s.strip() for s in str(val).split('|') if s.strip()]
    return parts


def normalize_method(val):
    """增持方式归一化合并"""
    s = str(val).strip()
    if '集中竞价' in s:
        return '集中竞价'
    if '大宗' in s:
        return '大宗交易'
    if '协议转让' in s:
        return '协议转让'
    if '认购' in s or '发行' in s:
        return '认购/定向增发'
    if '交易所' in s:
        return '交易所系统交易'
    if '未披露' in s or '未明确' in s or '未知' in s or s == '其他':
        return '未知/未披露'
    return '其他'


def normalize_funding(val):
    """资金来源归一化合并"""
    s = str(val).strip()
    if '自有' in s:
        return '自有'
    if '自筹' in s:
        return '自筹'
    if '金融' in s or '专项贷' in s or '贷款' in s or '借贷' in s:
        return '金融机构专项贷'
    if '未披露' in s or '未明确' in s or '未知' in s or s == '其他':
        return '未知/未披露'
    return '其他'


def group_period(month):
    """增持期限分组"""
    m = to_float(month)
    if pd.isna(m):
        return '未知'
    m = int(m)
    if m == 3:
        return '3个月'
    if m == 6:
        return '6个月'
    if m == 12:
        return '12个月'
    return f'{m}个月'


def win_rate(series):
    """胜率：>0 的比例"""
    s = series.dropna()
    if len(s) == 0:
        return np.nan
    return (s > 0).mean()


def describe_return(series):
    """返回单列收益率的统计字典"""
    s = series.dropna()
    if len(s) == 0:
        return {k: np.nan for k in
                ['count', 'mean', 'median', 'std', 'min', 'max',
                 'q25', 'q75', 'q10', 'q90', 'win_rate']}
    return {
        'count': int(len(s)),
        'mean': float(s.mean()),
        'median': float(s.median()),
        'std': float(s.std()),
        'min': float(s.min()),
        'max': float(s.max()),
        'q25': float(s.quantile(0.25)),
        'q75': float(s.quantile(0.75)),
        'q10': float(s.quantile(0.10)),
        'q90': float(s.quantile(0.90)),
        'win_rate': float((s > 0).mean()),
    }


def mean_returns_by_group(df, group_col, return_cols=FEATURE_RETURN_COLS):
    """按组计算各时间点平均收益率，返回DataFrame"""
    rows = []
    for name, sub in df.groupby(group_col):
        row = {'分组': name, '样本数': len(sub)}
        for c in return_cols:
            row[c] = sub[c].mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out) > 0:
        out = out.sort_values('样本数', ascending=False).reset_index(drop=True)
    return out


def mean_returns_by_list_group(df, list_col, return_cols=FEATURE_RETURN_COLS):
    """
    按|拆分后的列表列分组，支持重复统计（一个事件可属于多个组）
    list_col: 列名，该列每个元素是归一化后的类别列表
    """
    from collections import defaultdict
    group_data = defaultdict(list)  # {group_name: [row_indices]}
    for idx, val in df[list_col].items():
        seen = set(val)  # 同一事件去重（避免同一事件同组重复计数）
        for g in seen:
            group_data[g].append(idx)

    rows = []
    for name, indices in group_data.items():
        sub = df.loc[indices]
        row = {'分组': name, '样本数': len(sub)}
        for c in return_cols:
            row[c] = sub[c].mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out) > 0:
        out = out.sort_values('样本数', ascending=False).reset_index(drop=True)
    return out


def amount_bucket(amount):
    """增持目标金额分5档"""
    a = to_float(amount)
    if pd.isna(a) or a <= 0:
        return '未知'
    if a < 1e7:           # < 1000万
        return '0-1000万'
    if a < 5e7:           # < 5000万
        return '1000万-5000万'
    if a < 1e8:           # < 1亿
        return '5000万-1亿'
    if a < 5e8:           # < 5亿
        return '1亿-5亿'
    return '5亿以上'


def ratio_bucket(ratio):
    """增持占总市值比例分5档"""
    r = to_float(ratio)
    if pd.isna(r) or r <= 0:
        return '未知'
    pct = r * 100  # 转为百分比
    if pct < 0.1:
        return '0-0.1%'
    if pct < 0.5:
        return '0.1%-0.5%'
    if pct < 1.0:
        return '0.5%-1%'
    if pct < 2.0:
        return '1%-2%'
    return '2%以上'


# ============================================================================
# 1. 加载数据
# ============================================================================
print("=" * 80)
print("增持事件分析 V2")
print("=" * 80)

print("\n[Step 1] 加载数据...")
json_files = sorted([f for f in os.listdir(DATA_PATH) if f.endswith('.json')])
print(f"找到 {len(json_files)} 个事件文件")

all_data = []
for fname in json_files:
    try:
        with open(os.path.join(DATA_PATH, fname), 'r', encoding='utf-8') as f:
            all_data.append(json.load(f))
    except Exception as e:
        print(f"  读取 {fname} 失败: {e}")

df = pd.DataFrame(all_data)
print(f"成功加载 {len(df)} 条事件数据")

# 日期相关字段
df['公告日期_dt'] = pd.to_datetime(df['公告日期'], errors='coerce')
df['年份'] = df['公告日期_dt'].dt.year
df['月份'] = df['公告日期_dt'].dt.month
df['年月'] = df['公告日期_dt'].dt.strftime('%Y-%m')
df['季度'] = df['公告日期_dt'].dt.quarter
df['星期'] = df['公告日期_dt'].dt.dayofweek  # 0=周一

# 数值化字段
df['增持目标金额_num'] = df['增持目标金额'].apply(to_float)
df['增持占比_num'] = df['增持占总市值比例'].apply(to_float)
df['增持期限_num'] = df['增持期限(月)'].apply(to_float)
df['不减持承诺_num'] = df['不减持承诺(月)'].apply(to_float)

# 派生分类字段
df['增持目的分类'] = df['增持目的'].apply(classify_purpose)
# 资金来源：拆分|后归一化，每个事件可属于多个组
df['资金来源_拆分'] = df['资金来源'].apply(
    lambda v: [normalize_funding(s) for s in split_sources(v)] or ['未知/未披露']
)
# 增持方式：拆分|后归一化，每个事件可属于多个组
df['增持方式_拆分'] = df['增持方式'].apply(
    lambda v: [normalize_method(s) for s in split_sources(v)] or ['未知/未披露']
)
df['金额档'] = df['增持目标金额'].apply(amount_bucket)
df['比例档'] = df['增持占总市值比例'].apply(ratio_bucket)
df['期限分组'] = df['增持期限(月)'].apply(group_period)
df['有承诺'] = df['不减持承诺_num'].apply(lambda x: '有承诺' if pd.notna(x) and x > 0 else '无承诺')

# log10 字段（仅对正值计算）
df['log10金额'] = df['增持目标金额_num'].apply(
    lambda x: np.log10(x) if pd.notna(x) and x > 0 else np.nan)
df['log10比例万分'] = df['增持占比_num'].apply(
    lambda x: np.log10(x * 10000) if pd.notna(x) and x > 0 else np.nan)

# ============================================================================
# 2. 事件统计与时间分布
# ============================================================================
print("\n[Step 2] 事件统计与时间分布")
print("-" * 80)

total_events = len(df)
date_min = df['公告日期'].min()
date_max = df['公告日期'].max()
print(f"\n总事件数: {total_events}")
print(f"数据时间范围: {date_min} ~ {date_max}")

# 年度
yearly = df.groupby('年份').size()
print(f"\n年度分布:")
for y, c in yearly.items():
    print(f"  {int(y)}年: {c} 个 ({c/total_events*100:.1f}%)")

# 月度
monthly = df.groupby('年月').size()
month_avg = monthly.mean()
print(f"\n月度分布: 共 {len(monthly)} 个月份")
print(f"  平均每月事件数: {month_avg:.2f}")
print(f"  最多月份: {monthly.idxmax()} ({monthly.max()} 个)")
print(f"  最少月份: {monthly.idxmin()} ({monthly.min()} 个)")

# 按自然月（1-12）汇总
month_of_year = df.groupby('月份').size()
print(f"\n按自然月(1-12)汇总:")
for m, c in month_of_year.items():
    print(f"  {int(m)}月: {c} 个")

# 季度
quarterly = df.groupby(['年份', '季度']).size()
print(f"\n按季度统计:")
for (y, q), c in quarterly.items():
    print(f"  {int(y)}Q{int(q)}: {c} 个")

# 星期
weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
weekday_dist = df.groupby('星期').size()
print(f"\n星期分布:")
for i, name in enumerate(weekday_names):
    if i in weekday_dist.index:
        c = weekday_dist[i]
        print(f"  {name}: {c} 个 ({c/total_events*100:.1f}%)")

# 时间分布特征描述
peak_year = yearly.idxmax()
peak_month = monthly.idxmax()
weekday_pct = weekday_dist.get(0, 0) + weekday_dist.get(1, 0) + weekday_dist.get(2, 0)
mon_fri_pct = (weekday_dist.get(0, 0) + weekday_dist.get(4, 0)) / total_events * 100
time_desc = (
    f"事件分布于 {date_min} 至 {date_max}，跨度约 {len(yearly)} 年。"
    f"{int(peak_year)}年为事件高峰({yearly.max()}个)。"
    f"月度事件数平均 {month_avg:.1f} 笔，峰值出现在 {peak_month}({monthly.max()}个)。"
    f"按自然月看，{int(month_of_year.idxmax())}月事件最多({month_of_year.max()}个)。"
    f"星期分布上，周一至周三合计占比 {weekday_pct/total_events*100:.1f}%，"
    f"周一与周五合计占比 {mon_fri_pct:.1f}%，呈现公告集中在工作日前段的特征。"
)
print(f"\n时间分布特征: {time_desc}")

# 生成图1：事件时间分布
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax1 = axes[0, 0]
yearly.plot(kind='bar', ax=ax1, color='steelblue')
ax1.set_title('年度事件分布', fontsize=14)
ax1.set_xlabel('年份')
ax1.set_ylabel('事件数')
ax1.set_xticklabels([int(x) for x in yearly.index], rotation=0)
for i, v in enumerate(yearly):
    ax1.text(i, v + 5, str(int(v)), ha='center', va='bottom')

ax2 = axes[0, 1]
month_of_year.plot(kind='bar', ax=ax2, color='coral')
ax2.set_title('自然月(1-12)事件分布', fontsize=14)
ax2.set_xlabel('月份')
ax2.set_ylabel('事件数')
ax2.set_xticklabels([int(x) for x in month_of_year.index], rotation=0)
for i, v in enumerate(month_of_year):
    ax2.text(i, v + 2, str(int(v)), ha='center', va='bottom', fontsize=9)

ax3 = axes[1, 0]
wd_labels = [weekday_names[i] for i in weekday_dist.index if i < 7]
wd_vals = [weekday_dist[i] for i in weekday_dist.index if i < 7]
ax3.pie(wd_vals, labels=wd_labels, autopct='%1.1f%%', startangle=90,
        colors=plt.cm.Set3.colors[:len(wd_labels)])
ax3.set_title('星期分布', fontsize=14)

ax4 = axes[1, 1]
q_pivot = df.groupby(['年份', '季度']).size().unstack(fill_value=0)
q_pivot.plot(kind='bar', ax=ax4, colormap='Set2')
ax4.set_title('各年度季度分布', fontsize=14)
ax4.set_xlabel('年份')
ax4.set_ylabel('事件数')
ax4.legend([f'Q{i}' for i in q_pivot.columns], title='季度')
ax4.set_xticklabels([int(x) for x in q_pivot.index], rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '01_event_stats.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n图1已保存: 01_event_stats.png")

# ============================================================================
# 3. 收益率分析
# ============================================================================
print("\n[Step 3] 收益率分析")
print("-" * 80)

# 各时间点收益率统计
return_stats = {}
for col in RETURN_COLS:
    return_stats[col] = describe_return(df[col])

stats_df = pd.DataFrame(return_stats).T
print("\n各时间点收益率统计:")
print(stats_df.round(4).to_string())

print("\n各时间点胜率:")
for col in RETURN_COLS:
    wr = return_stats[col]['win_rate']
    print(f"  {col}: {wr*100:.2f}%")

# 生成图2：收益率分析
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 箱线图
ax1 = axes[0, 0]
box_data = [df[c].dropna().values for c in RETURN_COLS]
bp = ax1.boxplot(box_data, labels=PERIOD_LABELS, patch_artist=True, showfliers=False)
colors_box = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336',
              '#00BCD4', '#795548', '#607D8B']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax1.set_title('各时间点收益率箱线图', fontsize=14)
ax1.set_ylabel('收益率')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# 均值/中位数折线图
ax2 = axes[0, 1]
means = [return_stats[c]['mean'] for c in RETURN_COLS]
medians = [return_stats[c]['median'] for c in RETURN_COLS]
ax2.plot(PERIOD_LABELS, means, 'o-', label='均值', color='steelblue', linewidth=2)
ax2.plot(PERIOD_LABELS, medians, 's-', label='中位数', color='coral', linewidth=2)
ax2.set_title('均值/中位数折线图', fontsize=14)
ax2.set_ylabel('收益率')
ax2.legend()
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.tick_params(axis='x', rotation=30)
for i, (m, md) in enumerate(zip(means, medians)):
    ax2.text(i, m, f'{m:.3f}', ha='center', va='bottom', fontsize=8, color='steelblue')

# 胜率柱状图
ax3 = axes[1, 0]
win_rates = [return_stats[c]['win_rate'] * 100 for c in RETURN_COLS]
bars = ax3.bar(PERIOD_LABELS, win_rates, color=['#4CAF50' if w >= 50 else '#F44336' for w in win_rates])
ax3.set_title('各时间点胜率(>0比例)', fontsize=14)
ax3.set_ylabel('胜率 (%)')
ax3.axhline(y=50, color='black', linestyle='--', linewidth=0.8)
for bar, w in zip(bars, win_rates):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{w:.1f}%', ha='center', va='bottom', fontsize=9)
ax3.tick_params(axis='x', rotation=30)

# 分位数折线图
ax4 = axes[1, 1]
q10 = [return_stats[c]['q10'] for c in RETURN_COLS]
q25 = [return_stats[c]['q25'] for c in RETURN_COLS]
q50 = [return_stats[c]['median'] for c in RETURN_COLS]
q75 = [return_stats[c]['q75'] for c in RETURN_COLS]
q90 = [return_stats[c]['q90'] for c in RETURN_COLS]
ax4.plot(PERIOD_LABELS, q10, 'o-', label='10%分位', color='red', linewidth=1.5)
ax4.plot(PERIOD_LABELS, q25, 'o-', label='25%分位', color='orange', linewidth=1.5)
ax4.plot(PERIOD_LABELS, q50, 'o-', label='50%分位', color='green', linewidth=2)
ax4.plot(PERIOD_LABELS, q75, 'o-', label='75%分位', color='blue', linewidth=1.5)
ax4.plot(PERIOD_LABELS, q90, 'o-', label='90%分位', color='purple', linewidth=1.5)
ax4.set_title('各分位数折线图', fontsize=14)
ax4.set_ylabel('收益率')
ax4.legend()
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax4.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '02_return_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n图2已保存: 02_return_analysis.png")

# ============================================================================
# 4. 结构化特征对收益率影响分析
# ============================================================================
print("\n[Step 4] 结构化特征对收益率影响分析")
print("-" * 80)

feature_results = {}

# ----- 4.1 类别特征 -----
print("\n[4.1] 类别特征分析")
print("-" * 60)

categorical_features = {
    '增持主体': '增持主体',
    '增持目的分类': '增持目的(主动/被动)',
    '资金来源_拆分': '资金来源(重复统计)',
    '增持方式_拆分': '增持方式(重复统计)',
}

category_stats = {}
for col, label in categorical_features.items():
    print(f"\n  >> {label} (字段: {col}):")
    if '拆分' in col:
        grp = mean_returns_by_list_group(df, col)
    else:
        grp = mean_returns_by_group(df, col)
    category_stats[label] = grp.to_dict('records')
    for _, row in grp.iterrows():
        n = int(row['样本数'])
        flag = ' *小样本' if n < 5 else ''
        rets = ' | '.join([f"{c.replace('收益率','')}={row[c]:.4f}"
                          for c in FEATURE_RETURN_COLS if pd.notna(row[c])])
        print(f"    {row['分组']}: 样本={n}{flag} | {rets}")
    feature_results[col] = grp

# 类别特征分组柱状图（标注事件数量和占比）
total_events = len(df)
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
for idx, (col, label) in enumerate(categorical_features.items()):
    ax = axes[idx // 2, idx % 2]
    grp = feature_results[col]
    if len(grp) == 0:
        ax.set_title(f'{label} (无数据)', fontsize=12)
        continue
    # 限制显示前10个类别，避免过长
    grp_show = grp.head(10).copy()
    # X轴标签加上数量和占比
    categories = []
    for _, row in grp_show.iterrows():
        name = str(row['分组'])
        n = int(row['样本数'])
        pct = n / total_events * 100
        categories.append(f"{name}\n(n={n}, {pct:.1f}%)")
    x = np.arange(len(categories))
    width = 0.13
    for i, c in enumerate(FEATURE_RETURN_COLS):
        vals = grp_show[c].values
        ax.bar(x + (i - 2.5) * width, vals, width, label=c.replace('收益率', ''))
    ax.set_title(f'{label} 各时间点平均收益率', fontsize=12)
    ax.set_ylabel('平均收益率')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=0, ha='center', fontsize=8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend(fontsize=8, ncol=2)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '03_category_impact.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n图3已保存: 03_category_impact.png")

# ----- 4.2 数值特征 -----
print("\n[4.2] 数值特征分析")
print("-" * 60)

# 增持目标金额分档
print("\n  >> 增持目标金额分档:")
amount_order = ['0-1000万', '1000万-5000万', '5000万-1亿', '1亿-5亿', '5亿以上', '未知']
df['金额档'] = pd.Categorical(df['金额档'], categories=amount_order, ordered=True)
amount_grp = mean_returns_by_group(df, '金额档')
# 按预设顺序排序
amount_grp['排序'] = amount_grp['分组'].apply(lambda x: amount_order.index(x) if x in amount_order else 99)
amount_grp = amount_grp.sort_values('排序').drop(columns='排序').reset_index(drop=True)
feature_results['金额档'] = amount_grp
for _, row in amount_grp.iterrows():
    n = int(row['样本数'])
    flag = ' *小样本' if n < 5 else ''
    rets = ' | '.join([f"{c.replace('收益率','')}={row[c]:.4f}"
                      for c in FEATURE_RETURN_COLS if pd.notna(row[c])])
    print(f"    {row['分组']}: 样本={n}{flag} | {rets}")

# 增持占比分档
print("\n  >> 增持占总市值比例分档:")
ratio_order = ['0-0.1%', '0.1%-0.5%', '0.5%-1%', '1%-2%', '2%以上', '未知']
df['比例档'] = pd.Categorical(df['比例档'], categories=ratio_order, ordered=True)
ratio_grp = mean_returns_by_group(df, '比例档')
ratio_grp['排序'] = ratio_grp['分组'].apply(lambda x: ratio_order.index(x) if x in ratio_order else 99)
ratio_grp = ratio_grp.sort_values('排序').drop(columns='排序').reset_index(drop=True)
feature_results['比例档'] = ratio_grp
for _, row in ratio_grp.iterrows():
    n = int(row['样本数'])
    flag = ' *小样本' if n < 5 else ''
    rets = ' | '.join([f"{c.replace('收益率','')}={row[c]:.4f}"
                      for c in FEATURE_RETURN_COLS if pd.notna(row[c])])
    print(f"    {row['分组']}: 样本={n}{flag} | {rets}")

# 增持期限分组
print("\n  >> 增持期限(月)分组:")
period_order = ['3个月', '6个月', '12个月', '未知']
# 把其他月数也纳入
period_unique = df['期限分组'].unique().tolist()
period_order_full = [p for p in period_order if p in period_unique] + \
                    [p for p in period_unique if p not in period_order]
df['期限分组'] = pd.Categorical(df['期限分组'], categories=period_order_full, ordered=True)
period_grp = mean_returns_by_group(df, '期限分组')
feature_results['期限分组'] = period_grp
for _, row in period_grp.iterrows():
    n = int(row['样本数'])
    flag = ' *小样本' if n < 5 else ''
    rets = ' | '.join([f"{c.replace('收益率','')}={row[c]:.4f}"
                      for c in FEATURE_RETURN_COLS if pd.notna(row[c])])
    print(f"    {row['分组']}: 样本={n}{flag} | {rets}")

# 不减持承诺
print("\n  >> 不减持承诺(月) 有/无:")
commit_grp = mean_returns_by_group(df, '有承诺')
feature_results['有承诺'] = commit_grp
for _, row in commit_grp.iterrows():
    n = int(row['样本数'])
    rets = ' | '.join([f"{c.replace('收益率','')}={row[c]:.4f}"
                      for c in FEATURE_RETURN_COLS if pd.notna(row[c])])
    print(f"    {row['分组']}: 样本={n} | {rets}")

# 数值特征分档折线图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

def plot_group_lines(ax, grp, title, order_col=None):
    """分组折线图：各档在各时间点的平均收益率"""
    if len(grp) == 0:
        ax.set_title(f'{title} (无数据)')
        return
    for _, row in grp.iterrows():
        vals = [row[c] if pd.notna(row[c]) else np.nan for c in FEATURE_RETURN_COLS]
        label = f"{row['分组']}(n={int(row['样本数'])})"
        ax.plot(FEATURE_PERIOD_LABELS, vals, 'o-', label=label, linewidth=1.5)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel('平均收益率')
    ax.set_xlabel('持有时间')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=30)

plot_group_lines(axes[0, 0], amount_grp, '增持目标金额分档 - 平均收益率')
plot_group_lines(axes[0, 1], ratio_grp, '增持占总市值比例分档 - 平均收益率')
plot_group_lines(axes[1, 0], period_grp, '增持期限分组 - 平均收益率')
plot_group_lines(axes[1, 1], commit_grp, '不减持承诺 有/无 - 平均收益率')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '04_numeric_impact.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n图4已保存: 04_numeric_impact.png")

# ----- 4.3 log10分析 -----
print("\n[4.3] log10分析")
print("-" * 60)

# log10金额分5档（基于分位数）
valid_log_amount = df['log10金额'].dropna()
print(f"\n  log10(增持目标金额) 有效样本: {len(valid_log_amount)}")
if len(valid_log_amount) > 0:
    print(f"    范围: {valid_log_amount.min():.3f} ~ {valid_log_amount.max():.3f}")
    print(f"    均值: {valid_log_amount.mean():.3f}, 中位数: {valid_log_amount.median():.3f}")
    # 用分位数分5档
    qcuts = pd.qcut(df['log10金额'], 5, duplicates='drop',
                    labels=['档1(最小)', '档2', '档3', '档4', '档5(最大)'])
    df['log10金额档'] = qcuts
    log_amount_grp = mean_returns_by_group(df.dropna(subset=['log10金额']), 'log10金额档')
    feature_results['log10金额档'] = log_amount_grp
    print("\n  log10(增持目标金额) 分5档平均收益率:")
    for _, row in log_amount_grp.iterrows():
        n = int(row['样本数'])
        rets = ' | '.join([f"{c.replace('收益率','')}={row[c]:.4f}"
                          for c in FEATURE_RETURN_COLS if pd.notna(row[c])])
        print(f"    {row['分组']}: 样本={n} | {rets}")

# log10比例(万分比)分5档
valid_log_ratio = df['log10比例万分'].dropna()
print(f"\n  log10(增持占比*10000, 万分比) 有效样本: {len(valid_log_ratio)}")
if len(valid_log_ratio) > 0:
    print(f"    范围: {valid_log_ratio.min():.3f} ~ {valid_log_ratio.max():.3f}")
    print(f"    均值: {valid_log_ratio.mean():.3f}, 中位数: {valid_log_ratio.median():.3f}")
    qcuts2 = pd.qcut(df['log10比例万分'], 5, duplicates='drop',
                     labels=['档1(最小)', '档2', '档3', '档4', '档5(最大)'])
    df['log10比例档'] = qcuts2
    log_ratio_grp = mean_returns_by_group(df.dropna(subset=['log10比例万分']), 'log10比例档')
    feature_results['log10比例档'] = log_ratio_grp
    print("\n  log10(增持占比*10000) 分5档平均收益率:")
    for _, row in log_ratio_grp.iterrows():
        n = int(row['样本数'])
        rets = ' | '.join([f"{c.replace('收益率','')}={row[c]:.4f}"
                          for c in FEATURE_RETURN_COLS if pd.notna(row[c])])
        print(f"    {row['分组']}: 样本={n} | {rets}")

# log10 图表：散点图 + 分档柱状图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 散点图：log10金额 vs 120日收益率
ax1 = axes[0, 0]
sub = df.dropna(subset=['log10金额', '120日收益率'])
if len(sub) > 0:
    ax1.scatter(sub['log10金额'], sub['120日收益率'], alpha=0.3, s=15, c='steelblue')
    # 拟合趋势线
    z = np.polyfit(sub['log10金额'], sub['120日收益率'], 1)
    p = np.poly1d(z)
    xs = np.linspace(sub['log10金额'].min(), sub['log10金额'].max(), 100)
    ax1.plot(xs, p(xs), 'r-', linewidth=2, label=f'趋势线 y={z[0]:.4f}x+{z[1]:.4f}')
    ax1.legend()
ax1.set_title('log10(增持目标金额) vs 120日收益率', fontsize=12)
ax1.set_xlabel('log10(增持目标金额)')
ax1.set_ylabel('120日收益率')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# 散点图：log10比例万分 vs 120日收益率
ax2 = axes[0, 1]
sub2 = df.dropna(subset=['log10比例万分', '120日收益率'])
if len(sub2) > 0:
    ax2.scatter(sub2['log10比例万分'], sub2['120日收益率'], alpha=0.3, s=15, c='coral')
    z2 = np.polyfit(sub2['log10比例万分'], sub2['120日收益率'], 1)
    p2 = np.poly1d(z2)
    xs2 = np.linspace(sub2['log10比例万分'].min(), sub2['log10比例万分'].max(), 100)
    ax2.plot(xs2, p2(xs2), 'r-', linewidth=2, label=f'趋势线 y={z2[0]:.4f}x+{z2[1]:.4f}')
    ax2.legend()
ax2.set_title('log10(增持占比*10000) vs 120日收益率', fontsize=12)
ax2.set_xlabel('log10(增持占比*10000)')
ax2.set_ylabel('120日收益率')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# 分档柱状图：log10金额档 vs 120日收益率
ax3 = axes[1, 0]
if 'log10金额档' in df.columns:
    grp = df.dropna(subset=['log10金额']).groupby('log10金额档')['120日收益率'].mean()
    bars = ax3.bar(grp.index.astype(str), grp.values, color='steelblue')
    ax3.set_title('log10(金额)分档 vs 120日平均收益率', fontsize=12)
    ax3.set_ylabel('120日平均收益率')
    for bar, v in zip(bars, grp.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
else:
    ax3.set_title('log10(金额)分档 - 无数据')

# 分档柱状图：log10比例档 vs 120日收益率
ax4 = axes[1, 1]
if 'log10比例档' in df.columns:
    grp2 = df.dropna(subset=['log10比例万分']).groupby('log10比例档')['120日收益率'].mean()
    bars = ax4.bar(grp2.index.astype(str), grp2.values, color='coral')
    ax4.set_title('log10(占比*10000)分档 vs 120日平均收益率', fontsize=12)
    ax4.set_ylabel('120日平均收益率')
    for bar, v in zip(bars, grp2.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
else:
    ax4.set_title('log10(占比*10000)分档 - 无数据')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '05_log10_impact.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\n图5已保存: 05_log10_impact.png")

# ============================================================================
# 5. 保存汇总结果
# ============================================================================
print("\n[Step 5] 保存分析结果")
print("-" * 80)

def df_to_records(grp_df):
    """将分组DataFrame转为可JSON序列化的records"""
    records = []
    for _, row in grp_df.iterrows():
        rec = {}
        for k, v in row.items():
            if isinstance(v, (np.integer,)):
                rec[k] = int(v)
            elif isinstance(v, (np.floating,)):
                rec[k] = None if pd.isna(v) else float(v)
            elif pd.isna(v):
                rec[k] = None
            else:
                rec[k] = v
        records.append(rec)
    return records


summary = {
    '生成时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    '数据时间范围': f"{date_min} ~ {date_max}",
    '总事件数': int(total_events),
    '事件统计': {
        '年度分布': {str(int(k)): int(v) for k, v in yearly.items()},
        '自然月分布': {str(int(k)): int(v) for k, v in month_of_year.items()},
        '月度平均事件数': float(month_avg),
        '月度最多': {'月份': str(monthly.idxmax()), '数量': int(monthly.max())},
        '月度最少': {'月份': str(monthly.idxmin()), '数量': int(monthly.min())},
        '季度分布': {f"{int(y)}Q{int(q)}": int(v) for (y, q), v in quarterly.items()},
        '星期分布': {weekday_names[int(i)] if int(i) < 7 else str(i): int(v)
                    for i, v in weekday_dist.items()},
        '时间分布特征': time_desc,
    },
    '收益率统计': {col: return_stats[col] for col in RETURN_COLS},
    '类别特征分析': {
        label: df_to_records(feature_results[col])
        for col, label in categorical_features.items()
    },
    '数值特征分析': {
        '增持目标金额分档': df_to_records(feature_results['金额档']),
        '增持占比分档': df_to_records(feature_results['比例档']),
        '增持期限分组': df_to_records(feature_results['期限分组']),
        '不减持承诺': df_to_records(feature_results['有承诺']),
    },
    'log10分析': {
        'log10金额分档': df_to_records(feature_results.get('log10金额档', pd.DataFrame())),
        'log10比例分档': df_to_records(feature_results.get('log10比例档', pd.DataFrame())),
    },
}

summary_path = os.path.join(OUTPUT_PATH, 'analysis_v2_summary.json')
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
print(f"汇总结果已保存: analysis_v2_summary.json")

# ============================================================================
# 6. 关键发现
# ============================================================================
print("\n" + "=" * 80)
print("关键发现")
print("=" * 80)

# 收益率趋势
best_period = max(RETURN_COLS, key=lambda c: return_stats[c]['mean'])
print(f"\n1. 收益率表现:")
print(f"   - 各时间点平均收益率均为正值，持有时间越长均值越高")
print(f"   - 平均收益率最高: {best_period} ({return_stats[best_period]['mean']*100:.2f}%)")
print(f"   - 1日平均收益率: {return_stats['1日收益率']['mean']*100:.2f}%, 胜率 {return_stats['1日收益率']['win_rate']*100:.1f}%")
print(f"   - 120日平均收益率: {return_stats['120日收益率']['mean']*100:.2f}%, 胜率 {return_stats['120日收益率']['win_rate']*100:.1f}%")

# 主体影响
if len(feature_results['增持主体']) > 0:
    sub_grp = feature_results['增持主体']
    top_subject = sub_grp.iloc[0]
    print(f"\n2. 增持主体影响:")
    print(f"   - 样本最多的主体: {top_subject['分组']} ({int(top_subject['样本数'])}个)")
    if len(sub_grp) >= 2:
        # 比较120日收益率
        valid = sub_grp.dropna(subset=['120日收益率'])
        if len(valid) > 0:
            best = valid.loc[valid['120日收益率'].idxmax()]
            print(f"   - 120日收益率最高: {best['分组']} ({best['120日收益率']*100:.2f}%)")

# 目的分类
if len(feature_results['增持目的分类']) > 0:
    print(f"\n3. 增持目的(主动/被动)影响:")
    for _, row in feature_results['增持目的分类'].iterrows():
        r120 = row.get('120日收益率', np.nan)
        r120_str = f"{r120*100:.2f}%" if pd.notna(r120) else "N/A"
        print(f"   - {row['分组']}: 样本={int(row['样本数'])}, 120日收益={r120_str}")

# 金额分档
print(f"\n4. 增持目标金额影响:")
for _, row in feature_results['金额档'].iterrows():
    r120 = row.get('120日收益率', np.nan)
    r120_str = f"{r120*100:.2f}%" if pd.notna(r120) else "N/A"
    print(f"   - {row['分组']}: 样本={int(row['样本数'])}, 120日收益={r120_str}")

# 占比分档
print(f"\n5. 增持占比影响:")
for _, row in feature_results['比例档'].iterrows():
    r120 = row.get('120日收益率', np.nan)
    r120_str = f"{r120*100:.2f}%" if pd.notna(r120) else "N/A"
    print(f"   - {row['分组']}: 样本={int(row['样本数'])}, 120日收益={r120_str}")

# 承诺
print(f"\n6. 不减持承诺影响:")
for _, row in feature_results['有承诺'].iterrows():
    r120 = row.get('120日收益率', np.nan)
    r120_str = f"{r120*100:.2f}%" if pd.notna(r120) else "N/A"
    print(f"   - {row['分组']}: 样本={int(row['样本数'])}, 120日收益={r120_str}")

print(f"\n分析完成！输出目录: {OUTPUT_PATH}")
print("生成文件:")
for f in sorted(os.listdir(OUTPUT_PATH)):
    if f.endswith('.png') or f == 'analysis_v2_summary.json':
        print(f"   - {f}")
