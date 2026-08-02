"""
筛选后事件 - 滚动持仓数量时序分析
筛选条件：
1. 增持目的=主动
2. 排除资金来源仅含"其他"的
3. 排除增持方式仅含"协议转让/认购/定增"的
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

import rqdatac
rqdatac.init()

# ============================================================================
# 1. 加载数据
# ============================================================================
EVENT_PATH = r"F:\quant\research\holding_increase\increase_events"
OUTPUT_PATH = r"F:\quant\research\holding_increase\analysis_output"
os.makedirs(OUTPUT_PATH, exist_ok=True)

files = [f for f in os.listdir(EVENT_PATH) if f.endswith('.json')]
print(f"加载 {len(files)} 个事件文件...")

records = []
for f in files:
    with open(os.path.join(EVENT_PATH, f), 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    records.append(data)

df = pd.DataFrame(records)
df['公告日期'] = pd.to_datetime(df['公告日期'])
print(f"总事件数: {len(df)}")

# ============================================================================
# 2. 筛选
# ============================================================================
# 2.1 增持目的=主动
def is_active_purpose(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    s = str(val)
    if '被动' in s or '触发' in s or '稳定股价' in s:
        return False
    return True

df = df[df['增持目的'].apply(is_active_purpose)].copy()
print(f"筛选主动增持后: {len(df)}")

# 2.2 排除资金来源仅含"其他"的
def funding_only_other(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False  # 未知不排除
    parts = [s.strip() for s in str(val).split('|') if s.strip()]
    if not parts:
        return False
    # 所有部分都归为"其他"类
    for p in parts:
        if '自有' in p or '自筹' in p or '金融' in p or '专项贷' in p or '贷款' in p:
            return False
    # 如果全是"其他"/"未知"/"未披露"类，排除
    return True

df = df[~df['资金来源'].apply(funding_only_other)].copy()
print(f"排除资金来源仅其他后: {len(df)}")

# 2.3 排除增持方式仅含协议转让/认购/定增的
def method_only_excluded(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    parts = [s.strip() for s in str(val).split('|') if s.strip()]
    if not parts:
        return False
    excluded_keywords = ['协议转让', '认购', '定增', '定向增发']
    for p in parts:
        if not any(kw in p for kw in excluded_keywords):
            return False  # 有一个不在排除列表中的，保留
    return True  # 全部都在排除列表中

df = df[~df['增持方式'].apply(method_only_excluded)].copy()
print(f"排除协议转让/认购/定增后: {len(df)}")
print(f"\n最终筛选后事件数: {len(df)}")

# ============================================================================
# 3. 获取交易日历
# ============================================================================
start_date = df['公告日期'].min().strftime('%Y-%m-%d')
end_date = '2026-07-29'
trading_dates = rqdatac.get_trading_dates(start_date=start_date, end_date=end_date)
trading_dates = pd.to_datetime(trading_dates)
td_series = pd.Series(trading_dates)
print(f"交易日历: {len(trading_dates)} 天 ({trading_dates[0].strftime('%Y-%m-%d')} ~ {trading_dates[-1].strftime('%Y-%m-%d')})")

# 建立日期->索引映射
date_to_idx = {d: i for i, d in enumerate(trading_dates)}

# ============================================================================
# 4. 计算滚动持仓数量
# ============================================================================
def calc_rolling_positions(events_df, hold_days, trading_dates, date_to_idx):
    """
    计算每个交易日的持仓股票数
    事件公告日为D，买入日为D的下一交易日，持有hold_days个交易日
    """
    n_days = len(trading_dates)
    positions = np.zeros(n_days, dtype=int)

    for _, row in events_df.iterrows():
        ann_date = row['公告日期']
        # 找公告日在交易日历中的位置
        if ann_date not in date_to_idx:
            # 找最近的下一个交易日
            mask = trading_dates >= ann_date
            if not mask.any():
                continue
            ann_idx = mask.values.argmax()
        else:
            ann_idx = date_to_idx[ann_date]

        # 买入日 = 公告日下一交易日
        buy_idx = ann_idx + 1
        # 持有期: buy_idx 到 buy_idx + hold_days - 1
        sell_idx = buy_idx + hold_days - 1

        if buy_idx >= n_days:
            continue
        sell_idx = min(sell_idx, n_days - 1)

        positions[buy_idx:sell_idx + 1] += 1

    return positions

print("\n计算22日滚动持仓...")
pos_22 = calc_rolling_positions(df, 22, trading_dates, date_to_idx)
print(f"  最大同时持仓: {pos_22.max()} 只")
print(f"  平均持仓: {pos_22.mean():.1f} 只")

print("计算60日滚动持仓...")
pos_60 = calc_rolling_positions(df, 60, trading_dates, date_to_idx)
print(f"  最大同时持仓: {pos_60.max()} 只")
print(f"  平均持仓: {pos_60.mean():.1f} 只")

# ============================================================================
# 5. 绘图
# ============================================================================
fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

# 22日持仓
ax1 = axes[0]
ax1.plot(trading_dates, pos_22, color='steelblue', linewidth=0.8, alpha=0.9)
ax1.fill_between(trading_dates, pos_22, alpha=0.2, color='steelblue')
ax1.set_title(f'22日滚动持仓 - 持仓股票数随时间变化 (均值={pos_22.mean():.1f}, 峰值={pos_22.max()})', fontsize=13)
ax1.set_ylabel('持仓股票数')
ax1.axhline(y=pos_22.mean(), color='red', linestyle='--', linewidth=1, label=f'均值 {pos_22.mean():.1f}')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 60日持仓
ax2 = axes[1]
ax2.plot(trading_dates, pos_60, color='coral', linewidth=0.8, alpha=0.9)
ax2.fill_between(trading_dates, pos_60, alpha=0.2, color='coral')
ax2.set_title(f'60日滚动持仓 - 持仓股票数随时间变化 (均值={pos_60.mean():.1f}, 峰值={pos_60.max()})', fontsize=13)
ax2.set_ylabel('持仓股票数')
ax2.set_xlabel('日期')
ax2.axhline(y=pos_60.mean(), color='red', linestyle='--', linewidth=1, label=f'均值 {pos_60.mean():.1f}')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '06_rolling_positions.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图已保存: {os.path.join(OUTPUT_PATH, '06_rolling_positions.png')}")

# ============================================================================
# 6. 补充统计
# ============================================================================
print("\n" + "=" * 60)
print("持仓统计汇总")
print("=" * 60)
print(f"筛选后事件总数: {len(df)}")
print(f"\n22日滚动持仓:")
print(f"  平均持仓: {pos_22.mean():.1f} 只")
print(f"  中位数: {np.median(pos_22):.0f} 只")
print(f"  最大: {pos_22.max()} 只")
print(f"  持仓>0的天数占比: {(pos_22 > 0).sum() / len(pos_22) * 100:.1f}%")
print(f"\n60日滚动持仓:")
print(f"  平均持仓: {pos_60.mean():.1f} 只")
print(f"  中位数: {np.median(pos_60):.0f} 只")
print(f"  最大: {pos_60.max()} 只")
print(f"  持仓>0的天数占比: {(pos_60 > 0).sum() / len(pos_60) * 100:.1f}%")

# 按年度统计平均持仓
print("\n按年度平均持仓:")
pos_df = pd.DataFrame({'date': trading_dates, 'pos_22': pos_22, 'pos_60': pos_60})
pos_df['year'] = pos_df['date'].dt.year
for year, grp in pos_df.groupby('year'):
    print(f"  {year}年: 22日均持={grp['pos_22'].mean():.1f}, 60日均持={grp['pos_60'].mean():.1f}")
