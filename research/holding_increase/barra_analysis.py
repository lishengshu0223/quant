"""
Barra风格暴露分析 - 基于已完成的回测结果
读取回测持仓，计算组合/指数的Barra风格暴露，绘制对比图
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

import rqdatac
rqdatac.init()

sys.path.insert(0, r"F:\quant")
from local_api.barra import get_factor_exposure

# ============================================================================
# 配置
# ============================================================================
OUTPUT_PATH = r"F:\quant\research\holding_increase\analysis_output"
BACKTEST_OUTPUT = r"F:\quant\research\holding_increase\backtest_results"
START_DATE = "2023-01-01"
END_DATE = "2026-07-29"

# 本地Barra数据的实际列名（小写蛇形）
FACTORS = ["beta", "book_to_price", "earnings_yield", "growth", "leverage",
           "liquidity", "momentum", "non_linear_size", "residual_volatility", "size"]

FACTOR_CN = {
    "beta": "Beta",
    "book_to_price": "账面市值比",
    "earnings_yield": "盈利收益率",
    "growth": "成长",
    "leverage": "杠杆",
    "liquidity": "流动性",
    "momentum": "动量",
    "non_linear_size": "中盘",
    "residual_volatility": "残差波动",
    "size": "市值",
}

# ============================================================================
# 1. 加载回测持仓
# ============================================================================
print("=" * 60)
print("加载回测结果...")
print("=" * 60)

results = pd.read_pickle(os.path.join(BACKTEST_OUTPUT, "backtest_results.pkl"))
analyser = results['sys_analyser']
stock_positions = analyser['stock_positions']

# 只看多头持仓
positions_df = stock_positions[stock_positions['quantity'] > 0].copy()
print(f"持仓记录数: {len(positions_df)}")

all_codes = positions_df['order_book_id'].unique().tolist()
print(f"涉及股票数: {len(all_codes)}")

# 确定日期索引
positions_df = positions_df.reset_index()
date_col = [c for c in positions_df.columns if 'date' in c.lower()]
if date_col:
    date_col = date_col[0]
else:
    date_col = positions_df.columns[0]
positions_df[date_col] = pd.to_datetime(positions_df[date_col])
print(f"日期列: {date_col}")
print(f"日期范围: {positions_df[date_col].min()} ~ {positions_df[date_col].max()}")

# ============================================================================
# 2. 获取Barra暴露数据
# ============================================================================
print("\n" + "=" * 60)
print("获取Barra风格暴露数据...")
print("=" * 60)

barra_exposure = get_factor_exposure(
    order_book_ids=all_codes,
    start_date=START_DATE,
    end_date=END_DATE,
    factors=FACTORS,
    model="v1"
)
print(f"Barra暴露数据shape: {barra_exposure.shape}")
print(f"列名: {barra_exposure.columns.tolist()}")

if barra_exposure.empty or len(barra_exposure.columns) == 0:
    print("ERROR: Barra数据为空，尝试不指定factors参数...")
    barra_exposure = get_factor_exposure(
        order_book_ids=all_codes,
        start_date=START_DATE,
        end_date=END_DATE,
        factors=None,
        model="v1"
    )
    print(f"重新获取shape: {barra_exposure.shape}")
    print(f"列名: {barra_exposure.columns.tolist()[:20]}")
    # 筛选出需要的因子列
    available_factors = [f for f in FACTORS if f in barra_exposure.columns]
    print(f"可用因子: {available_factors}")
    FACTORS = available_factors

# ============================================================================
# 3. 计算每日组合加权暴露
# ============================================================================
print("\n" + "=" * 60)
print("计算每日组合加权暴露...")
print("=" * 60)

daily_portfolio_exposure = []
grouped = positions_df.groupby(date_col)
total_days = len(grouped)

for day_idx, (date, day_pos) in enumerate(grouped):
    if day_idx % 100 == 0:
        print(f"  进度: {day_idx}/{total_days}")

    day_pos_valid = day_pos[day_pos['market_value'] > 0]
    if day_pos_valid.empty:
        continue
    total_mv = day_pos_valid['market_value'].sum()
    weights = day_pos_valid.set_index('order_book_id')['market_value'] / total_mv

    try:
        day_exp = barra_exposure.loc[date]
        if isinstance(day_exp, pd.Series):
            day_exp = day_exp.to_frame().T
        # 只保留当日持仓的股票
        common_codes = weights.index.intersection(day_exp.index)
        if len(common_codes) == 0:
            continue
        day_exp = day_exp.loc[common_codes]
        w = weights.loc[common_codes]

        weighted_exp = {}
        for factor in FACTORS:
            if factor in day_exp.columns:
                weighted_exp[factor] = (day_exp[factor] * w).sum()
        weighted_exp['date'] = date
        daily_portfolio_exposure.append(weighted_exp)
    except (KeyError, TypeError) as e:
        continue

portfolio_exp_df = pd.DataFrame(daily_portfolio_exposure)
if not portfolio_exp_df.empty:
    portfolio_exp_df = portfolio_exp_df.set_index('date').sort_index()
print(f"组合暴露数据: {portfolio_exp_df.shape}")
print(f"日期范围: {portfolio_exp_df.index.min()} ~ {portfolio_exp_df.index.max()}")

# ============================================================================
# 4. 获取指数成分股Barra暴露
# ============================================================================
print("\n" + "=" * 60)
print("获取指数成分股Barra暴露...")
print("=" * 60)

INDICES = {
    '中证500': '000905.XSHG',
    '中证1000': '000852.XSHG',
    '中证2000': '932000.INDX',
}

# 按月采样（使用交易日）
trading_dates_all = rqdatac.get_trading_dates(start_date=START_DATE, end_date=END_DATE)
trading_dates_all = pd.to_datetime(trading_dates_all)
# 每月取第一个交易日
sample_dates = trading_dates_all.to_series().groupby(trading_dates_all.to_period('M')).first().values
sample_dates = pd.DatetimeIndex(sample_dates)
print(f"采样日期数: {len(sample_dates)}")

index_exposures = {}

for idx_name, idx_code in INDICES.items():
    print(f"\n  处理 {idx_name} ({idx_code})...")
    monthly_exps = []

    for si, sample_date in enumerate(sample_dates):
        sd_str = sample_date.strftime('%Y-%m-%d')
        try:
            # 获取指数成分股权重（返回Series: index=order_book_id, values=weight）
            weights_series = rqdatac.index_weights(idx_code, date=sd_str)
            if weights_series is None or weights_series.empty:
                continue
            components = weights_series.index.tolist()
            stock_weights = weights_series / weights_series.sum()  # 归一化
        except Exception as e:
            print(f"    {sd_str} 获取权重失败: {e}")
            continue

        try:
            exp = get_factor_exposure(
                order_book_ids=components,
                start_date=sd_str,
                end_date=sd_str,
                factors=FACTORS,
                model="v1"
            )
            if exp.empty:
                if si < 3:
                    print(f"    {sd_str} 暴露数据为空")
                continue

            # exp可能是MultiIndex (date, code)，需要按日期切片
            if isinstance(exp.index, pd.MultiIndex):
                # 取第一个日期层级
                try:
                    exp_day = exp.xs(exp.index.get_level_values(0)[0], level=0)
                except:
                    exp_day = exp.droplevel(0)
            else:
                exp_day = exp

            # 按指数权重加权平均（与组合计算方式一致）
            common_codes = stock_weights.index.intersection(exp_day.index)
            if len(common_codes) == 0:
                if si < 3:
                    print(f"    {sd_str} 无交集 (权重{len(stock_weights)}只, 暴露{len(exp_day)}只)")
                    print(f"      权重index示例: {stock_weights.index[:3].tolist()}")
                    print(f"      暴露index示例: {exp_day.index[:3].tolist()}")
                continue
            w = stock_weights.loc[common_codes]
            exp_subset = exp_day.loc[common_codes]
            weighted_exp = {}
            for factor in FACTORS:
                if factor in exp_subset.columns:
                    weighted_exp[factor] = (exp_subset[factor] * w).sum()
            weighted_exp['date'] = sample_date
            monthly_exps.append(pd.Series(weighted_exp))
        except Exception as e:
            if si < 3:
                print(f"    {sd_str} 异常: {e}")
            continue

        if (si + 1) % 10 == 0:
            print(f"    进度: {si+1}/{len(sample_dates)}")

    if monthly_exps:
        idx_exp_df = pd.DataFrame(monthly_exps).set_index('date').sort_index()
        # 插值到日频（对齐组合暴露的日期）
        idx_exp_df = idx_exp_df.reindex(portfolio_exp_df.index).interpolate(method='linear')
        index_exposures[idx_name] = idx_exp_df
        print(f"    {idx_name}: {len(idx_exp_df)} 天, 有效因子: {idx_exp_df.columns.tolist()}")

# ============================================================================
# 5. 绘制风格暴露对比图
# ============================================================================
print("\n" + "=" * 60)
print("绘制风格暴露对比图...")
print("=" * 60)

colors = {'增持组合': 'steelblue', '中证500': 'orange', '中证1000': 'green', '中证2000': 'red'}
n_factors = len(FACTORS)
n_rows = (n_factors + 1) // 2

fig, axes = plt.subplots(n_rows, 2, figsize=(18, 5 * n_rows))
axes = axes.flatten()

for i, factor in enumerate(FACTORS):
    ax = axes[i]
    cn_name = FACTOR_CN.get(factor, factor)

    if factor in portfolio_exp_df.columns:
        ax.plot(portfolio_exp_df.index, portfolio_exp_df[factor],
                color=colors['增持组合'], linewidth=1.2, alpha=0.9, label='增持组合')

    for idx_name, idx_df in index_exposures.items():
        if factor in idx_df.columns:
            ax.plot(idx_df.index, idx_df[factor],
                    color=colors[idx_name], linewidth=0.8, alpha=0.6, label=idx_name)

    ax.set_title(f'{cn_name} ({factor})', fontsize=12)
    ax.set_ylabel('暴露度')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# 隐藏多余子图
for j in range(n_factors, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Barra风格暴露对比: 增持组合 vs 中证500/1000/2000', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '08_barra_exposure_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("图8已保存: 08_barra_exposure_comparison.png")

# ============================================================================
# 6. 绘制相对暴露图
# ============================================================================
print("绘制相对暴露图...")

fig, axes = plt.subplots(n_rows, 2, figsize=(18, 5 * n_rows))
axes = axes.flatten()

for i, factor in enumerate(FACTORS):
    ax = axes[i]
    cn_name = FACTOR_CN.get(factor, factor)

    if factor not in portfolio_exp_df.columns:
        ax.set_visible(False)
        continue

    port_vals = portfolio_exp_df[factor]
    for idx_name, idx_df in index_exposures.items():
        if factor in idx_df.columns:
            relative = port_vals - idx_df[factor]
            ax.plot(relative.index, relative.values,
                    color=colors[idx_name], linewidth=0.8, alpha=0.7,
                    label=f'vs {idx_name}')

    ax.set_title(f'{cn_name} 相对暴露', fontsize=12)
    ax.set_ylabel('相对暴露度')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

for j in range(n_factors, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('相对风格暴露: 增持组合 - 指数', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '09_barra_relative_exposure.png'), dpi=150, bbox_inches='tight')
plt.close()
print("图9已保存: 09_barra_relative_exposure.png")

# ============================================================================
# 7. 汇总统计
# ============================================================================
print("\n" + "=" * 60)
print("风格暴露汇总（均值）")
print("=" * 60)

summary_data = {}
if not portfolio_exp_df.empty:
    summary_data['增持组合'] = portfolio_exp_df[FACTORS].mean()
for idx_name, idx_df in index_exposures.items():
    summary_data[idx_name] = idx_df[FACTORS].mean()

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    summary_df.index = [FACTOR_CN.get(f, f) for f in summary_df.index]
    print(summary_df.round(4).to_string())
    summary_df.to_csv(os.path.join(OUTPUT_PATH, 'barra_exposure_summary.csv'))
    print(f"\n汇总表已保存: barra_exposure_summary.csv")

print("\n全部完成！")
