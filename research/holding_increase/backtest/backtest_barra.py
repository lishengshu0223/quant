"""
高管增持事件 - 多头FIFO回测 + Barra风格暴露分析
- 本金1000万，单笔10万
- 22日持仓，FIFO卖出
- 现金耗尽时卖出最早持仓
- 不做对冲，纯多头
- 回测后计算组合Barra风格暴露，对比中证500/1000/2000
"""
import datetime
import os
import sys
import copy
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

import rqdatac
rqdatac.init()

from rqalpha import run_func
from rqalpha_plus.apis import *
from queue import Queue

# ============================================================================
# 配置
# ============================================================================
EVENT_PATH = r"F:\quant\research\holding_increase\increase_events"
SIGNAL_PATH = r"F:\quant\research\holding_increase\backtest_signals"
OUTPUT_PATH = r"F:\quant\research\holding_increase\analysis_output"
BACKTEST_OUTPUT = r"F:\quant\research\holding_increase\backtest_results"

START_DATE = "2023-01-01"
END_DATE = "2026-07-29"
INIT_CASH = 10_000_000  # 1000万
BUY_VALUE = 100_000     # 单笔10万
HOLDING_DAYS = 22       # 持仓22个交易日

os.makedirs(SIGNAL_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(BACKTEST_OUTPUT, exist_ok=True)

# ============================================================================
# Part 1: 生成每日买入信号
# ============================================================================
print("=" * 60)
print("Part 1: 生成每日买入信号")
print("=" * 60)

# 加载事件
files = [f for f in os.listdir(EVENT_PATH) if f.endswith('.json')]
records = []
for f in files:
    with open(os.path.join(EVENT_PATH, f), 'r', encoding='utf-8') as fp:
        records.append(json.load(fp))

df = pd.DataFrame(records)
df['公告日期'] = pd.to_datetime(df['公告日期'])
print(f"总事件数: {len(df)}")

# 筛选
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

# 获取交易日历
trading_dates = rqdatac.get_trading_dates(start_date=START_DATE, end_date=END_DATE)
trading_dates = pd.to_datetime(trading_dates)
date_to_idx = {d: i for i, d in enumerate(trading_dates)}

# 生成买入信号：公告日的下一交易日买入
buy_signals = {}  # {date_str: [order_book_id, ...]}
for _, row in df.iterrows():
    ann_date = row['公告日期']
    code = row['股票代码']
    if ann_date not in date_to_idx:
        mask = trading_dates >= ann_date
        if not mask.any():
            continue
        ann_idx = mask.values.argmax()
    else:
        ann_idx = date_to_idx[ann_date]
    buy_idx = ann_idx + 1
    if buy_idx >= len(trading_dates):
        continue
    buy_date = trading_dates[buy_idx]
    buy_date_str = buy_date.strftime('%Y%m%d')
    if buy_date_str not in buy_signals:
        buy_signals[buy_date_str] = []
    buy_signals[buy_date_str].append(code)

# 保存为parquet
for date_str, codes in buy_signals.items():
    signal_df = pd.DataFrame({'order_book_id': codes})
    signal_df.to_parquet(os.path.join(SIGNAL_PATH, f"{date_str}.parquet"), index=False)

print(f"生成 {len(buy_signals)} 个交易日的买入信号")
print(f"信号路径: {SIGNAL_PATH}")

# ============================================================================
# Part 2: 回测
# ============================================================================
print("\n" + "=" * 60)
print("Part 2: 运行回测")
print("=" * 60)

__sample_config__ = {
    "base": {
        "data_bundle_path": r"F:\Trade_data\rq_backtest_data",
        "start_date": "",
        "end_date": "",
        "margin_multiplier": 1,
        "frequency": "1m",
        "accounts": {"STOCK": 0.0},
        "auto_update_bundle": True,
    },
    "extra": {
        "log_level": "warning",
        "enable_profiler": False,
        "log_file": None,
    },
    "mod": {
        "sys_accounts": {
            "stock_t1": True,
            "dividend_reinvestment": False,
            "cash_return_by_stock_delisted": True,
            "auto_switch_order_value": False,
            "validate_stock_position": True,
            "validate_future_position": True,
            "financing_rate": 0.00,
            "financing_stocks_restriction_enabled": False,
            "futures_settlement_price_type": "close",
        },
        "sys_risk": {
            "validate_price": True,
            "validate_is_trading": True,
            "validate_cash": True,
            "validate_self_trade": False,
        },
        "sys_simulation": {
            "signal": True,
            "matching_type": None,
            "price_limit": True,
            "liquidity_limit": False,
            "volume_limit": False,
            "volume_percent": 0.25,
            "slippage_model": "PriceRatioSlippage",
            "slippage": 1e-3,
            "inactive_limit": True,
            "management_fee": [],
        },
        "sys_transaction_cost": {
            "cn_stock_min_commission": 0,
            "stock_commission_multiplier": 0,
            "futures_commission_multiplier": 0,
            "tax_multiplier": 1.0,
            "pit_tax": True,
        },
        "sys_analyser": {
            "benchmark": None,
            "record": True,
            "strategy_name": "holding_increase_long_only",
            "output_file": "",
            "report_save_path": "",
        },
        "sys_progress": {"show": True},
    },
}


def on_order_unsolicited_update(context, event):
    """跌停未成交卖单记录"""
    if event.order.side == SIDE.SELL:
        context.unfinished_sell_orders.put(
            [event.order.order_book_id, event.order.quantity]
        )


def init(context):
    context.unfinished_sell_orders = Queue(maxsize=0)
    subscribe_event(EVENT.ORDER_UNSOLICITED_UPDATE, on_order_unsolicited_update)


def handle_bar(context, bar_dict):
    if context.now.time() != datetime.time(9, 31):
        return

    present_date = pd.to_datetime(context.now.date())
    date_str = present_date.strftime('%Y%m%d')

    # 1. 处理昨日未成交卖单
    if not context.unfinished_sell_orders.empty():
        orders = []
        while not context.unfinished_sell_orders.empty():
            orders.append(context.unfinished_sell_orders.get())
        for stock_name, quantity in orders:
            pos = get_position(stock_name)
            if pos and pos.quantity > 0:
                order_shares(stock_name, -int(min(pos.quantity, quantity)))

    # 2. FIFO到期卖出（持仓>=22个交易日）
    positions = get_positions()
    for position in positions:
        position_queue = position.get_state()["position_queue"]
        for buy_date, quantity in position_queue:
            holding_dates = rqdatac.get_trading_dates(buy_date, present_date)
            if len(holding_dates) >= HOLDING_DAYS:
                try:
                    order_shares(position.order_book_id, -quantity)
                except:
                    pass

    # 3. 买入信号
    signal_file = os.path.join(SIGNAL_PATH, f"{date_str}.parquet")
    if not os.path.exists(signal_file):
        return

    df_buy = pd.read_parquet(signal_file)
    if df_buy.empty:
        return

    for code in df_buy["order_book_id"]:
        # 检查现金是否足够
        cash = context.stock_account.cash
        if cash < BUY_VALUE:
            # 现金不足，FIFO卖出最早持仓腾出资金
            positions = get_positions()
            if not positions:
                break
            # 找最早买入的持仓
            earliest_pos = None
            earliest_date = None
            for pos in positions:
                pq = pos.get_state()["position_queue"]
                if pq:
                    first_buy_date = pq[0][0]
                    if earliest_date is None or first_buy_date < earliest_date:
                        earliest_date = first_buy_date
                        earliest_pos = pos
            if earliest_pos:
                # 卖出最早持仓的第一笔
                pq = earliest_pos.get_state()["position_queue"]
                if pq:
                    _, qty = pq[0]
                    try:
                        order_shares(earliest_pos.order_book_id, -qty)
                    except:
                        pass
            # 重新检查现金
            cash = context.stock_account.cash
            if cash < BUY_VALUE:
                break

        order_value(code, BUY_VALUE)


def after_trading(context):
    pass


# 运行回测
my_config = copy.deepcopy(__sample_config__)
my_config["base"]["start_date"] = START_DATE
my_config["base"]["end_date"] = END_DATE
my_config["base"]["accounts"]["STOCK"] = INIT_CASH
my_config["mod"]["sys_analyser"]["output_file"] = os.path.join(BACKTEST_OUTPUT, "backtest_results.pkl")
my_config["mod"]["sys_analyser"]["report_save_path"] = BACKTEST_OUTPUT

print(f"回测区间: {START_DATE} ~ {END_DATE}")
print(f"本金: {INIT_CASH:,.0f}, 单笔: {BUY_VALUE:,.0f}, 持仓: {HOLDING_DAYS}日")

results = run_func(init=init, handle_bar=handle_bar, after_trading=after_trading, config=my_config)
pd.to_pickle(results, os.path.join(BACKTEST_OUTPUT, "backtest_results.pkl"))
print("回测完成，结果已保存")

# ============================================================================
# Part 3: 回测绩效统计
# ============================================================================
print("\n" + "=" * 60)
print("Part 3: 回测绩效")
print("=" * 60)

import empyrical as ep

analyser = results['sys_analyser']
stock_account = analyser['stock_account']
stock_positions = analyser['stock_positions']

total_value = stock_account['total_value']
daily_ret = total_value.pct_change().fillna(0)

total_ret = (1 + daily_ret).prod() - 1
annual_ret = ep.annual_return(daily_ret, annualization=240)
max_dd = ep.max_drawdown(daily_ret)
sharpe = ep.sharpe_ratio(daily_ret, annualization=240)
calmar = ep.calmar_ratio(daily_ret, annualization=240)
vol = ep.annual_volatility(daily_ret, annualization=240)
win_rate = (daily_ret > 0).sum() / len(daily_ret)

print(f"总收益: {total_ret:.2%}")
print(f"年化收益: {annual_ret:.2%}")
print(f"最大回撤: {max_dd:.2%}")
print(f"夏普比率: {sharpe:.3f}")
print(f"Calmar: {calmar:.3f}")
print(f"年化波动: {vol:.2%}")
print(f"日胜率: {win_rate:.2%}")

# 资金利用率
market_value = stock_account['market_value']
cash_usage = market_value / total_value
print(f"平均资金利用率: {cash_usage.mean():.2%}")

# 净值曲线图
fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True, height_ratios=[2, 1, 1])
net_value = (1 + daily_ret).cumprod()
drawdown = net_value / net_value.cummax() - 1

axes[0].plot(net_value.index, net_value.values, color='steelblue', linewidth=1)
axes[0].set_title(f'高管增持多头策略净值 (年化{annual_ret:.1%}, 夏普{sharpe:.2f}, 最大回撤{max_dd:.1%})')
axes[0].set_ylabel('净值')
axes[0].grid(True, alpha=0.3)

axes[1].fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
axes[1].set_title('回撤')
axes[1].set_ylabel('回撤')
axes[1].grid(True, alpha=0.3)

axes[2].plot(cash_usage.index, cash_usage.values, color='green', linewidth=0.8)
axes[2].set_title('资金利用率')
axes[2].set_ylabel('利用率')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '07_backtest_performance.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n净值图已保存: 07_backtest_performance.png")

# ============================================================================
# Part 4: Barra风格暴露分析
# ============================================================================
print("\n" + "=" * 60)
print("Part 4: Barra风格暴露分析")
print("=" * 60)

sys.path.insert(0, r"F:\quant")
from local_api.barra import get_factor_exposure

FACTORS = ["Liquidity", "Leverage", "BTOP", "Earnings Yield", "Growth",
           "Momentum", "Size", "Beta", "Mid cap", "Profitability"]

# 4.1 从回测结果提取每日持仓
print("提取每日持仓...")
# stock_positions 是 DataFrame，index是日期，columns包含order_book_id, quantity, market_value等
# 需要按日期分组获取持仓
positions_df = stock_positions.copy()
positions_df = positions_df[positions_df['quantity'] > 0]  # 只看多头
print(f"持仓记录数: {len(positions_df)}")

# 按日期分组，计算每日组合的加权Barra暴露
# 先获取所有持仓股票的Barra暴露
all_codes = positions_df['order_book_id'].unique().tolist()
print(f"涉及股票数: {len(all_codes)}")

# 分批获取Barra暴露（避免内存过大）
print("获取Barra风格暴露数据...")
barra_exposure = get_factor_exposure(
    order_book_ids=all_codes,
    start_date=START_DATE,
    end_date=END_DATE,
    factors=FACTORS,
    model="v1"
)
print(f"Barra暴露数据: {barra_exposure.shape}")

# 计算每日组合加权暴露
print("计算每日组合加权暴露...")
daily_portfolio_exposure = []

# 按日期分组持仓
positions_df_reset = positions_df.reset_index()
# 确定日期列名
date_col = 'date' if 'date' in positions_df_reset.columns else positions_df_reset.columns[0]
positions_df_reset[date_col] = pd.to_datetime(positions_df_reset[date_col])

for date, day_pos in positions_df_reset.groupby(date_col):
    # 计算市值权重
    day_pos_valid = day_pos[day_pos['market_value'] > 0]
    if day_pos_valid.empty:
        continue
    total_mv = day_pos_valid['market_value'].sum()
    day_pos_valid = day_pos_valid.copy()
    day_pos_valid['weight'] = day_pos_valid['market_value'] / total_mv

    # 获取当日Barra暴露
    codes_today = day_pos_valid['order_book_id'].tolist()
    weights_today = day_pos_valid.set_index('order_book_id')['weight']

    try:
        day_exposure = barra_exposure.loc[date]
        if isinstance(day_exposure, pd.Series):
            day_exposure = day_exposure.to_frame().T
        day_exposure = day_exposure[day_exposure.index.isin(codes_today)]
        if day_exposure.empty:
            continue
        # 加权平均
        weighted_exp = {}
        for factor in FACTORS:
            if factor in day_exposure.columns:
                vals = day_exposure[factor]
                w = weights_today.reindex(vals.index).fillna(0)
                weighted_exp[factor] = (vals * w).sum()
        weighted_exp['date'] = date
        daily_portfolio_exposure.append(weighted_exp)
    except (KeyError, TypeError):
        continue

portfolio_exp_df = pd.DataFrame(daily_portfolio_exposure)
if not portfolio_exp_df.empty:
    portfolio_exp_df = portfolio_exp_df.set_index('date').sort_index()
print(f"组合暴露数据: {portfolio_exp_df.shape}")

# 4.2 获取指数成分股的Barra暴露
print("\n获取指数成分股Barra暴露...")

# 中证500: 000905.XSHG, 中证1000: 000852.XSHG, 中证2000: 932000.CSI
INDICES = {
    '中证500': '000905.XSHG',
    '中证1000': '000852.XSHG',
    '中证2000': '932000.CSI',
}

# 按月采样获取指数成分（减少API调用）
sample_dates = pd.date_range(START_DATE, END_DATE, freq='MS')
# 确保包含首尾
sample_dates = sample_dates.union(pd.DatetimeIndex([pd.Timestamp(START_DATE), pd.Timestamp(END_DATE)]))
sample_dates = sample_dates.unique().sort_values()

index_exposures = {}  # {index_name: DataFrame}

for idx_name, idx_code in INDICES.items():
    print(f"  处理 {idx_name} ({idx_code})...")
    monthly_exps = []

    for sample_date in sample_dates:
        sd_str = sample_date.strftime('%Y-%m-%d')
        try:
            components = rqdatac.index_components(idx_code, date=sd_str)
            if not components:
                continue
        except:
            continue

        # 获取该日Barra暴露
        try:
            exp = get_factor_exposure(
                order_book_ids=components,
                start_date=sd_str,
                end_date=sd_str,
                factors=FACTORS,
                model="v1"
            )
            if exp.empty:
                continue
            # 等权平均（简化，不用权重）
            mean_exp = exp[FACTORS].mean()
            mean_exp['date'] = sample_date
            monthly_exps.append(mean_exp)
        except:
            continue

    if monthly_exps:
        idx_exp_df = pd.DataFrame(monthly_exps).set_index('date').sort_index()
        # 插值到日频
        idx_exp_df = idx_exp_df.reindex(portfolio_exp_df.index).interpolate(method='linear')
        index_exposures[idx_name] = idx_exp_df
        print(f"    {idx_name}: {len(idx_exp_df)} 天")

# 4.3 绘制相对暴露对比图
print("\n绘制风格暴露对比图...")

n_factors = len(FACTORS)
fig, axes = plt.subplots(5, 2, figsize=(18, 25))
axes = axes.flatten()

colors = {'组合': 'steelblue', '中证500': 'orange', '中证1000': 'green', '中证2000': 'red'}

for i, factor in enumerate(FACTORS):
    ax = axes[i]
    # 组合暴露
    if factor in portfolio_exp_df.columns:
        ax.plot(portfolio_exp_df.index, portfolio_exp_df[factor],
                color=colors['组合'], linewidth=1, alpha=0.8, label='增持组合')

    # 指数暴露
    for idx_name, idx_df in index_exposures.items():
        if factor in idx_df.columns:
            ax.plot(idx_df.index, idx_df[factor],
                    color=colors[idx_name], linewidth=0.8, alpha=0.6, label=idx_name)

    ax.set_title(f'{factor}', fontsize=12)
    ax.set_ylabel('暴露度')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Barra风格暴露对比: 增持组合 vs 中证500/1000/2000', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '08_barra_exposure_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"风格暴露对比图已保存: 08_barra_exposure_comparison.png")

# 4.4 计算相对暴露（组合 - 各指数）
print("\n绘制相对暴露图...")
fig, axes = plt.subplots(5, 2, figsize=(18, 25))
axes = axes.flatten()

for i, factor in enumerate(FACTORS):
    ax = axes[i]
    if factor not in portfolio_exp_df.columns:
        continue
    port_vals = portfolio_exp_df[factor]

    for idx_name, idx_df in index_exposures.items():
        if factor in idx_df.columns:
            relative = port_vals - idx_df[factor]
            ax.plot(relative.index, relative.values,
                    color=colors[idx_name], linewidth=0.8, alpha=0.7,
                    label=f'vs {idx_name}')

    ax.set_title(f'{factor} 相对暴露', fontsize=12)
    ax.set_ylabel('相对暴露度')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('相对风格暴露: 增持组合 - 指数', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '09_barra_relative_exposure.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"相对暴露图已保存: 09_barra_relative_exposure.png")

# 4.5 汇总统计
print("\n" + "=" * 60)
print("风格暴露汇总（均值）")
print("=" * 60)
summary_data = {'增持组合': portfolio_exp_df[FACTORS].mean()}
for idx_name, idx_df in index_exposures.items():
    summary_data[idx_name] = idx_df[FACTORS].mean()

summary_df = pd.DataFrame(summary_data)
print(summary_df.round(4).to_string())

# 保存
summary_df.to_csv(os.path.join(OUTPUT_PATH, 'barra_exposure_summary.csv'))
print(f"\n汇总表已保存: barra_exposure_summary.csv")
print("\n全部完成！")
