"""
高管增持事件 - 多头+IM对冲回测
基于已生成的买入信号，加入中证1000股指期货(IM)对冲
"""
import datetime
import os
import copy
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import empyrical as ep

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

import rqdatac
rqdatac.init()

from rqalpha import run_func
from rqalpha_plus.apis import *
from rqalpha.mod.rqalpha_mod_sys_accounts.position_model import FuturePosition, StockPosition
from queue import Queue

# ============================================================================
# 配置
# ============================================================================
SIGNAL_PATH = r"F:\quant\research\holding_increase\backtest_signals"
OUTPUT_PATH = r"F:\quant\research\holding_increase\analysis_output"
BACKTEST_OUTPUT = r"F:\quant\research\holding_increase\backtest_results"

START_DATE = "2023-01-01"
END_DATE = "2026-07-29"
INIT_CASH = 10_000_000  # 股票账户1000万
FUTURE_CASH = 10_000_000  # 期货账户1000万
BUY_VALUE = 100_000
HOLDING_DAYS = 22

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(BACKTEST_OUTPUT, exist_ok=True)

# ============================================================================
# 回测配置
# ============================================================================
__sample_config__ = {
    "base": {
        "data_bundle_path": r"F:\Trade_data\rq_backtest_data",
        "start_date": "",
        "end_date": "",
        "margin_multiplier": 1,
        "frequency": "1m",
        "accounts": {"STOCK": 0.0, "FUTURE": 0.0},
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
            "strategy_name": "holding_increase_hedged",
            "output_file": "",
            "report_save_path": "",
        },
        "sys_progress": {"show": True},
    },
}


# ============================================================================
# 策略函数
# ============================================================================
def on_order_unsolicited_update(context, event):
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
        if not isinstance(position, StockPosition):
            continue
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
    if os.path.exists(signal_file):
        df_buy = pd.read_parquet(signal_file)
        if not df_buy.empty:
            for code in df_buy["order_book_id"]:
                cash = context.stock_account.cash
                if cash < BUY_VALUE:
                    # FIFO卖出最早持仓
                    stock_positions = [p for p in get_positions() if isinstance(p, StockPosition)]
                    if not stock_positions:
                        break
                    earliest_pos = None
                    earliest_date = None
                    for pos in stock_positions:
                        pq = pos.get_state()["position_queue"]
                        if pq:
                            first_buy_date = pq[0][0]
                            if earliest_date is None or first_buy_date < earliest_date:
                                earliest_date = first_buy_date
                                earliest_pos = pos
                    if earliest_pos:
                        pq = earliest_pos.get_state()["position_queue"]
                        if pq:
                            _, qty = pq[0]
                            try:
                                order_shares(earliest_pos.order_book_id, -qty)
                            except:
                                pass
                    cash = context.stock_account.cash
                    if cash < BUY_VALUE:
                        break
                order_value(code, BUY_VALUE)

    # ===== 4. IM期货对冲逻辑 =====
    # 获取IM主力合约
    dominant_contract = futures.get_dominant('IM')
    if isinstance(dominant_contract, pd.Series):
        dominant_contract = dominant_contract.iloc[0]

    # 平掉非主力IM空头（移仓换月）
    for position in get_positions():
        if isinstance(position, FuturePosition):
            if 'IM' in position.order_book_id and position.direction == POSITION_DIRECTION.SHORT:
                if position.order_book_id != dominant_contract and position.quantity > 0:
                    buy_close(position.order_book_id, int(position.quantity))

    # 计算股票多头市值 和 IM空头市值
    stock_long_mv = 0
    future_short_mv = 0
    for position in get_positions():
        if isinstance(position, StockPosition):
            stock_long_mv += position.market_value
        elif isinstance(position, FuturePosition):
            if position.direction == POSITION_DIRECTION.SHORT and 'IM' in position.order_book_id:
                future_short_mv += abs(position.market_value)

    # 净敞口
    net_exposure = stock_long_mv - future_short_mv

    # 获取主力合约开盘价，计算一手IM市值（合约乘数200）
    today = context.now.date()
    try:
        dominant_price_data = futures.get_dominant_price(
            'IM', start_date=today, end_date=today, frequency='1m', fields='open'
        )
        dominant_price = dominant_price_data.loc['IM', 'open'].iloc[0]
    except:
        return

    contract_notional = dominant_price * 200

    # 目标空头手数
    target_short_contracts = math.floor(net_exposure / contract_notional)

    if target_short_contracts > 0:
        sell_open(dominant_contract, target_short_contracts)
    elif target_short_contracts < 0:
        buy_close(dominant_contract, -target_short_contracts)


def after_trading(context):
    pass


# ============================================================================
# 运行回测
# ============================================================================
print("=" * 60)
print("高管增持 - IM对冲回测")
print("=" * 60)
print(f"回测区间: {START_DATE} ~ {END_DATE}")
print(f"股票本金: {INIT_CASH:,.0f}, 期货本金: {FUTURE_CASH:,.0f}")
print(f"单笔: {BUY_VALUE:,.0f}, 持仓: {HOLDING_DAYS}日")

my_config = copy.deepcopy(__sample_config__)
my_config["base"]["start_date"] = START_DATE
my_config["base"]["end_date"] = END_DATE
my_config["base"]["accounts"]["STOCK"] = INIT_CASH
my_config["base"]["accounts"]["FUTURE"] = FUTURE_CASH
my_config["mod"]["sys_analyser"]["output_file"] = os.path.join(BACKTEST_OUTPUT, "backtest_hedged.pkl")
my_config["mod"]["sys_analyser"]["report_save_path"] = BACKTEST_OUTPUT

results = run_func(init=init, handle_bar=handle_bar, after_trading=after_trading, config=my_config)
pd.to_pickle(results, os.path.join(BACKTEST_OUTPUT, "backtest_hedged.pkl"))
print("\n回测完成！")

# ============================================================================
# 绩效统计
# ============================================================================
print("\n" + "=" * 60)
print("绩效统计")
print("=" * 60)

analyser = results['sys_analyser']
stock_account = analyser['stock_account']
future_account = analyser['future_account']

# 多空轧差日收益 = (股票端日PnL + 期货端日PnL) / 股票端期初本金
stock_account["daily_pnl"] = stock_account["total_value"].diff().fillna(0)
future_account["daily_pnl"] = future_account["total_value"].diff().fillna(0)

# 对冲后净值（以股票本金为基准）
net_value = ((stock_account["daily_pnl"] + future_account["daily_pnl"]).cumsum() / INIT_CASH) + 1
daily_ret = net_value.pct_change().fillna(0)

# 纯多头净值（对比用）
stock_nv = stock_account["total_value"] / INIT_CASH
stock_ret = stock_nv.pct_change().fillna(0)

# 对冲后指标
total_ret = (1 + daily_ret).prod() - 1
annual_ret = ep.annual_return(daily_ret, annualization=240)
max_dd = ep.max_drawdown(daily_ret)
sharpe = ep.sharpe_ratio(daily_ret, annualization=240)
calmar = ep.calmar_ratio(daily_ret, annualization=240)
vol = ep.annual_volatility(daily_ret, annualization=240)
win_rate = (daily_ret > 0).sum() / len(daily_ret)

# 纯多头指标
stock_total_ret = (1 + stock_ret).prod() - 1
stock_annual_ret = ep.annual_return(stock_ret, annualization=240)
stock_max_dd = ep.max_drawdown(stock_ret)
stock_sharpe = ep.sharpe_ratio(stock_ret, annualization=240)

print(f"\n{'指标':<16} {'对冲后':<14} {'纯多头':<14}")
print("-" * 44)
print(f"{'总收益':<16} {total_ret:<14.2%} {stock_total_ret:<14.2%}")
print(f"{'年化收益':<14} {annual_ret:<14.2%} {stock_annual_ret:<14.2%}")
print(f"{'最大回撤':<14} {max_dd:<14.2%} {stock_max_dd:<14.2%}")
print(f"{'夏普比率':<14} {sharpe:<14.3f} {stock_sharpe:<14.3f}")
print(f"{'Calmar':<16} {calmar:<14.3f}")
print(f"{'年化波动':<14} {vol:<14.2%}")
print(f"{'日胜率':<16} {win_rate:<14.2%}")

# 资金利用率
cash_usage = stock_account["market_value"] / stock_account["total_value"]
print(f"{'平均资金利用率':<12} {cash_usage.mean():<14.2%}")

# ============================================================================
# 绘图
# ============================================================================
fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True, height_ratios=[2, 1, 1])

# 净值对比
axes[0].plot(net_value.index, net_value.values, color='steelblue', linewidth=1.2, label=f'对冲后 (年化{annual_ret:.1%}, 夏普{sharpe:.2f})')
axes[0].plot(stock_nv.index, stock_nv.values, color='coral', linewidth=0.8, alpha=0.7, label=f'纯多头 (年化{stock_annual_ret:.1%}, 夏普{stock_sharpe:.2f})')
axes[0].axhline(y=1, color='black', linestyle='-', linewidth=0.5)
axes[0].set_title(f'高管增持策略 - IM对冲 vs 纯多头 (22日持仓, 单笔10万)')
axes[0].set_ylabel('净值')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# 对冲后回撤
dd_hedged = net_value / net_value.cummax() - 1
dd_long = stock_nv / stock_nv.cummax() - 1
axes[1].fill_between(dd_hedged.index, dd_hedged.values, 0, color='steelblue', alpha=0.3, label=f'对冲后 (最大{max_dd:.1%})')
axes[1].fill_between(dd_long.index, dd_long.values, 0, color='coral', alpha=0.2, label=f'纯多头 (最大{stock_max_dd:.1%})')
axes[1].set_title('回撤对比')
axes[1].set_ylabel('回撤')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

# 资金利用率
axes[2].plot(cash_usage.index, cash_usage.values, color='green', linewidth=0.8)
axes[2].set_title('股票账户资金利用率')
axes[2].set_ylabel('利用率')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '10_hedged_performance.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n图表已保存: 10_hedged_performance.png")

# 月度收益热力图
monthly_ret = daily_ret.resample("M").apply(lambda x: (1 + x).prod() - 1)
monthly_df = pd.DataFrame({'ret': monthly_ret})
monthly_df['year'] = monthly_df.index.year
monthly_df['month'] = monthly_df.index.month
pivot = monthly_df.pivot(index='year', columns='month', values='ret')

fig, ax = plt.subplots(figsize=(14, 5))
import seaborn as sns
sns.heatmap(pivot, fmt=".2%", cmap="coolwarm", annot=True, center=0,
            xticklabels=[f'{m}月' for m in pivot.columns],
            yticklabels=pivot.index, ax=ax)
ax.set_title(f'对冲后月度收益 (年化{annual_ret:.1%}, 夏普{sharpe:.2f}, 最大回撤{max_dd:.1%})')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '11_hedged_monthly.png'), dpi=150, bbox_inches='tight')
plt.close()
print("月度热力图已保存: 11_hedged_monthly.png")

print("\n全部完成！")
