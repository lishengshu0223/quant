"""
高管增持 - IM对冲回测多参数对比
配置1: 60日持仓, 10万/笔, FIFO
配置2: 22日持仓, 20万/笔, FIFO
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
INIT_CASH = 10_000_000
FUTURE_CASH = 10_000_000

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(BACKTEST_OUTPUT, exist_ok=True)

# 回测参数组合
CONFIGS = [
    {"name": "60日_10万", "holding_days": 60, "buy_value": 100_000},
    {"name": "22日_20万", "holding_days": 22, "buy_value": 200_000},
]

# ============================================================================
# 回测框架
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
    "extra": {"log_level": "warning", "enable_profiler": False, "log_file": None},
    "mod": {
        "sys_accounts": {
            "stock_t1": True, "dividend_reinvestment": False,
            "cash_return_by_stock_delisted": True, "auto_switch_order_value": False,
            "validate_stock_position": True, "validate_future_position": True,
            "financing_rate": 0.00, "financing_stocks_restriction_enabled": False,
            "futures_settlement_price_type": "close",
        },
        "sys_risk": {"validate_price": True, "validate_is_trading": True, "validate_cash": True, "validate_self_trade": False},
        "sys_simulation": {
            "signal": True, "matching_type": None, "price_limit": True,
            "liquidity_limit": False, "volume_limit": False, "volume_percent": 0.25,
            "slippage_model": "PriceRatioSlippage", "slippage": 1e-3,
            "inactive_limit": True, "management_fee": [],
        },
        "sys_transaction_cost": {
            "cn_stock_min_commission": 0, "stock_commission_multiplier": 0,
            "futures_commissions_multiplier": 0, "tax_multiplier": 1.0, "pit_tax": True,
        },
        "sys_analyser": {"benchmark": None, "record": True, "strategy_name": "", "output_file": "", "report_save_path": ""},
        "sys_progress": {"show": True},
    },
}


def make_strategy(holding_days, buy_value):
    """生成策略函数"""

    def on_order_unsolicited_update(context, event):
        if event.order.side == SIDE.SELL:
            context.unfinished_sell_orders.put([event.order.order_book_id, event.order.quantity])

    def init(context):
        context.unfinished_sell_orders = Queue(maxsize=0)
        subscribe_event(EVENT.ORDER_UNSOLICITED_UPDATE, on_order_unsolicited_update)

    def handle_bar(context, bar_dict):
        if context.now.time() != datetime.time(9, 31):
            return

        present_date = pd.to_datetime(context.now.date())
        date_str = present_date.strftime('%Y%m%d')

        # 1. 未成交卖单
        if not context.unfinished_sell_orders.empty():
            orders = []
            while not context.unfinished_sell_orders.empty():
                orders.append(context.unfinished_sell_orders.get())
            for stock_name, quantity in orders:
                pos = get_position(stock_name)
                if pos and pos.quantity > 0:
                    order_shares(stock_name, -int(min(pos.quantity, quantity)))

        # 2. FIFO到期卖出
        for position in get_positions():
            if not isinstance(position, StockPosition):
                continue
            pq = position.get_state()["position_queue"]
            for buy_date, quantity in pq:
                holding_dates = rqdatac.get_trading_dates(buy_date, present_date)
                if len(holding_dates) >= holding_days:
                    try:
                        order_shares(position.order_book_id, -quantity)
                    except:
                        pass

        # 3. 买入
        signal_file = os.path.join(SIGNAL_PATH, f"{date_str}.parquet")
        if os.path.exists(signal_file):
            df_buy = pd.read_parquet(signal_file)
            if not df_buy.empty:
                for code in df_buy["order_book_id"]:
                    cash = context.stock_account.cash
                    if cash < buy_value:
                        # FIFO卖最早持仓
                        stock_positions = [p for p in get_positions() if isinstance(p, StockPosition)]
                        if not stock_positions:
                            break
                        earliest_pos = None
                        earliest_date = None
                        for pos in stock_positions:
                            pq = pos.get_state()["position_queue"]
                            if pq:
                                if earliest_date is None or pq[0][0] < earliest_date:
                                    earliest_date = pq[0][0]
                                    earliest_pos = pos
                        if earliest_pos:
                            pq = earliest_pos.get_state()["position_queue"]
                            if pq:
                                try:
                                    order_shares(earliest_pos.order_book_id, -pq[0][1])
                                except:
                                    pass
                        cash = context.stock_account.cash
                        if cash < buy_value:
                            break
                    order_value(code, buy_value)

        # 4. IM对冲
        dominant_contract = futures.get_dominant('IM')
        if isinstance(dominant_contract, pd.Series):
            dominant_contract = dominant_contract.iloc[0]

        for position in get_positions():
            if isinstance(position, FuturePosition):
                if 'IM' in position.order_book_id and position.direction == POSITION_DIRECTION.SHORT:
                    if position.order_book_id != dominant_contract and position.quantity > 0:
                        buy_close(position.order_book_id, int(position.quantity))

        stock_long_mv = 0
        future_short_mv = 0
        for position in get_positions():
            if isinstance(position, StockPosition):
                stock_long_mv += position.market_value
            elif isinstance(position, FuturePosition):
                if position.direction == POSITION_DIRECTION.SHORT and 'IM' in position.order_book_id:
                    future_short_mv += abs(position.market_value)

        net_exposure = stock_long_mv - future_short_mv
        today = context.now.date()
        try:
            dp = futures.get_dominant_price('IM', start_date=today, end_date=today, frequency='1m', fields='open')
            dominant_price = dp.loc['IM', 'open'].iloc[0]
        except:
            return

        contract_notional = dominant_price * 200
        target_short = math.floor(net_exposure / contract_notional)
        if target_short > 0:
            sell_open(dominant_contract, target_short)
        elif target_short < 0:
            buy_close(dominant_contract, -target_short)

    def after_trading(context):
        pass

    return init, handle_bar, after_trading


# ============================================================================
# 运行多组回测
# ============================================================================
all_results = {}

for cfg in CONFIGS:
    name = cfg["name"]
    print(f"\n{'='*60}")
    print(f"运行: {name} (持仓{cfg['holding_days']}日, 单笔{cfg['buy_value']/10000:.0f}万)")
    print(f"{'='*60}")

    my_config = copy.deepcopy(__sample_config__)
    my_config["base"]["start_date"] = START_DATE
    my_config["base"]["end_date"] = END_DATE
    my_config["base"]["accounts"]["STOCK"] = INIT_CASH
    my_config["base"]["accounts"]["FUTURE"] = FUTURE_CASH
    my_config["mod"]["sys_analyser"]["strategy_name"] = f"holding_increase_{name}"
    my_config["mod"]["sys_analyser"]["output_file"] = os.path.join(BACKTEST_OUTPUT, f"backtest_{name}.pkl")
    my_config["mod"]["sys_analyser"]["report_save_path"] = BACKTEST_OUTPUT

    init_fn, handle_bar_fn, after_trading_fn = make_strategy(cfg["holding_days"], cfg["buy_value"])
    results = run_func(init=init_fn, handle_bar=handle_bar_fn, after_trading=after_trading_fn, config=my_config)
    pd.to_pickle(results, os.path.join(BACKTEST_OUTPUT, f"backtest_{name}.pkl"))

    # 计算绩效
    analyser = results['sys_analyser']
    stock_account = analyser['stock_account']
    future_account = analyser['future_account']

    stock_account["daily_pnl"] = stock_account["total_value"].diff().fillna(0)
    future_account["daily_pnl"] = future_account["total_value"].diff().fillna(0)

    # 对冲后净值（以股票本金为基准）
    net_value = ((stock_account["daily_pnl"] + future_account["daily_pnl"]).cumsum() / INIT_CASH) + 1
    daily_ret = net_value.pct_change().fillna(0)

    metrics = {
        "name": name,
        "holding_days": cfg["holding_days"],
        "buy_value": cfg["buy_value"],
        "total_ret": (1 + daily_ret).prod() - 1,
        "annual_ret": ep.annual_return(daily_ret, annualization=240),
        "max_dd": ep.max_drawdown(daily_ret),
        "sharpe": ep.sharpe_ratio(daily_ret, annualization=240),
        "calmar": ep.calmar_ratio(daily_ret, annualization=240),
        "vol": ep.annual_volatility(daily_ret, annualization=240),
        "win_rate": (daily_ret > 0).sum() / len(daily_ret),
        "cash_usage": (stock_account["market_value"] / stock_account["total_value"]).mean(),
        "net_value": net_value,
        "daily_ret": daily_ret,
    }
    all_results[name] = metrics
    print(f"  年化: {metrics['annual_ret']:.2%}, 夏普: {metrics['sharpe']:.3f}, 最大回撤: {metrics['max_dd']:.2%}, 资金利用率: {metrics['cash_usage']:.2%}")

# 加入之前的22日_10万结果
prev_pkl = os.path.join(BACKTEST_OUTPUT, "backtest_hedged.pkl")
if os.path.exists(prev_pkl):
    prev_results = pd.read_pickle(prev_pkl)
    pa = prev_results['sys_analyser']
    ps = pa['stock_account']
    pf = pa['future_account']
    ps["daily_pnl"] = ps["total_value"].diff().fillna(0)
    pf["daily_pnl"] = pf["total_value"].diff().fillna(0)
    nv = ((ps["daily_pnl"] + pf["daily_pnl"]).cumsum() / INIT_CASH) + 1
    dr = nv.pct_change().fillna(0)
    all_results["22日_10万"] = {
        "name": "22日_10万",
        "holding_days": 22,
        "buy_value": 100_000,
        "total_ret": (1 + dr).prod() - 1,
        "annual_ret": ep.annual_return(dr, annualization=240),
        "max_dd": ep.max_drawdown(dr),
        "sharpe": ep.sharpe_ratio(dr, annualization=240),
        "calmar": ep.calmar_ratio(dr, annualization=240),
        "vol": ep.annual_volatility(dr, annualization=240),
        "win_rate": (dr > 0).sum() / len(dr),
        "cash_usage": (ps["market_value"] / ps["total_value"]).mean(),
        "net_value": nv,
        "daily_ret": dr,
    }

# ============================================================================
# 汇总对比
# ============================================================================
print(f"\n{'='*60}")
print("对冲后绩效对比（以股票本金为基准）")
print(f"{'='*60}")
print(f"{'配置':<12} {'年化收益':<10} {'最大回撤':<10} {'夏普':<8} {'Calmar':<8} {'波动':<8} {'胜率':<8} {'利用率':<8}")
print("-" * 74)
for name, m in all_results.items():
    print(f"{name:<12} {m['annual_ret']:<10.2%} {m['max_dd']:<10.2%} {m['sharpe']:<8.3f} {m['calmar']:<8.3f} {m['vol']:<8.2%} {m['win_rate']:<8.2%} {m['cash_usage']:<8.2%}")

# ============================================================================
# 绘图
# ============================================================================
fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True, height_ratios=[2, 1])

colors_map = {"22日_10万": "steelblue", "60日_10万": "coral", "22日_20万": "green"}

for name, m in all_results.items():
    c = colors_map.get(name, "gray")
    axes[0].plot(m["net_value"].index, m["net_value"].values, color=c, linewidth=1.2,
                 label=f"{name} (年化{m['annual_ret']:.1%}, 夏普{m['sharpe']:.2f})")

axes[0].axhline(y=1, color='black', linestyle='-', linewidth=0.5)
axes[0].set_title('高管增持 IM对冲策略 - 多参数对比 (以股票本金为基准)')
axes[0].set_ylabel('净值')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

for name, m in all_results.items():
    c = colors_map.get(name, "gray")
    nv = m["net_value"]
    dd = nv / nv.cummax() - 1
    axes[1].plot(dd.index, dd.values, color=c, linewidth=0.8, alpha=0.7, label=f"{name} ({m['max_dd']:.1%})")

axes[1].set_title('回撤对比')
axes[1].set_ylabel('回撤')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '12_multi_config_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n对比图已保存: 12_multi_config_comparison.png")
print("全部完成！")
