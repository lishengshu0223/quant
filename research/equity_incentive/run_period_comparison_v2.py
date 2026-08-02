import datetime
import os
import re
import copy
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rqalpha_plus.apis import *
from rqalpha.apis import *
import rqdatac
from queue import Queue


__sample_config__ = {
    "base": {
        "data_bundle_path": r"F:\Trade_data\rq_backtest_data",
        "start_date": "",
        "end_date": "",
        "margin_multiplier": 1,
        "frequency": "1m",
        "accounts": {
            "STOCK": 0.0,
            "FUTURE": 0.0
        },
        "auto_update_bundle": True,
    },
    "extra": {
        "log_level": "info",
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
            "strategy_name": "",
            "output_file": "",
            "report_save_path": "",
        },
        "sys_progress": {
            "show": True,
        },
    },
}


def make_strategy_functions(holding_days, buy_value):
    """根据持仓周期和单笔买入金额创建策略函数"""

    def init(context):
        context.unfinished_sell_orders = Queue(maxsize=0)
        context.holding_records = {}
        context.holding_days = holding_days
        context.buy_value = buy_value

    def handle_bar(context, bar_dict):
        path = r"F:\quant\research\equity_incentive\incentive_plan"
        from rqalpha.mod.rqalpha_mod_sys_accounts.position_model import FuturePosition, StockPosition
        if context.now.time() == datetime.time(9, 31):
            present_date = pd.to_datetime(context.now.date())
            parquet_file = os.path.join(path, f"{present_date.strftime('%Y%m%d')}.parquet")
            if os.path.exists(parquet_file):
                df_buy = pd.read_parquet(parquet_file)
            else:
                df_buy = pd.DataFrame()
            if not df_buy.empty:
                for code in df_buy["order_book_id"]:
                    order_value(code, buy_value)

            positions = get_positions()
            for position in positions:
                position_queue = position.get_state()["position_queue"]
                for buy_date, quantity in position_queue:
                    holding_date = rqdatac.get_trading_dates(buy_date, present_date)
                    if len(holding_date) >= holding_days:
                        try:
                            order_shares(position.order_book_id, -quantity)
                        except:
                            print(position)

            # ===== 期货对冲逻辑 =====
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
            dominant_price_data = futures.get_dominant_price('IM', start_date=today, end_date=today, frequency='1m', fields='open')
            dominant_price = dominant_price_data.loc['IM', 'open'].iloc[0]
            contract_notional = dominant_price * 200

            target_short_contracts = math.floor(net_exposure / contract_notional)

            if target_short_contracts > 0:
                sell_open(dominant_contract, target_short_contracts)
            elif target_short_contracts < 0:
                buy_close(dominant_contract, -target_short_contracts)

    def after_trading(context):
        if context.run_info.end_date == context.now.date():
            if not context.unfinished_sell_orders.empty():
                stock_name, quantity = context.unfinished_sell_orders.get()
                print(f"{stock_name} {quantity} shares unfinished sell order")
            else:
                print("No unfinished sell order")

    return init, handle_bar, after_trading


def get_buy_value_tag(buy_value):
    """生成买入金额的文件名标签，支持小数（如3.5万→3p5w）"""
    w = buy_value / 10000
    if w == int(w):
        return f"{int(w)}w"
    else:
        return f"{w}w".replace(".", "p")


def get_buy_value_label(buy_value):
    """生成买入金额的中文标签"""
    w = buy_value / 10000
    if w == int(w):
        return f"{int(w)}万"
    else:
        return f"{w}万"


def run_single_backtest(holding_days, buy_value, start_date, end_date, init_stock_amount):
    """运行单个回测"""
    tag = get_buy_value_tag(buy_value)
    label_str = get_buy_value_label(buy_value)
    print(f"\n{'='*60}")
    print(f"  开始回测: 持仓 {holding_days} 天 | 单笔买入 {label_str}")
    print(f"{'='*60}")

    my_config = copy.deepcopy(__sample_config__)
    my_config["base"]["start_date"] = start_date
    my_config["base"]["end_date"] = end_date
    my_config["base"]["accounts"]["STOCK"] = init_stock_amount
    my_config["base"]["accounts"]["FUTURE"] = init_stock_amount
    my_config["mod"]["sys_analyser"]["benchmark"] = None
    my_config["mod"]["sys_analyser"]["strategy_name"] = f"incentive_{holding_days}d"

    init_func, handle_bar_func, after_trading_func = make_strategy_functions(holding_days, buy_value)

    from rqalpha import run_func

    results = run_func(
        init=init_func,
        handle_bar=handle_bar_func,
        after_trading=after_trading_func,
        config=my_config,
    )

    save_path = rf"F:\quant\research\equity_incentive\results_{holding_days}d_{tag}.pkl"
    pd.to_pickle(results, save_path)
    print(f"  ✓ 持仓 {holding_days} 天 | 单笔{label_str} 回测完成，结果已保存至: {save_path}")

    return results


def compute_metrics(results, label):
    """计算回测指标"""
    results = results['sys_analyser']

    stock_start_cash = float(
        {k: v.strip() for k, v in re.findall(r'(\w+):([^,]+)', results["summary"]["starting_cash"])}["STOCK"])

    stock_account = results['stock_account']
    stock_account["daily_pnl"] = stock_account["total_value"].diff().fillna(
        stock_account["total_value"].iloc[0] - stock_start_cash
    )

    future_account = results['future_account']

    # 多空轧差后累计收益
    daily_total_pnl = stock_account["daily_pnl"] + future_account["daily_pnl"]

    # 固定本金日收益
    daily_return = daily_total_pnl / stock_start_cash

    # 净值曲线
    net_value = 1 + daily_return.cumsum()

    # 最大净值亏损（绝对回撤）
    abs_drawdown = net_value.cummax() - net_value
    max_drawdown_abs = abs_drawdown.max()

    # 总收益率
    total_return = net_value.iloc[-1] - 1

    # 资金利用率 = 持仓市值 / 总资产
    cash_usage = stock_account["market_value"] / stock_account["total_value"]
    avg_cash_usage = cash_usage.mean()
    max_cash_usage = cash_usage.max()

    # 其他指标
    n_days = len(net_value)
    annualized_return = (net_value.iloc[-1]) ** (252 / n_days) - 1
    sharpe_ratio = np.sqrt(252) * daily_return.mean() / daily_return.std() if daily_return.std() > 0 else 0
    win_rate = (daily_return > 0).mean()
    annualized_vol = daily_return.std() * np.sqrt(252)

    return {
        "label": label,
        "net_value": net_value,
        "daily_return": daily_return,
        "total_return": total_return,
        "total_return_pct": total_return * 100,
        "max_drawdown_abs": max_drawdown_abs,
        "max_drawdown_pct": max_drawdown_abs * 100,
        "annualized_return": annualized_return,
        "annualized_return_pct": annualized_return * 100,
        "sharpe_ratio": sharpe_ratio,
        "win_rate": win_rate,
        "win_rate_pct": win_rate * 100,
        "annualized_vol": annualized_vol,
        "annualized_vol_pct": annualized_vol * 100,
        "final_net_value": net_value.iloc[-1],
        "n_days": n_days,
        "stock_start_cash": stock_start_cash,
        "abs_drawdown": abs_drawdown,
        "cash_usage": cash_usage,
        "avg_cash_usage": avg_cash_usage,
        "avg_cash_usage_pct": avg_cash_usage * 100,
        "max_cash_usage": max_cash_usage,
        "max_cash_usage_pct": max_cash_usage * 100,
    }


def plot_comparison(all_metrics):
    """绘制对比图"""
    fig, axes = plt.subplots(4, 1, figsize=(20, 16), sharex=True,
                             height_ratios=[3, 1, 1, 1])

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, metrics in enumerate(all_metrics):
        net_value = metrics["net_value"]
        label = metrics["label"]
        color = colors[i % len(colors)]

        axes[0].plot(net_value, label=label, color=color, linewidth=1.2)
        axes[1].fill_between(metrics["abs_drawdown"].index,
                            metrics["abs_drawdown"].values,
                            alpha=0.3, color=color, label=label)
        axes[2].plot(metrics["daily_return"], label=label, color=color, linewidth=0.8, alpha=0.8)
        axes[3].plot(metrics["cash_usage"], label=label, color=color, linewidth=0.8, alpha=0.8)

    axes[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_title("不同策略净值对比", fontsize=14)
    axes[0].set_ylabel("净值")
    axes[0].legend(loc="upper left")

    axes[1].set_title("绝对回撤对比", fontsize=12)
    axes[1].set_ylabel("回撤")
    axes[1].legend(loc="upper left")

    axes[2].set_title("日收益率对比", fontsize=12)
    axes[2].set_ylabel("日收益率")
    axes[2].legend(loc="upper left")

    axes[3].set_title("资金利用率对比", fontsize=12)
    axes[3].set_ylabel("资金利用率")
    axes[3].set_xlabel("日期")
    axes[3].legend(loc="upper left")

    plt.tight_layout()
    output_path = r"F:\quant\research\equity_incentive\strategy_comparison_v2.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n对比图已保存至: {output_path}")


def print_comparison_table(all_metrics):
    """打印对比表格"""
    print("\n" + "=" * 100)
    print("  不同策略收益与资金利用率对比")
    print("=" * 100)

    headers = ["指标"]
    for m in all_metrics:
        headers.append(m["label"])

    rows = []
    rows.append(["总收益率(%)"] + [f"{m['total_return_pct']:.2f}" for m in all_metrics])
    rows.append(["年化收益率(%)"] + [f"{m['annualized_return_pct']:.2f}" for m in all_metrics])
    rows.append(["最大净值亏损(绝对)"] + [f"{m['max_drawdown_abs']:.4f}" for m in all_metrics])
    rows.append(["最大净值亏损(%)"] + [f"{m['max_drawdown_pct']:.2f}" for m in all_metrics])
    rows.append(["夏普比率"] + [f"{m['sharpe_ratio']:.4f}" for m in all_metrics])
    rows.append(["年化波动率(%)"] + [f"{m['annualized_vol_pct']:.2f}" for m in all_metrics])
    rows.append(["日胜率(%)"] + [f"{m['win_rate_pct']:.2f}" for m in all_metrics])
    rows.append(["期末净值"] + [f"{m['final_net_value']:.4f}" for m in all_metrics])
    rows.append(["平均资金利用率(%)"] + [f"{m['avg_cash_usage_pct']:.2f}" for m in all_metrics])
    rows.append(["最大资金利用率(%)"] + [f"{m['max_cash_usage_pct']:.2f}" for m in all_metrics])
    rows.append(["回测天数"] + [str(m['n_days']) for m in all_metrics])

    # 计算列宽
    col_widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]

    header_line = "│ " + " │ ".join(headers[i].center(col_widths[i]) for i in range(len(headers))) + " │"
    sep_line = "┼─" + "─┼─".join("─" * col_widths[i] for i in range(len(headers))) + "─┤"

    print(header_line)
    print(sep_line)

    for row in rows:
        row_line = "│ " + " │ ".join(str(row[i]).rjust(col_widths[i]) for i in range(len(row))) + " │"
        print(row_line)

    print("=" * 100)
    print("\n计算说明:")
    print("  总收益率 = 多空轧差累计收益 / 多头股票端期初本金(固定值1000万)")
    print("  最大净值亏损 = max(净值前高 - 当前净值)，即距离最高点的最大绝对值亏损")
    print("  资金利用率 = 股票端持仓市值 / 股票端总资产")


def main():
    rqdatac.init()

    start_date = "2023-01-01"
    end_date = "2026-07-01"
    init_stock_amount = 10000000

    # 策略配置: (持仓天数, 单笔买入金额)
    strategy_configs = [
        (22, 100000),   # 22天, 单笔10万
        (60, 50000),    # 60天, 单笔5万
        (90, 35000),    # 90天, 单笔3.5万
    ]

    # 运行所有回测（跳过已有结果）
    all_results = {}
    for holding_days, buy_value in strategy_configs:
        tag = get_buy_value_tag(buy_value)
        save_path = rf"F:\quant\research\equity_incentive\results_{holding_days}d_{tag}.pkl"
        if os.path.exists(save_path):
            print(f"\n  跳过已有结果: {save_path}")
            all_results[(holding_days, buy_value)] = pd.read_pickle(save_path)
        else:
            results = run_single_backtest(holding_days, buy_value, start_date, end_date, init_stock_amount)
            all_results[(holding_days, buy_value)] = results

    # 计算指标
    all_metrics = []
    for holding_days, buy_value in strategy_configs:
        label = f"{holding_days}天/单笔{get_buy_value_label(buy_value)}"
        metrics = compute_metrics(all_results[(holding_days, buy_value)], label)
        all_metrics.append(metrics)

    # 打印对比表格
    print_comparison_table(all_metrics)

    # 绘制对比图
    plot_comparison(all_metrics)

    # 保存数值对比CSV
    csv_data = {
        "策略": [m["label"] for m in all_metrics],
        "总收益率(%)": [round(m["total_return_pct"], 2) for m in all_metrics],
        "年化收益率(%)": [round(m["annualized_return_pct"], 2) for m in all_metrics],
        "最大净值亏损(绝对)": [round(m["max_drawdown_abs"], 4) for m in all_metrics],
        "最大净值亏损(%)": [round(m["max_drawdown_pct"], 2) for m in all_metrics],
        "夏普比率": [round(m["sharpe_ratio"], 4) for m in all_metrics],
        "年化波动率(%)": [round(m["annualized_vol_pct"], 2) for m in all_metrics],
        "日胜率(%)": [round(m["win_rate_pct"], 2) for m in all_metrics],
        "期末净值": [round(m["final_net_value"], 4) for m in all_metrics],
        "平均资金利用率(%)": [round(m["avg_cash_usage_pct"], 2) for m in all_metrics],
        "最大资金利用率(%)": [round(m["max_cash_usage_pct"], 2) for m in all_metrics],
    }
    df = pd.DataFrame(csv_data)
    csv_path = r"F:\quant\research\equity_incentive\strategy_comparison_v2.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n数值对比已保存至: {csv_path}")


if __name__ == "__main__":
    main()