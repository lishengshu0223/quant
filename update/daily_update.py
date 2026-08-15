"""量价数据每日更新统一入口（股票/指数/权重/Barra/换手率/股本/可交易状态）

此前由 update_latest.py（最新交易日）与 update_history.py（历史补全）两个脚本
分担，二者内容高度重叠，现合并为本脚本，以 --mode 区分两个阶段，可分开运行
也可默认串行执行：

    --mode latest  强制更新最新交易日数据（捕获盘后延迟发布/修正的数据，幂等）
    --mode history 补全 2016-01-01 以来历史缺失交易日（自动跳过已有日期，断点续传）
    --mode all     先 latest 后 history（默认）

公告数据按自然日发布（含周末/节假日），与量价数据相互独立，由
update/update_announcements.py（00:30 / 08:50 两个计划任务）单独调度，本脚本
不再涉及公告下载。

用法：
    python update/daily_update.py --mode latest
    python update/daily_update.py --mode history
    python update/daily_update.py            # 等价 --mode all
"""
import argparse
import datetime
import os
import sys
import warnings

import pandas as pd
import rqdatac

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)

START_DATE = "2016-01-01"


def latest_trading_date():
    """最新交易日：17 点后数据已收齐，取当日；否则回退到上一交易日"""
    now = datetime.datetime.now()
    latest = rqdatac.get_latest_trading_date(market="cn")
    if now.hour >= 17:
        return latest
    return rqdatac.get_previous_trading_date(latest, market="cn")


def _run_latest():
    """强制更新最新交易日数据（幂等，可安全重复执行）"""
    from download import (
        run_with_exception_handling,
        download_all_instruments,
        download_trading_dates,
        download_stock_daily_price,
        download_index_daily_price,
        download_index_weights,
        download_all_barra,
        download_turnover_rate,
        download_shares,
        logger,
    )
    from update.tradable_status import download_tradable_status

    end_date = latest_trading_date()
    date_str = end_date.strftime("%Y%m%d")

    logger.info("=" * 60)
    logger.info("强制更新最新交易日数据")
    logger.info(f"目标日期: {date_str}")
    logger.info(f"当前时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    run_with_exception_handling(download_all_instruments)

    run_with_exception_handling(download_trading_dates)

    run_with_exception_handling(download_stock_daily_price, download_dates=[date_str], force=True)

    run_with_exception_handling(download_index_daily_price, download_dates=[date_str], force=True)

    run_with_exception_handling(download_index_weights, download_dates=[date_str], force=True)

    run_with_exception_handling(download_all_barra, download_dates=[date_str], force=True)

    run_with_exception_handling(download_turnover_rate, download_dates=[date_str], force=True)

    run_with_exception_handling(download_shares, download_dates=[date_str], force=True)

    run_with_exception_handling(download_tradable_status, download_dates=[date_str], force=True)

    logger.info("=" * 60)
    logger.info(f"最新交易日数据更新完成: {date_str}")
    logger.info("=" * 60)

    print(f"最新交易日数据更新完成: {date_str}")


def _run_history():
    """补全历史缺失交易数据（自动跳过已有日期，可中断后重跑续传）"""
    from download import (
        run_with_exception_handling,
        download_stock_daily_price,
        download_index_daily_price,
        download_index_weights,
        download_all_barra,
        download_turnover_rate,
        download_shares,
        logger,
    )
    from update.tradable_status import download_tradable_status

    now = datetime.datetime.now()
    # 17 点后视为当日数据已收齐，否则补到前一天（与 latest 阶段口径一致）
    end_date = (pd.Timestamp.today() if now.hour >= 17
                else (pd.Timestamp.today() - pd.Timedelta(days=1))).date()
    end_date_str = end_date.strftime("%Y%m%d")

    logger.info("=" * 60)
    logger.info("补全历史缺失交易数据")
    logger.info(f"日期范围: {START_DATE} 至 {end_date_str}")
    logger.info(f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    run_with_exception_handling(download_stock_daily_price, end_date=end_date)

    run_with_exception_handling(download_index_daily_price, end_date=end_date)

    run_with_exception_handling(download_index_weights, end_date=end_date)

    run_with_exception_handling(download_all_barra, end_date=end_date)

    run_with_exception_handling(download_tradable_status, end_date=end_date)

    run_with_exception_handling(download_turnover_rate, end_date=end_date)

    run_with_exception_handling(download_shares, end_date=end_date)

    logger.info("=" * 60)
    logger.info("历史数据补全检查完成")
    logger.info("=" * 60)

    print(f"历史数据补全检查完成: {START_DATE} 至 {end_date_str}")


def main():
    parser = argparse.ArgumentParser(description="量价数据每日更新统一入口")
    parser.add_argument("--mode", choices=["latest", "history", "all"], default="all",
                        help="latest=强制最新交易日；history=补历史缺失；all=先 latest 后 history（默认）")
    args = parser.parse_args()

    rqdatac.init()

    if args.mode in ("latest", "all"):
        _run_latest()
    if args.mode in ("history", "all"):
        _run_history()


if __name__ == "__main__":
    main()