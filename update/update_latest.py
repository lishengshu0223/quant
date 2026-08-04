import sys
import warnings
import os
import pandas as pd
import rqdatac
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)


def get_end_date():
    now = datetime.datetime.now()
    if now.hour >= 17:
        return rqdatac.get_latest_trading_date(market="cn")
    else:
        latest_date = rqdatac.get_latest_trading_date(market="cn")
        return rqdatac.get_previous_trading_date(latest_date, market="cn")


def main():
    rqdatac.init()
    
    from download import (
        run_with_exception_handling,
        download_all_instruments,
        download_trading_dates,
        download_stock_daily_price,
        download_index_daily_price,
        download_index_weights,
        download_all_barra,
        logger,
    )
    from update.tradable_status import download_tradable_status
    
    end_date = get_end_date()
    date_str = end_date.strftime("%Y%m%d")
    
    logger.info("=" * 60)
    logger.info("强制更新最新交易日数据")
    logger.info("=" * 60)
    logger.info(f"目标日期: {date_str}")
    logger.info(f"当前时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    run_with_exception_handling(download_all_instruments)
    
    run_with_exception_handling(download_trading_dates)
    
    run_with_exception_handling(download_stock_daily_price, download_dates=[date_str], force=True)
    
    run_with_exception_handling(download_index_daily_price, download_dates=[date_str], force=True)
    
    run_with_exception_handling(download_index_weights, download_dates=[date_str], force=True)
    
    run_with_exception_handling(download_all_barra, download_dates=[date_str], force=True)
    
    run_with_exception_handling(download_tradable_status, download_dates=[date_str], force=True)
    
    logger.info("=" * 60)
    logger.info("最新交易日数据更新完成")
    logger.info("=" * 60)
    
    print(f"最新交易日数据更新完成: {date_str}")


if __name__ == "__main__":
    main()