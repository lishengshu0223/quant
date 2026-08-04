import sys
import warnings
import os
import pandas as pd
import rqdatac
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)

START_DATE = "2016-01-01"


def get_end_date():
    now = datetime.datetime.now()
    if now.hour >= 17:
        return pd.Timestamp.today().date()
    else:
        return (pd.Timestamp.today() - pd.Timedelta(days=1)).date()


def main():
    rqdatac.init()
    
    from download import (
        run_with_exception_handling,
        download_stock_daily_price,
        download_index_daily_price,
        download_index_weights,
        download_all_barra,
        logger,
    )
    from update.tradable_status import download_tradable_status
    
    end_date = get_end_date()
    end_date_str = end_date.strftime("%Y%m%d")
    
    logger.info("=" * 60)
    logger.info("补全历史缺失交易数据")
    logger.info("=" * 60)
    logger.info(f"日期范围: {START_DATE} 至 {end_date_str}")
    logger.info(f"当前时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    run_with_exception_handling(download_stock_daily_price, end_date=end_date)
    
    run_with_exception_handling(download_index_daily_price, end_date=end_date)
    
    run_with_exception_handling(download_index_weights, end_date=end_date)
    
    run_with_exception_handling(download_all_barra, end_date=end_date)
    
    run_with_exception_handling(download_tradable_status, end_date=end_date)
    
    logger.info("=" * 60)
    logger.info("历史数据补全检查完成")
    logger.info("=" * 60)
    
    print(f"历史数据补全检查完成: {START_DATE} 至 {end_date_str}")


if __name__ == "__main__":
    main()