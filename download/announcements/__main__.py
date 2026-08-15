"""CLI 入口：python -m download.announcements [--start D [--end D] | --recent N] [--force]

用法：
    # 补全 2016 年至今的历史数据（自动跳过已下载日期，可中断后重跑续传）
    python -m download.announcements --start 2016-01-01

    # 每日更新最近 3 天（强制覆盖，捕获盘后延迟发布的公告）
    python -m download.announcements --recent 3
"""
import argparse

from .announcements import download_announcements, download_recent_announcements


def main():
    parser = argparse.ArgumentParser(description="A股公告特征数据下载（巨潮资讯网）")
    parser.add_argument("--start", default=None, help="起始日期，默认 2016-01-01")
    parser.add_argument("--end", default=None, help="结束日期，默认今天")
    parser.add_argument("--recent", type=int, default=0,
                        help="强制重下最近 N 天（与 --start/--end 互斥）")
    parser.add_argument("--force", action="store_true", help="强制覆盖已有日期")
    parser.add_argument("--category-cache", default=None,
                        help="本地 category 缓存 parquet 路径（可选，存在则下载时直接读取 category 列）")
    args = parser.parse_args()

    if args.recent > 0:
        download_recent_announcements(days=args.recent)
    else:
        download_announcements(start_date=args.start, end_date=args.end,
                               force=args.force,
                               category_cache_path=args.category_cache)


if __name__ == "__main__":
    main()