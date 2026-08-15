"""A股公告数据每日独立更新脚本（与量价数据分开调度）

背景：公告按自然日发布（含周末/节假日），披露时点分两批：
  00:00 档 —— 前一交易日盘后提交、选"次日披露"的公告凌晨 0 点统一上架；
  早间档   —— 当日 7:30-8:30 披露（沪市早间披露时段，深市 7:30-8:00 提交实时发送）。
因此每天在 00:30 与 08:50 两个时点各跑一次，分别覆盖两批，开盘前信息全齐。

流程：
  1. 强制重下最近 N 个自然日（捕获盘后延迟发布的公告，幂等，可重复执行）；
  2. 用最新映射表重算最近若干天的分类列（cninfo_*/csrc_*）；
  3. 回填 category 列（巨潮官方 26 类）：先按 type_codes 静态映射解码
     （type_category_map），映射未覆盖的公告自动调用巨潮 26 类查询补学
     并更新映射，确保遇到当前分类之外的代码编号也能自动解码入库。

说明：
- category 静态解码依据：巨潮公告的 announcementType 细码串唯一确定官方
  26 类（实测一致），无需逐月查询 26 类接口，速度提升数十倍且不受巨潮
  单次查询 3000 条上限影响（详见 download/announcements/type_category_map.py）。
- 分类列对未收录的巨潮细码不会失败——层级码按 2 位一段自动切分，
  中文名取最近已知前缀并日志告警，数据照常入库。

不依赖 rqdatac（公告按自然日发布，周末/节假日也要跑），与量价数据更新完全独立。

用法：
    python -m update.update_announcements [--slot 0030] [--days 3] [--recalc-days 7]
"""
import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from download import (
    run_with_exception_handling,
    download_recent_announcements,
    logger,
)
from download.announcements.reclassify_announcements import reclassify_files
from download.announcements.type_category_map import (
    ensure_map, decode_apply_and_learn,
)
from download.config import get_data_path, ANNOUNCEMENTS_DIR, ensure_dir


def _recent_range(days):
    """最近 days 个自然日的 [YYYYMMDD 起, YYYYMMDD 止]"""
    end = datetime.date.today()
    start = end - datetime.timedelta(days=days - 1)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


def main():
    parser = argparse.ArgumentParser(description="A股公告数据每日独立更新")
    parser.add_argument("--slot", default="", help="调度时段标识（如 0030/0845），仅用于日志")
    parser.add_argument("--days", type=int, default=3, help="强制重下的最近自然日天数")
    parser.add_argument("--recalc-days", type=int, default=7,
                        help="重算分类/回填 category 的最近自然日天数")
    parser.add_argument("--no-learn-api", action="store_true",
                        help="关闭对映射未覆盖 type 串的巨潮 API 补学（默认开启）")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"公告数据独立更新开始（slot={args.slot or '手动'}）")
    logger.info(f"当前时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # 1. 强制重下最近 N 个自然日（覆盖 00:00 档 + 早间档 + 延迟补录）
    run_with_exception_handling(download_recent_announcements, days=args.days)

    # 2. 用最新映射表重算最近若干天的分类列（cninfo_*/csrc_*）
    s, e = _recent_range(args.recalc_days)
    n_files, n_rows = reclassify_files(start=s, end=e)
    logger.info(f"重算分类列: {n_files} 个文件, {n_rows} 行")

    # 3. 回填 category 列（巨潮官方 26 类）：
    #    先确保 type→category 静态映射存在，再按 type_codes 解码回填；
    #    新公告中映射未覆盖的 type 串自动调用巨潮 26 类查询补学并更新映射。
    learn_api = not args.no_learn_api
    run_with_exception_handling(
        ensure_map,
        learn_api=learn_api,
    )
    filled, still_missing = decode_apply_and_learn(
        start=s, end=e, learn_api=learn_api)
    logger.info(f"category 回填: 已填 {filled} 条，仍未覆盖 {still_missing} 条")

    logger.info("=" * 60)
    logger.info("公告数据独立更新结束")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
