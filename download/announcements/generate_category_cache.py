"""生成巨潮官方 26 类 category 缓存（按月分片，支持断点续传）

背景：category 列（巨潮官方 26 类：年报/业绩预告/股东会等）需要把全市场公告
按 26 个类别各查一遍才能还原，查询总量与公告总量成正比，逐批实时查询会触发
巨潮限流。因此改为后台一次性生成缓存，下载时不再查询、之后统一回填。

巨潮接口单次查询最多返回 3000 条（100 页 × 30 条），超过后循环返回重复内容。
为避免整月查询被截断（高峰期月份如 4 月/8 月/12 月单月单类常超 3000 条），
本脚本按月内 **3 天小窗口** 逐类别查询合并；窗口内仍超限时自动降级为
逐日 × 逐板块（sz/sh）分片查询（见 _paginate_split）。

输出：{ANNOUNCEMENTS_DIR}/category_cache_YYYYMM.parquet，每文件两列：
    ann_id      巨潮公告唯一编号
    categories  所属巨潮类别（";" 连接，可能多个）

用法：
    python -m download.announcements.generate_category_cache --start 2016-01 --end 2026-08
    （已存在的月份文件自动跳过，可中断后重跑续传；如需强制重建当月缓存，
      由 update_announcements.py 的 _rebuild_month_cache 直接调用 generate_month）
"""
import argparse
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from .announcements import (_paginate, _paginate_split, _post_query,
                            ANNOUNCEMENT_CATEGORIES, logger)
from ..config import get_data_path, ANNOUNCEMENTS_DIR, ensure_dir

WINDOW_DAYS = 3  # 窗口天数：实测 3 天窗口内单类公告数远低于 3000 上限


def _merge_items(cat_map, items, name):
    """把一类查询结果合并进 cat_map"""
    for a in items:
        aid = str(a.get("announcementId") or "")
        if aid:
            cat_map.setdefault(aid, set()).add(name)


def _fetch_category(se_date, code, name, cat_map):
    """查询窗口内某类别公告并合并到 cat_map；返回该类别是否被 3000 条截断"""
    items, _, truncated = _paginate(se_date, category=code)
    _merge_items(cat_map, items, name)
    return truncated


def generate_month(year_month):
    """生成单月 category 缓存，返回 (查询范围, 映射条数)；异常抛出

    自适应分片：
      1. 先按整月查询各类别（page=1 的 total ≤ 3000 时一次拿全）；
      2. 整月超限的类别，按月内 3 天小窗口细分查询；
      3. 3 天窗口仍超限（极端高峰日），再降级为逐日 × 板块分片。
    """
    y, m = year_month
    last_day = (datetime.date(y + (1 if m == 12 else 0), 1 if m == 12 else m + 1, 1)
                - datetime.timedelta(days=1)).day
    se_date = f"{y:04d}-{m:02d}-01~{y:04d}-{m:02d}-{last_day:02d}"
    cat_map = {}

    # 1) 整月查询：先探测各类别 total，超 3000 才细分（避免无谓翻页）
    need_window = []
    for code, name in ANNOUNCEMENT_CATEGORIES.items():
        j = _post_query(se_date, page_num=1, category=code)
        total = j.get("totalAnnouncement") or 0
        if total > 3000:
            logger.info(f"{y:04d}-{m:02d} 类别[{name}] 整月 {total} 条超上限，按窗口细分")
            need_window.append((code, name))
        else:
            _fetch_category(se_date, code, name, cat_map)

    # 2) 超限类别：按月内 3 天窗口细分
    for code, name in need_window:
        day = 1
        while day <= last_day:
            d_end = min(day + WINDOW_DAYS - 1, last_day)
            w_se = f"{y:04d}-{m:02d}-{day:02d}~{y:04d}-{m:02d}-{d_end:02d}"
            truncated = _fetch_category(w_se, code, name, cat_map)
            if truncated:
                # 3) 极端高峰窗口（如 4 月末）：逐日 × 板块分片兜底
                logger.warning(f"{w_se} 类别[{name}] 仍超 3000 条上限，逐日×板块分片")
                items = _paginate_split(w_se, category=code)
                _merge_items(cat_map, items, name)
            day = d_end + 1
    return se_date, cat_map


def main():
    parser = argparse.ArgumentParser(description="生成巨潮官方 26 类 category 缓存（按月分片）")
    parser.add_argument("--start", default="2016-01", help="起始年月 YYYY-MM，默认 2016-01")
    parser.add_argument("--end", default=None, help="结束年月 YYYY-MM，默认当前月")
    args = parser.parse_args()

    sy, sm = map(int, args.start.split("-"))
    ey, em = (map(int, args.end.split("-")) if args.end
              else (datetime.date.today().year, datetime.date.today().month))

    out_dir = ensure_dir(get_data_path(ANNOUNCEMENTS_DIR))
    months = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        months.append((y, m))
        m += 1
        if m > 12:
            y, m = y + 1, 1

    logger.info(f"生成 category 缓存: {len(months)} 个月份 ({args.start} ~ "
                f"{args.end or f'{ey:04d}-{em:02d}'})，输出目录 {out_dir}")
    for y, m in months:
        fname = f"category_cache_{y:04d}{m:02d}.parquet"
        fpath = os.path.join(out_dir, fname)
        if os.path.exists(fpath):
            logger.info(f"跳过已存在: {fname}")
            continue
        try:
            se_date, cat_map = generate_month((y, m))
            if not cat_map:
                logger.warning(f"{y:04d}-{m:02d} 分类映射为空，仍落盘占位")
            df = pd.DataFrame(
                {"ann_id": list(cat_map.keys()),
                 "categories": [";".join(sorted(v)) for v in cat_map.values()]})
            df.to_parquet(fpath, engine="pyarrow")
            logger.info(f"{fname} 完成: {len(df)} 条 ({se_date})")
        except Exception as e:
            logger.error(f"{y:04d}-{m:02d} 生成失败（下次重跑续传）: {e}")

    logger.info("category 缓存生成结束（失败的月份将在下次重跑时重试）")


if __name__ == "__main__":
    main()
