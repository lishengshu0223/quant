"""把本地 category 缓存回填到已下载公告 parquet 的 category 列

读取 {ANNOUNCEMENTS_DIR}/category_cache_*.parquet 合并成 {annId: 类别串}，
按 ann_id 更新各日公告文件的 category 列（仅更新当前为空的行，非空跳过）。

用法：
    python -m download.announcements.apply_category_cache [--start 20160101] [--end 20260805]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from ..logger import logger
from ..config import get_data_path, ANNOUNCEMENTS_DIR


def load_all_cache():
    """合并所有月份 category 缓存 → {ann_id: "类1;类2"}"""
    out_dir = get_data_path(ANNOUNCEMENTS_DIR)
    cache = {}
    for fname in sorted(os.listdir(out_dir)):
        if not fname.startswith("category_cache_") or not fname.endswith(".parquet"):
            continue
        fpath = os.path.join(out_dir, fname)
        try:
            df = pd.read_parquet(fpath)
            for aid, c in zip(df["ann_id"], df["categories"]):
                aid = str(aid)
                cache[aid] = str(c) if c else ""
        except Exception as e:
            logger.warning(f"读取缓存失败 {fname}: {e}")
    logger.info(f"合并 category 缓存 {len(cache)} 条")
    return cache


def apply_to_files(cache, start=None, end=None):
    """按日期文件回填 category 列"""
    out_dir = get_data_path(ANNOUNCEMENTS_DIR)
    updated_days, skipped_days, missing = 0, 0, 0
    for fname in sorted(os.listdir(out_dir)):
        if not (fname.endswith(".parquet")
                and fname[:8].isdigit() and len(fname) == 16):
            continue
        ds = fname[:8]
        if start and ds < start:
            continue
        if end and ds > end:
            continue
        fpath = os.path.join(out_dir, fname)
        df = pd.read_parquet(fpath)
        if "ann_id" not in df.columns or "category" not in df.columns:
            continue
        empty_mask = df["category"].fillna("").eq("")
        if not empty_mask.any():
            skipped_days += 1
            continue
        n = int(empty_mask.sum())
        filled = df.loc[empty_mask, "ann_id"].map(cache).fillna("")
        df.loc[empty_mask, "category"] = filled
        still_missing = int(filled.eq("").sum())
        missing += still_missing
        df.to_parquet(fpath, engine="pyarrow")
        updated_days += 1
        if n > 0 and (n - still_missing) > 0:
            logger.info(f"{ds}: 回填 {n - still_missing}/{n} 条（缓存缺 {still_missing}）")
    logger.info(f"回填结束: 更新 {updated_days} 个文件，跳过 {skipped_days} 个"
                f"（category 已完整），缓存未命中 {missing} 条")


def main():
    parser = argparse.ArgumentParser(description="回填公告 category 列（巨潮官方 26 类）")
    parser.add_argument("--start", default=None, help="起始日期 YYYYMMDD，默认全部")
    parser.add_argument("--end", default=None, help="结束日期 YYYYMMDD，默认全部")
    args = parser.parse_args()

    cache = load_all_cache()
    if not cache:
        logger.warning("未找到 category 缓存，请先运行 generate_category_cache.py")
        return
    apply_to_files(cache, args.start, args.end)


if __name__ == "__main__":
    main()
