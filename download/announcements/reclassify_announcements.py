"""用最新映射表重算已下载公告的分类列（cninfo_*/csrc_*）

背景：回填/更新进程启动时加载当次的映射表；运行期间映射表被增量补充后，
已落盘文件的分类列可能为空或不完整。本脚本用当前最新映射重算全部日期文件。

用法：
    python -m download.announcements.reclassify_announcements [--start 20160101] [--end 20260805]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from .announcement_type_map import CLASSIFY_COLUMNS, classify
from ..logger import logger
from ..config import get_data_path, ANNOUNCEMENTS_DIR


def reclassify_files(start=None, end=None):
    """按日期文件重算分类列并写回，返回 (处理文件数, 更新行数)"""
    out_dir = get_data_path(ANNOUNCEMENTS_DIR)
    n_files = n_rows = 0
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
        if "type_codes" not in df.columns:
            continue
        need = df["type_codes"].notna() & (df["type_codes"] != "")
        if not need.any():
            continue
        rows = [classify(str(c)) for c in df.loc[need, "type_codes"]]
        for col in CLASSIFY_COLUMNS:
            df.loc[need, col] = [r[col] for r in rows]
        df.to_parquet(fpath, engine="pyarrow")
        n_files += 1
        n_rows += int(need.sum())
    return n_files, n_rows


def main():
    parser = argparse.ArgumentParser(description="重算公告分类列（最新映射表）")
    parser.add_argument("--start", default=None, help="起始日期 YYYYMMDD，默认全部")
    parser.add_argument("--end", default=None, help="结束日期 YYYYMMDD，默认全部")
    args = parser.parse_args()

    n_files, n_rows = reclassify_files(args.start, args.end)
    logger.info(f"重算完成: {n_files} 个文件, {n_rows} 行分类列已更新")


if __name__ == "__main__":
    main()
