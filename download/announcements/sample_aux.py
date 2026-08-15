# -*- coding: utf-8 -*-
"""对指定巨潮码抽样公告标题，供人工判断语义后补入映射表

用途：公告分类映射表维护第 2 步（见 .trae/rules/announcement_classification.md 第 5 节）。
配合 scan_all_codes.py 使用：先扫描出未收录码，再对本脚本传入这些码抽样标题。

使用方式:
  conda activate multifactor
  python f:\quant\download\announcements\sample_aux.py 01234567 012345 ... [--n 10] [--start 2026-01-01] [--end 2026-12-31]
"""
import argparse
import glob
import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, r"f:\quant")
from download.config import get_data_path, ANNOUNCEMENTS_DIR  # noqa: E402

DATA_DIR = get_data_path(ANNOUNCEMENTS_DIR)


def main():
    ap = argparse.ArgumentParser(description="抽样指定巨潮码的公告标题")
    ap.add_argument("codes", nargs="+", help="待判断语义的巨潮码（可多个）")
    ap.add_argument("--n", type=int, default=10, help="每个码抽样标题条数（默认 10）")
    ap.add_argument("--start", default=None, help="起始日期 YYYY-MM-DD（默认全部）")
    ap.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD（默认全部）")
    args = ap.parse_args()

    targets = set(args.codes)
    samples = defaultdict(list)
    n_files = 0
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "20*.parquet"))):
        ds = os.path.basename(f)[:8]
        if not ds.isdigit():
            continue
        if args.start and ds < args.start.replace("-", ""):
            continue
        if args.end and ds > args.end.replace("-", ""):
            continue
        n_files += 1
        df = pd.read_parquet(f, columns=["type_codes", "title"])
        for tc, ti in zip(df["type_codes"], df["title"]):
            codes = {c.strip() for c in str(tc or "").split("||") if c.strip()}
            hit = codes & targets
            if not hit:
                continue
            for c in hit:
                if len(samples[c]) < args.n and ti:
                    samples[c].append(str(ti)[:80])

    for c in args.codes:
        ss = samples.get(c, [])
        print("=" * 70)
        print(f"码 {c}: 抽样 {len(ss)} 条标题（已扫 {n_files} 个文件）")
        for t in ss:
            print("  -", t)
        if not ss:
            print("  未找到携带该码的公告")


if __name__ == "__main__":
    main()
