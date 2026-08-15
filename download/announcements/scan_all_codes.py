# -*- coding: utf-8 -*-
"""全量扫描公告类型码，与 CN_CODE_NAMES 比对，暴露未收录码（含非主码角色）

用途：公告分类映射表维护第 1 步（见 .trae/rules/announcement_classification.md 第 5 节）。
对已下载公告的 announcementType 码串做全量统计，区分主码/辅助码角色，
与 download/announcement_type_map.py 的 CN_CODE_NAMES 比对，输出：
  1. 未收录码（含出现行数、主码行数、示例标题）—— 需人工补码
  2. 已收录但仅作辅助角色出现的码 —— 供中性命名核查
  3. 缺失汇总

使用方式:
  conda activate multifactor
  python f:\quant\download\announcements\scan_all_codes.py [--start 2026-01-01] [--end 2026-12-31] [--limit N]
"""
import argparse
import glob
import os
import sys
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, r"f:\quant")
from download.announcements.announcement_type_map import CN_CODE_NAMES  # noqa: E402
from download.config import get_data_path, ANNOUNCEMENTS_DIR  # noqa: E402

DATA_DIR = get_data_path(ANNOUNCEMENTS_DIR)


def iter_files(start=None, end=None):
    """按日期范围遍历公告 parquet 文件（YYYYMMDD.parquet）"""
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "20*.parquet"))):
        ds = os.path.basename(f)[:8]
        if not ds.isdigit():
            continue
        if start and ds < start.replace("-", ""):
            continue
        if end and ds > end.replace("-", ""):
            continue
        yield f


def main():
    ap = argparse.ArgumentParser(description="全量扫描巨潮公告类型码，与 CN_CODE_NAMES 比对")
    ap.add_argument("--start", default=None, help="起始日期 YYYY-MM-DD（默认全部）")
    ap.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD（默认全部）")
    ap.add_argument("--limit", type=int, default=0, help="限定扫描文件数（冒烟测试用）")
    args = ap.parse_args()

    code_hits = Counter()          # 码 -> 出现行数（所有角色）
    code_primary = Counter()       # 码 -> 作为主码的行数
    sample_titles = defaultdict(list)  # 码 -> 示例标题

    n_files = 0
    for f in iter_files(args.start, args.end):
        n_files += 1
        if args.limit and n_files > args.limit:
            break
        df = pd.read_parquet(f, columns=["type_codes", "cninfo_type_code", "title"])
        for tc, pc, ti in zip(df["type_codes"], df["cninfo_type_code"], df["title"]):
            codes = [c.strip() for c in str(tc or "").split("||") if c.strip()]
            for c in codes:
                code_hits[c] += 1
                if str(pc or "") == c:
                    code_primary[c] += 1
                if len(sample_titles[c]) < 5 and ti:
                    sample_titles[c].append(str(ti)[:60])
        if n_files % 200 == 0:
            print(f"已扫描 {n_files} 个文件...", flush=True)

    known = set(CN_CODE_NAMES)
    all_codes = set(code_hits)
    unknown = sorted(all_codes - known)
    aux_only = sorted(c for c in (known & all_codes) if code_primary[c] == 0)

    print("=" * 70)
    print(f"扫描完成: {n_files} 个文件, 出现 {len(all_codes)} 个码, 映射表 {len(known)} 个码")
    print(f"未收录码: {len(unknown)} 个 | 已收录但仅辅助角色: {len(aux_only)} 个")
    print("=" * 70)
    if unknown:
        print("\n【未收录码】(码 | 出现行数 | 主码行数 | 示例标题)")
        for c in unknown:
            titles = " | ".join(sample_titles[c][:3]) or "(无标题)"
            print(f"  {c} | {code_hits[c]} | {code_primary[c]} | {titles}")
    if aux_only:
        print("\n【已收录但仅辅助角色】(码 | 出现行数 | 示例标题)")
        for c in aux_only:
            titles = " | ".join(sample_titles[c][:2]) or "(无标题)"
            print(f"  {c} | {code_hits[c]} | {titles}")
    print("\n缺失为 0 即视为映射表完整；仍有未收录码时，用 download/announcements/sample_aux.py 抽样标题判断语义后补码。")


if __name__ == "__main__":
    main()
