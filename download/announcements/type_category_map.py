"""巨潮公告 category（官方 26 类）静态解码映射模块

核心发现：巨潮公告的 category 列（官方 26 类）可由公告的 announcementType
细码串唯一确定（实测单日 881/881 完全一致）。因此无需逐日/逐月调用 26 类
查询接口，只需学习一张"type 段集合 → 26 类"的静态映射表，即可在下载/回填时
直接解码，速度提升数十倍，且不受巨潮单次查询 3000 条上限的影响。

映射表来源：
1. 本地已下载文件中 category 已回填的行（累计 224 万行，覆盖 99.73% 空行）
2. 对映射未覆盖的 type 串，按出现日期定向调用巨潮 26 类查询补学
   （遇到当前分类之外的代码编号时，先自动按日期补学，学不到才告警）

type 串规范化：announcementType 的段顺序不稳定（同一公告在不同查询中可能
返回段序不同的串），故统一按"段集合排序"后作为映射键。

映射文件：{ANNOUNCEMENTS_DIR}/type_category_map.parquet，两列：
    type_codes   规范化后的段集合（"|" 连接，段已排序去重）
    categories   巨潮官方 26 类（";" 连接，可能多个）
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from ..config import get_data_path, ANNOUNCEMENTS_DIR, ensure_dir
from ..logger import logger

MAP_FNAME = "type_category_map.parquet"


def norm_type_codes(type_codes):
    """type 串 → 规范化段集合键（"|" 连接，段已排序去重）"""
    segs = [s for s in str(type_codes).split("||") if s]
    return "|".join(sorted(set(segs))) if segs else ""


def map_path():
    """映射文件完整路径"""
    return os.path.join(get_data_path(ANNOUNCEMENTS_DIR), MAP_FNAME)


def learn_from_files(start=None, end=None):
    """从本地已下载文件中 category 已回填的行学习映射

    返回 {规范段集合: set(类别)}；同一段集合出现过的所有类别取并集
    （历史缓存曾因 3000 条上限缺失部分类别，并集才是完整答案）。
    """
    out_dir = get_data_path(ANNOUNCEMENTS_DIR)
    type_map = {}
    for fname in sorted(os.listdir(out_dir)):
        if not (fname.endswith(".parquet") and fname[:8].isdigit()
                and len(fname) == 16):
            continue
        ds = fname[:8]
        if start and ds < start:
            continue
        if end and ds > end:
            continue
        fpath = os.path.join(out_dir, fname)
        try:
            df = pd.read_parquet(fpath, columns=["type_codes", "category"])
        except Exception as e:
            logger.warning(f"读取 {fname} 失败: {e}")
            continue
        for t, c in zip(df["type_codes"].astype(str), df["category"].astype(str)):
            if not c or c == "nan":
                continue
            cats = {x for x in str(c).split(";") if x}
            if not cats:
                continue
            key = norm_type_codes(t)
            if not key:
                continue
            if key in type_map:
                type_map[key].update(cats)
            else:
                type_map[key] = set(cats)
    logger.info(f"从已回填行学习 type→category 映射: {len(type_map)} 种")
    return type_map


def save_map(type_map):
    """映射落盘（parquet，两列 type_codes/categories）"""
    fpath = map_path()
    df = pd.DataFrame({
        "type_codes": list(type_map.keys()),
        "categories": [";".join(sorted(v)) for v in type_map.values()],
    })
    df.to_parquet(fpath, engine="pyarrow")
    logger.info(f"type→category 映射已保存: {fpath}（{len(df)} 条）")


def load_map():
    """加载映射 → {规范段集合: set(类别)}；不存在或异常返回空 dict"""
    fpath = map_path()
    if not os.path.exists(fpath):
        logger.warning(f"type→category 映射不存在: {fpath}")
        return {}
    try:
        df = pd.read_parquet(fpath)
        return {str(t): set(str(c).split(";")) if c else set()
                for t, c in zip(df["type_codes"], df["categories"])}
    except Exception as e:
        logger.warning(f"type→category 映射加载失败 {fpath}: {e}")
        return {}


def decode(type_codes, type_map):
    """按 type 串解码 category（";" 连接字符串，未覆盖返回空串）"""
    key = norm_type_codes(type_codes)
    cats = type_map.get(key) if key else None
    return ";".join(sorted(cats)) if cats else ""


def apply_to_files(type_map, start=None, end=None):
    """把 type 映射解码结果回填到各日文件 category 列（仅更新空行）"""
    out_dir = get_data_path(ANNOUNCEMENTS_DIR)
    updated_days, skipped_days, filled, still_missing = 0, 0, 0, 0
    for fname in sorted(os.listdir(out_dir)):
        if not (fname.endswith(".parquet") and fname[:8].isdigit()
                and len(fname) == 16):
            continue
        ds = fname[:8]
        if start and ds < start:
            continue
        if end and ds > end:
            continue
        fpath = os.path.join(out_dir, fname)
        df = pd.read_parquet(fpath)
        if "type_codes" not in df.columns or "category" not in df.columns:
            continue
        empty_mask = df["category"].fillna("").eq("")
        if not empty_mask.any():
            skipped_days += 1
            continue
        filled_vals = [decode(t, type_map) for t in df.loc[empty_mask, "type_codes"]]
        df.loc[empty_mask, "category"] = filled_vals
        n_filled = int(sum(1 for v in filled_vals if v))
        filled += n_filled
        still_missing += len(filled_vals) - n_filled
        df.to_parquet(fpath, engine="pyarrow")
        updated_days += 1
        if n_filled > 0:
            logger.info(f"{ds}: type 解码回填 {n_filled}/{len(filled_vals)} 条")
    logger.info(f"type 解码回填结束: 更新 {updated_days} 个文件，跳过 {skipped_days} 个，"
                f"已回填 {filled} 条，仍未覆盖 {still_missing} 条")
    return filled, still_missing


def get_uncovered_rows(start=None, end=None):
    """扫描本地文件，返回 category 为空且映射未覆盖的行

    返回 [(date_str, ann_id, type_codes)]，供 API 定向补学使用。
    """
    out_dir = get_data_path(ANNOUNCEMENTS_DIR)
    type_map = load_map() or learn_from_files(start, end)
    rows = []
    for fname in sorted(os.listdir(out_dir)):
        if not (fname.endswith(".parquet") and fname[:8].isdigit()
                and len(fname) == 16):
            continue
        ds = fname[:8]
        if start and ds < start:
            continue
        if end and ds > end:
            continue
        fpath = os.path.join(out_dir, fname)
        try:
            df = pd.read_parquet(fpath, columns=["type_codes", "category", "ann_id"])
        except Exception:
            continue
        for t, c, aid in zip(df["type_codes"].astype(str),
                             df["category"].astype(str),
                             df["ann_id"].astype(str)):
            if (not c or c == "nan") and t and t != "nan":
                key = norm_type_codes(t)
                if key and key not in type_map:
                    rows.append((ds, aid, t))
    logger.info(f"未覆盖 type 串的行: {len(rows)} 条")
    return rows


def learn_from_api(type_map, max_dates=None, start=None, end=None,
                   save_interval=20, retry_wait=30):
    """对未覆盖 type 串出现的日期，定向调用巨潮 26 类查询补学

    对每个日期查询全部 26 类，从返回公告中学习 (announcementType → 类别)。
    每处理一个日期，若其未覆盖行全部学到，则从待学集合移除。

    限流容错（巨潮对高频请求会返回 403）：
    - 26 类低并发（4 线程）查询，日期间隔 0.5s，降低触发概率；
    - 单日期查询失败时等待 retry_wait 秒重试一次，仍失败则中断补学并返回
      （已学部分由 checkpoint/调用方保存，重新运行即可断点续跑）；
    - 每处理 save_interval 个日期 checkpoint 保存一次映射。
    返回更新后的 type_map 及本次新增的映射条数。
    """
    from .announcements import _paginate, ANNOUNCEMENT_CATEGORIES
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time

    rows = get_uncovered_rows(start, end)
    if not rows:
        logger.info("无未覆盖 type 串，无需补学")
        return type_map, 0

    # 按日期聚合未覆盖行，并按未覆盖行数降序处理（信息量大的日期优先）
    from collections import defaultdict
    by_date = defaultdict(list)
    for ds, aid, t in rows:
        by_date[ds].append((aid, t))
    dates = sorted(by_date.keys(), key=lambda d: -len(by_date[d]))

    pending = set(rows)  # (ds, aid, t) 未学到的行
    added = 0
    processed = 0

    def _learn_one_day(ds):
        """查询某日期全部 26 类并学习映射；成功返回 True，异常向上抛"""
        nonlocal added
        se = f"{ds[:4]}-{ds[4:6]}-{ds[6:]}~{ds[:4]}-{ds[4:6]}-{ds[6:]}"
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_paginate, se, category=code): (code, name)
                       for code, name in ANNOUNCEMENT_CATEGORIES.items()}
            for fut in as_completed(futures):
                code, name = futures[fut]
                items, _, _ = fut.result()
                for a in items:
                    aid_ = str(a.get("announcementId") or "")
                    t_ = str(a.get("announcementType") or "")
                    key = norm_type_codes(t_)
                    if not key:
                        continue
                    if key in type_map:
                        type_map[key].add(name)
                    else:
                        type_map[key] = {name}
                        added += 1
                    hit = (ds, aid_, t_)
                    if hit in pending:
                        pending.discard(hit)
        return True

    for ds in dates:
        if max_dates and processed >= max_dates:
            break
        try:
            _learn_one_day(ds)
        except Exception as e:
            logger.warning(f"{ds}: 26 类查询失败，{retry_wait}s 后重试: {e}")
            time.sleep(retry_wait)
            try:
                _learn_one_day(ds)
            except Exception as e2:
                logger.error(f"{ds}: 重试仍失败（疑似限流），中断补学，"
                             f"已学 {added} 条映射已 checkpoint: {e2}")
                break
        processed += 1
        remain = sum(1 for r in pending if r[0] == ds)
        logger.info(f"{ds}: 处理完成，本日期剩 {remain} 条未覆盖"
                    f"（累计新增映射 {added}）")
        if processed % save_interval == 0:
            save_map(type_map)  # checkpoint：中断时已学成果不丢
            logger.info(f"checkpoint: 已保存 {len(type_map)} 条映射")
        if not pending:
            logger.info("全部未覆盖行已学到")
            break
        time.sleep(0.5)  # 日期间隔，降低限流概率
    logger.info(f"API 补学结束: 处理 {processed} 个日期，新增映射 {added} 条，"
                f"仍剩 {len(pending)} 行未覆盖")
    return type_map, added


def ensure_map(learn_api=False):
    """确保 type→category 静态映射存在，返回加载后的 type_map

    映射文件缺失时自动学习：先从本地已回填行学习（覆盖 99%+），
    若指定 learn_api 再对未覆盖 type 串调用巨潮 API 补学。
    学习结果落盘，供每日更新与历史回填复用。
    """
    type_map = load_map()
    if type_map:
        return type_map
    logger.info("映射文件缺失，开始学习 type→category 映射")
    type_map = learn_from_files()
    if learn_api:
        type_map, added = learn_from_api(type_map)
        logger.info(f"API 补学新增映射 {added} 条")
    save_map(type_map)
    return type_map


def decode_apply_and_learn(start=None, end=None, learn_api=False):
    """按 type_codes 解码回填 category 列，映射未覆盖的自动补学后再回填

    流程：
      1. 确保静态映射存在（缺失则自动学习）；
      2. 对 [start, end] 范围内文件回填空行 category；
      3. 若 learn_api 且仍有未覆盖行，调用巨潮 26 类查询补学、
         更新映射后二次回填（仅补本范围，控制请求量）。
    返回 (累计已回填条数, 仍未覆盖条数)。
    """
    type_map = ensure_map(learn_api=learn_api)
    if not type_map:
        logger.warning("无可用 type→category 映射，跳过 category 回填")
        return 0, 0
    filled, still_missing = apply_to_files(type_map, start, end)
    if learn_api and still_missing:
        type_map2, added = learn_from_api(type_map, start=start, end=end)
        if added:
            save_map(type_map2)
            filled2, still_missing2 = apply_to_files(type_map2, start, end)
            logger.info(f"补学后二次回填: 新增 {filled2} 条，"
                        f"仍缺 {still_missing2} 条")
            return filled + filled2, still_missing2
    return filled, still_missing


def main():
    """学习并回填：python -m download.announcements.type_category_map"""
    import argparse
    parser = argparse.ArgumentParser(description="type_codes → 官方26类 category 静态映射")
    parser.add_argument("--learn-api", action="store_true",
                        help="对未覆盖 type 串调用巨潮 API 补学（耗时较长）")
    parser.add_argument("--learn-api-dates", type=int, default=0,
                        help="API 补学的日期数上限（默认全部）")
    parser.add_argument("--start", default=None, help="起始日期 YYYYMMDD")
    parser.add_argument("--end", default=None, help="结束日期 YYYYMMDD")
    parser.add_argument("--no-backfill", action="store_true", help="只学映射，不回填")
    args = parser.parse_args()

    ensure_dir(get_data_path(ANNOUNCEMENTS_DIR))
    # 1) 从已回填行学习（覆盖历史绝大多数）
    type_map = learn_from_files(args.start, args.end)
    save_map(type_map)

    # 2) 可选：API 定向补学未覆盖 type 串
    if args.learn_api:
        type_map, added = learn_from_api(
            type_map, max_dates=args.learn_api_dates or None)
        if added:
            save_map(type_map)

    # 3) 回填 category 列
    if not args.no_backfill:
        apply_to_files(type_map, args.start, args.end)


if __name__ == "__main__":
    main()
