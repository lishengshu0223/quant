"""A股公告特征数据下载（数据源：巨潮资讯网 cninfo.com.cn）

巨潮资讯网是证监会指定的上市公司信息披露平台，覆盖沪深 A 股：
- 仅抓取公告特征信息（标题、链接、类别、代码等），不下载公告正文
- 公告 PDF 链接为 static.cninfo.com.cn 静态地址，历史链接长期有效
- 公告类别采用巨潮官方 26 类分类，比米筐单一分类更细致
- 历史数据可追溯到 2005 年以前，满足 2016 年起的需求
- 剔除 B 股与北交所，仅保留沪深 A 股

存储：F:\\Trade_data\\announcements\\YYYYMMDD.parquet，每个自然日一个文件
（公告全年均可发布，故按自然日而非交易日存储；无公告的空日也会落空文件占位）

字段说明：
    date         公告发布日（文件对应日期）
    order_book_id  股票代码（6→.XSHG，0/3→.XSHE，剔除 B 股和北交所）
    sec_code     6 位数字代码
    sec_name     股票简称
    title        公告标题
    category     公告类别（巨潮官方 26 类，可能属于多类，用 ";" 连接）
    type_codes   巨潮原始分类代码串（如 "01011101||010112||..."）
    url          公告 PDF 链接
    ann_id       巨潮公告唯一编号
    ann_time     公告发布时间（北京时间，精确到秒）

    以下分类列由 announcement_type_map 静态查表得到（主码 = 剔除市场辅助码后
    最具体的业务码，详见该模块 docstring）：
    cninfo_type_code  巨潮主码（原始最细分类代码）
    cninfo_l1_code/_name ~ cninfo_l4_code/_name  巨潮一至四级分类（层级切分，无该级则空）
    cninfo_leaf_name  巨潮最细分类中文名
    csrc_l1_code/_name  证监会 JR/T 0021.1—2023 一级分类（7 类，全覆盖）
    csrc_l2_code/_name  证监会二级分类（51 类，仅与巨潮码明确对应时填充）

用法：
    # 补全 2016 年至今的历史数据（自动跳过已下载日期，可中断后重跑续传）
    python -m download.announcements --start 2016-01-01

    # 每日更新最近 3 天（强制覆盖，捕获盘后延迟发布的公告）
    python -m download.announcements --recent 3
"""
import datetime
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from .announcement_type_map import CLASSIFY_COLUMNS, CN_CODE_NAMES, classify
from ..config import get_data_path, ANNOUNCEMENTS_DIR, START_DATE, ensure_dir
from ..logger import logger
from ..utils import get_existing_dates

# 巨潮资讯网公告查询接口
CNINFO_QUERY_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_STATIC_PREFIX = "http://static.cninfo.com.cn/"
PLATE_ALL = "sz;sh"  # 沪市 + 深市（剔除北交所，B 股在代码层面二次过滤）
PLATES = ["sz", "sh"]  # 板块拆分子板（用于绕开单次查询 3000 条上限）
PAGE_SIZE = 30  # 服务端每页上限固定为 30，传更大值无效
MAX_PAGES = 1000  # 最大翻页数保护
STALL_PAGES = 3  # 连续 N 页无新增内容即视为翻到重复区（3000 条上限）
REQUEST_INTERVAL = 0.2  # 请求间隔（秒），避免对巨潮服务器造成压力
MAX_RETRIES = 3
CATEGORY_BATCH_DAYS = 7  # 旧版逐批分类查询的分批天数（当前默认跳过分类查询，仅缓存模式复用）

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "X-Requested-With": "XMLHttpRequest",
}

# 公告大类（26 类，来源：cninfo.com.cn 官方公告查询页分类）
ANNOUNCEMENT_CATEGORIES = {
    "category_ndbg_szsh": "年报",
    "category_bndbg_szsh": "半年报",
    "category_yjdbg_szsh": "一季报",
    "category_sjdbg_szsh": "三季报",
    "category_yjygjxz_szsh": "业绩预告",
    "category_qyfpxzcs_szsh": "权益分派",
    "category_dshgg_szsh": "董事会",
    "category_jshgg_szsh": "监事会",
    "category_gddh_szsh": "股东会",
    "category_rcjy_szsh": "日常经营",
    "category_gszl_szsh": "公司治理",
    "category_zj_szsh": "中介报告",
    "category_sf_szsh": "首发",
    "category_zf_szsh": "增发",
    "category_gqjl_szsh": "股权激励",
    "category_pg_szsh": "配股",
    "category_jj_szsh": "解禁",
    "category_gszq_szsh": "公司债",
    "category_kzzq_szsh": "可转债",
    "category_qtrz_szsh": "其他融资",
    "category_gqbd_szsh": "股权变动",
    "category_bcgz_szsh": "补充更正",
    "category_cqdq_szsh": "澄清致歉",
    "category_fxts_szsh": "风险提示",
    "category_tbclts_szsh": "特别处理和退市",
    "category_tszlq_szsh": "退市整理期",
}

COLUMNS = ["date", "order_book_id", "sec_code", "sec_name", "title",
           "category", "type_codes", "url", "ann_id", "ann_time"] \
    + CLASSIFY_COLUMNS

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _empty_df():
    """构造带完整 schema 的空 DataFrame（空日占位用）"""
    df = pd.DataFrame(columns=COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    df["ann_time"] = pd.to_datetime(df["ann_time"])
    for col in ["order_book_id", "sec_code", "sec_name", "title",
                "category", "type_codes", "url", "ann_id"] + CLASSIFY_COLUMNS:
        df[col] = df[col].astype(str)
    return df


def _to_order_book_id(sec_code):
    """6 位代码 → order_book_id；仅保留沪深 A 股，其余（B 股/北交所/债券/基金等）返回 None

    用 A 股代码段白名单判断，避免债券（深市 1 开头）、基金（沪市 5 开头）等混入。
    """
    if not sec_code:
        return None
    # 深市 A 股：000/001/002/003 主板，300/301/302 创业板
    if sec_code.startswith(("000", "001", "002", "003", "300", "301", "302")):
        return sec_code + ".XSHE"
    # 沪市 A 股：600/601/603/605 主板，688/689 科创板
    if sec_code.startswith(("600", "601", "603", "605", "688", "689")):
        return sec_code + ".XSHG"
    return None


def _post_query(se_date, page_num=1, category="", plate=None):
    """单次查询巨潮公告接口（带重试）"""
    data = {
        "pageNum": page_num,
        "pageSize": PAGE_SIZE,
        "column": "szse",
        "tabName": "fulltext",
        "plate": plate or PLATE_ALL,
        "stock": "",
        "searchkey": "",
        "secid": "",
        "category": category,
        "trade": "",
        "seDate": se_date,
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            # verify=False：巨潮证书链偶发验证失败（SSL: CERTIFICATE_VERIFY_FAILED），
            # 关闭证书校验（数据为公开公告，无中间人风险）
            # proxies 显式置空：巨潮为国内公网站点，直连即可，不走系统代理
            # （本机 127.0.0.1:8080 端口为其他程序占用，若代理未监听会导致
            #   ProxyError 连接被拒，公告下载全量失败，故必须绕过代理）
            resp = requests.post(CNINFO_QUERY_URL, data=data,
                                 headers=HEADERS, timeout=30, verify=False,
                                 proxies={"http": None, "https": None})
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            wait = 2 ** attempt * 2
            logger.warning(
                f"巨潮查询失败({se_date}, page={page_num})，{wait}s 后第 "
                f"{attempt + 1} 次重试: {e}")
            time.sleep(wait)
    raise RuntimeError(f"巨潮查询重试后仍失败: {se_date} page={page_num}: {last_err}")


def _paginate(se_date, category="", plate=None):
    """翻页抓取指定日期范围的全部公告原始记录，返回 (items, total, truncated)

    巨潮接口单次查询最多返回 100 页 × 30 条 = 3000 条；超过后从第 101 页起
    循环返回前 100 页的内容（page 越界但不报错）。因此翻页时以"连续
    STALL_PAGES 页无新增内容"判定已触及 3000 条上限，提前停止并返回
    truncated=True（调用方应缩小时间窗口或按板块分片重查）。
    """
    items = []
    seen = set()
    total = None
    page = 1
    stall = 0  # 连续无新增页计数
    stalled = False  # 是否因触及 3000 条上限（重复页）中断
    while page <= MAX_PAGES:
        j = _post_query(se_date, page_num=page, category=category, plate=plate)
        anns = j.get("announcements") or []
        if total is None:
            total = j.get("totalAnnouncement") or 0
        if not anns:
            break
        added = 0
        for a in anns:
            key = (str(a.get("announcementId") or ""), a.get("secCode"))
            if key[0] and key not in seen:
                seen.add(key)
                items.append(a)
                added += 1
        if added == 0:
            stall += 1
            if stall >= STALL_PAGES:
                stalled = True  # 连续多页全为重复 → 已到 3000 条上限
                break
        else:
            stall = 0
        # totalpages 字段不可靠（可能小于实际页数），以空页/重复页或抓满 total 为准
        if total and len(seen) >= total:
            break
        page += 1
        time.sleep(REQUEST_INTERVAL)
    # 截断判定以"重复页中断"为准（total 可能虚高，不可靠）
    truncated = stalled
    return items, total, truncated


def _paginate_split(se_date, category=""):
    """按时间+板块分片翻页，确保抓取指定日期范围的全部公告（绕开 3000 条上限）

    巨潮接口单次查询最多返回 3000 条，超过则循环返回重复内容。本函数逐日、
    逐板块查询并合并去重，任何子片内仍触及 3000 条上限时记录警告但不中断。
    """
    start, end = se_date.split("~")
    d0 = datetime.date.fromisoformat(start)
    d1 = datetime.date.fromisoformat(end)
    day = d0
    all_items = []
    seen = set()
    while day <= d1:
        ds = f"{day:%Y-%m-%d}~{day:%Y-%m-%d}"
        for plate in PLATES:
            items, total, truncated = _paginate(ds, category=category, plate=plate)
            for a in items:
                key = (str(a.get("announcementId") or ""), a.get("secCode"))
                if key[0] and key not in seen:
                    seen.add(key)
                    all_items.append(a)
            if truncated:
                logger.warning(
                    f"{ds} 类别[{category}] 板块[{plate}] 仍超 3000 条上限"
                    f"（total={total}，已取 {len(items)}），可能缺失部分公告")
        day += datetime.timedelta(days=1)
    return all_items


def _fetch_category_map(se_date):
    """查询日期范围内公告的分类映射 {announcementId: set(类别名)}"""
    cat_map = {}
    for code, name in ANNOUNCEMENT_CATEGORIES.items():
        items, _, _ = _paginate(se_date, category=code)
        for a in items:
            aid = str(a.get("announcementId") or "")
            if aid:
                cat_map.setdefault(aid, set()).add(name)
    return cat_map


def _load_category_cache(cache_path):
    """加载本地 category 缓存 parquet → {annId: set(类别名)}；不存在或异常返回空 dict

    缓存文件由 generate_category_cache.py 按月生成，列为 ann_id + categories
    （categories 为 ";" 连接的类别名）。
    """
    if not cache_path or not os.path.exists(cache_path):
        return {}
    try:
        df = pd.read_parquet(cache_path)
        if "ann_id" not in df.columns or "categories" not in df.columns:
            logger.warning(f"category 缓存缺少 ann_id/categories 列: {cache_path}")
            return {}
        cat_map = {}
        for aid, c in zip(df["ann_id"], df["categories"]):
            aid = str(aid)
            cat_map[aid] = set(str(c).split(";")) if c and str(c) else set()
        return cat_map
    except Exception as e:
        logger.warning(f"category 缓存加载失败 {cache_path}: {e}")
        return {}


def _build_daily_df(date_str, items, cat_map):
    """原始记录 → 日度 DataFrame"""
    date = pd.Timestamp(date_str)
    rows = []
    for a in items:
        sec_code = str(a.get("secCode") or "").strip()
        order_book_id = _to_order_book_id(sec_code)
        if order_book_id is None:
            continue
        ann_id = str(a.get("announcementId") or "")
        cats = cat_map.get(ann_id)
        ts = a.get("announcementTime")
        ann_time = (pd.to_datetime(ts, unit="ms", utc=True)
                    .tz_convert("Asia/Shanghai").tz_localize(None)
                    if ts else pd.NaT)
        type_codes = str(a.get("announcementType") or "")
        row = {
            "date": date,
            "order_book_id": order_book_id,
            "sec_code": sec_code,
            "sec_name": str(a.get("secName") or "").strip(),
            "title": _HTML_TAG_RE.sub("", a.get("announcementTitle") or "").strip(),
            "category": ";".join(sorted(cats)) if cats else "",
            "type_codes": type_codes,
            "url": CNINFO_STATIC_PREFIX + str(a.get("adjunctUrl") or ""),
            "ann_id": ann_id,
            "ann_time": ann_time,
        }
        row.update(classify(type_codes))  # 巨潮层级 + 证监会分类（静态查表）
        rows.append(row)
    if not rows:
        return _empty_df()
    # 提示未破译的巨潮细码（层级仍按字符串切分回退，不影响入库）
    unknown = sorted({r["cninfo_type_code"] for r in rows
                      if r["cninfo_type_code"] and r["cninfo_type_code"] not in CN_CODE_NAMES})
    if unknown:
        logger.warning("出现未收录的巨潮细码（请补充 announcement_type_map）：{}",
                       ",".join(unknown))
    return pd.DataFrame(rows, columns=COLUMNS)


def _download_one_day(day, base_dir, cat_map):
    """下载并保存单个自然日的公告数据，返回 (date_str, 条数)

    巨潮接口单次查询最多 3000 条，超限日期（如 4 月末年报披露高峰）会自动
    按板块分片抓取合并，确保不截断。
    """
    ds = day.strftime("%Y%m%d")
    se_date = f"{day:%Y-%m-%d}~{day:%Y-%m-%d}"
    items, total, truncated = _paginate(se_date)
    if truncated:
        logger.warning(f"{ds} 全量查询超 3000 条（total={total}），按板块分片重抓")
        items = _paginate_split(se_date)
        total = len(items)
    df = _build_daily_df(ds, items, cat_map)
    df.to_parquet(os.path.join(base_dir, f"{ds}.parquet"), engine="pyarrow")
    logger.info(f"公告下载完成 {ds}: {len(df)} 条（当日全量 {total}）")
    return ds


def download_announcements(start_date=None, end_date=None, force=False,
                           workers=3, category_cache_path=None):
    """下载日期范围内的 A 股公告特征数据（按自然日逐日存储）

    - 默认从 START_DATE(2016-01-01) 下载到今天
    - 自动跳过已存在的日期文件（force=True 时强制覆盖），支持中断续传
    - category 列（巨潮官方 26 类）默认留空：逐批实时查询巨潮分类接口会因
      查询总量过大触发限流（50~70 小时），故改为先纯下载、后用本地缓存回填。
      提供 category_cache_path 且文件存在时，直接从缓存读取 category 列。
    - 缓存由 download/announcements/generate_category_cache.py 按月生成，由
      download/announcements/apply_category_cache.py 回填到已下载文件
    - workers: 每批内并发下载的天数（首次全量回填建议 3~5，勿过大以免被限流）
    """
    start = pd.Timestamp(start_date or START_DATE).date()
    end = pd.Timestamp(end_date or datetime.date.today()).date()
    if start > end:
        logger.warning(f"download_announcements: 起始日期 {start} 晚于结束日期 {end}")
        return []

    # 加载本地 category 缓存（可选）：{annId: "类别1;类别2"}
    cat_map = _load_category_cache(category_cache_path) if category_cache_path else {}
    if category_cache_path and not cat_map:
        logger.warning(f"category 缓存不存在或为空: {category_cache_path}，"
                       f"本次下载 category 列将留空（后续用 apply_category_cache 回填）")

    base_dir = ensure_dir(get_data_path(ANNOUNCEMENTS_DIR))
    existing = set(get_existing_dates(base_dir))
    all_dates = [start + datetime.timedelta(days=i)
                 for i in range((end - start).days + 1)]

    logger.info(f"开始下载公告数据: {start} ~ {end}，共 {len(all_dates)} 天"
                f"（category 缓存 {'已加载 %d 条' % len(cat_map) if cat_map else '未加载'}）")
    success_dates, failed_dates = [], []

    for chunk_start in range(0, len(all_dates), CATEGORY_BATCH_DAYS):
        chunk = all_dates[chunk_start:chunk_start + CATEGORY_BATCH_DAYS]
        todo = [d for d in chunk
                if force or d.strftime("%Y%m%d") not in existing]
        if not todo:
            continue

        if workers <= 1 or len(todo) == 1:
            for d in todo:
                ds = d.strftime("%Y%m%d")
                try:
                    success_dates.append(_download_one_day(d, base_dir, cat_map))
                except Exception as e:
                    logger.error(f"公告下载失败 {ds}: {e}")
                    failed_dates.append(ds)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_download_one_day, d, base_dir, cat_map): d
                           for d in todo}
                for fut in as_completed(futures):
                    d = futures[fut]
                    ds = d.strftime("%Y%m%d")
                    try:
                        success_dates.append(fut.result())
                    except Exception as e:
                        logger.error(f"公告下载失败 {ds}: {e}")
                        failed_dates.append(ds)
        time.sleep(REQUEST_INTERVAL * 5)

    logger.info(f"download_announcements 结束: 成功 {len(success_dates)} 天, "
                f"失败 {len(failed_dates)} 天")
    if failed_dates:
        logger.warning(f"失败日期: {failed_dates}")
    return success_dates


def download_recent_announcements(days=3):
    """每日更新：强制重新下载最近 N 个自然日（捕获盘后延迟发布的公告）"""
    end = datetime.date.today()
    start = end - datetime.timedelta(days=days - 1)
    return download_announcements(start_date=start, end_date=end, force=True)
