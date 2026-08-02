"""
高管/大股东增持事件分析系统 - 全量预加载+纯LLM多线程版
1. 预加载所有米筐数据到内存（公告、价格、股份数）——单线程，一次性
2. 两级并行处理（纯LLM多线程，零rqdatac调用）
   第一级（10线程）：跨日期并行 - 切片公告→关键词筛选→LLM筛选→分组
   第二级（10线程）：跨日期全局并行 - PDF下载→LLM分析→内存切片计算→保存
"""
import os
import sys
import time
import json
import threading
import queue
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, List

from loguru import logger

import rqdatac
rqdatac.init()

# 本地API（无连接数限制）
sys.path.insert(0, r'f:\quant')
from local_api import init as _local_init, get_price as _local_get_price, get_trading_dates as _local_get_trading_dates
_local_init()

from prompt import FILTER_ANNOUNCEMENT_PROMPT, STRUCTURED_ANALYSIS_PROMPT
from func_tools import (
    filter_increase_announcements,
    analyze_announcement_structured,
    download_pdf,
    extract_text_from_pdf,
    keyword_filter_announcements,
    calculate_target_amount,
    get_specific_period_returns,
    save_increase_event,
)

# ============================================================================
# 配置
# ============================================================================
SEARCH_START_DATE = "2023-01-01"
SEARCH_END_DATE = "2026-07-29"
SAVE_PATH = r"F:\quant\research\holding_increase\increase_events"
PROCESSED_DATES_FILE = r"F:\quant\research\holding_increase\processed_dates.json"
NUM_DATE_WORKERS = 10   # 日期预处理线程（LLM筛选）
NUM_EVENT_WORKERS = 10  # 公司事件处理线程（LLM分析）

os.makedirs(SAVE_PATH, exist_ok=True)

# ============================================================================
# 全局预加载数据
# ============================================================================
ALL_ANNOUNCEMENTS: Optional[pd.DataFrame] = None   # 全市场公告
PRICE_CACHE: dict = {}       # {stock_code: Series(date -> close)}
PRICE_POST_CACHE: dict = {}  # {stock_code: Series(date -> post_open)}
SHARES_CACHE: dict = {}      # {stock_code: Series(date -> total_shares)}
TRADING_DATES_LIST: list = []  # 交易日列表
CODE_LIST: list = []           # 股票代码列表

# 线程锁
file_lock = threading.Lock()
counter_lock = threading.Lock()
processed_dates_lock = threading.Lock()

# 计数器
total_events_found = 0
total_events_processed = 0
total_events_failed = 0
processed_dates: set = set()


# ============================================================================
# 预加载函数
# ============================================================================
def preload_all_data():
    """预加载所有米筐数据到内存"""
    global ALL_ANNOUNCEMENTS, TRADING_DATES_LIST, CODE_LIST

    logger.info("=" * 60)
    logger.info("阶段1: 全量数据预加载")
    logger.info("=" * 60)

    # 1. 股票代码
    logger.info("[1/5] 获取全市场A股股票代码...")
    instruments = rqdatac.all_instruments(type='CS', date=SEARCH_END_DATE)
    CODE_LIST = instruments['order_book_id'].tolist()
    logger.info(f"  {len(CODE_LIST)} 只A股股票")

    # 2. 交易日列表
    logger.info("[2/5] 获取交易日列表...")
    TRADING_DATES_LIST = _local_get_trading_dates(start_date=SEARCH_START_DATE, end_date=SEARCH_END_DATE)
    logger.info(f"  {len(TRADING_DATES_LIST)} 个交易日")

    # 3. 全市场价格数据（本地API，不复权+后复权）
    logger.info("[3/5] 获取全市场价格数据（本地API）...")
    t0 = time.time()
    price_df = _local_get_price(CODE_LIST, start_date=SEARCH_START_DATE, end_date=SEARCH_END_DATE, fields=['open', 'close'])
    logger.info(f"  不复权价格: {len(price_df)} 行, 耗时{time.time()-t0:.1f}s")

    t0 = time.time()
    price_post_df = _local_get_price(CODE_LIST, start_date=SEARCH_START_DATE, end_date=SEARCH_END_DATE, fields=['open'], adjust_type='post')
    logger.info(f"  后复权价格: {len(price_post_df)} 行, 耗时{time.time()-t0:.1f}s")

    # 按股票代码构建缓存（每只股票一个Series，索引为date）
    logger.info("  构建价格缓存...")
    if not price_df.empty:
        price_df = price_df.reset_index()
        for code, group in price_df.groupby('code'):
            PRICE_CACHE[code] = group.set_index('date')['close'].sort_index()
    if not price_post_df.empty:
        price_post_df = price_post_df.reset_index()
        for code, group in price_post_df.groupby('code'):
            PRICE_POST_CACHE[code] = group.set_index('date')['open'].sort_index()
    logger.info(f"  价格缓存: {len(PRICE_CACHE)} 只股票, 后复权缓存: {len(PRICE_POST_CACHE)} 只股票")

    # 4. 全市场股份数（rqdatac，分批获取）
    logger.info("[4/5] 获取全市场股份数据（rqdatac）...")
    batch_size = 200
    batches = [CODE_LIST[i:i+batch_size] for i in range(0, len(CODE_LIST), batch_size)]
    shares_list = []
    for idx, batch in enumerate(batches):
        try:
            shares = rqdatac.get_shares(batch, start_date=SEARCH_START_DATE, end_date=SEARCH_END_DATE)
            if shares is not None and not shares.empty:
                shares_list.append(shares)
        except Exception as e:
            logger.warning(f"  股份数批次 {idx+1}/{len(batches)} 失败: {e}")
            time.sleep(2)
            # 重试
            try:
                shares = rqdatac.get_shares(batch, start_date=SEARCH_START_DATE, end_date=SEARCH_END_DATE)
                if shares is not None and not shares.empty:
                    shares_list.append(shares)
            except Exception as e2:
                logger.error(f"  股份数批次 {idx+1} 重试失败: {e2}")
        if (idx+1) % 5 == 0:
            logger.info(f"  股份数进度: {idx+1}/{len(batches)} 批")

    # 处理股份数据格式并构建缓存
    # get_shares 返回: MultiIndex(order_book_id, date), 列 ['total', ...]
    if shares_list:
        all_shares = pd.concat(shares_list)
        all_shares = all_shares.reset_index()
        for code, group in all_shares.groupby('order_book_id'):
            SHARES_CACHE[code] = group.set_index('date')['total'].sort_index()

    logger.info(f"  股份缓存: {len(SHARES_CACHE)} 只股票")

    # 5. 全市场公告（按季度+分批获取）
    logger.info("[5/5] 获取全市场公告数据（按季度分批）...")
    quarters_start = pd.date_range(SEARCH_START_DATE, SEARCH_END_DATE, freq='QS')
    all_ann_list = []
    for q_idx, q_start in enumerate(quarters_start):
        q_end = (q_start + pd.DateOffset(months=3) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        q_start_str = q_start.strftime('%Y-%m-%d')

        ann_batches = [CODE_LIST[i:i+500] for i in range(0, len(CODE_LIST), 500)]
        for batch_idx, batch in enumerate(ann_batches):
            try:
                ann = rqdatac.get_announcement(order_book_ids=batch, start_date=q_start_str, end_date=q_end)
                if ann is not None and not ann.empty:
                    all_ann_list.append(ann.reset_index())
            except Exception as e:
                logger.warning(f"  公告获取失败 {q_start_str}~{q_end} 批次{batch_idx+1}: {e}")
                time.sleep(2)

        logger.info(f"  公告进度: {q_idx+1}/{len(quarters_start)} 季度 ({q_start_str}~{q_end})")

    if all_ann_list:
        ALL_ANNOUNCEMENTS = pd.concat(all_ann_list, ignore_index=True)
    else:
        ALL_ANNOUNCEMENTS = pd.DataFrame()
    logger.info(f"  公告数据: {len(ALL_ANNOUNCEMENTS)} 条")

    logger.info("\n✅ 全量数据预加载完成！")
    mem_mb = (sum(s.memory_usage() for s in PRICE_CACHE.values()) +
              sum(s.memory_usage() for s in PRICE_POST_CACHE.values()) +
              sum(s.memory_usage() for s in SHARES_CACHE.values())) / 1e6
    logger.info(f"  缓存内存估算: {mem_mb:.0f}MB")


# ============================================================================
# 内存切片函数（零rqdatac调用，线程安全）
# ============================================================================
def get_close_price_mem(stock_code, date_str):
    """从内存获取收盘价"""
    try:
        s = PRICE_CACHE.get(stock_code)
        if s is None:
            return None
        dt = pd.Timestamp(date_str)
        if dt in s.index:
            val = s.loc[dt]
            return float(val) if pd.notna(val) else None
        return None
    except:
        return None


def get_post_open_series_mem(stock_code, start_date, end_date):
    """从内存获取后复权开盘价序列"""
    try:
        s = PRICE_POST_CACHE.get(stock_code)
        if s is None:
            return None
        mask = (s.index >= pd.Timestamp(start_date)) & (s.index <= pd.Timestamp(end_date))
        result = s[mask]
        return result if not result.empty else None
    except:
        return None


def get_total_shares_mem(stock_code, date_str):
    """从内存获取总股本"""
    try:
        s = SHARES_CACHE.get(stock_code)
        if s is None:
            return None
        dt = pd.Timestamp(date_str)
        if dt in s.index:
            val = s.loc[dt]
            return float(val) if pd.notna(val) else None
        # 如果精确日期没有，找最近的
        before = s[s.index <= dt]
        if not before.empty:
            return float(before.iloc[-1])
        return None
    except:
        return None


def get_next_trading_date_mem(date_str, n=1):
    """从内存获取第N个下一交易日"""
    try:
        dt = pd.Timestamp(date_str).date()
        dates = pd.Series(TRADING_DATES_LIST)
        future = dates[dates > dt]
        if len(future) >= n:
            return future.iloc[n - 1]
        return None
    except:
        return None


def get_announcements_for_date_mem(date_str):
    """从内存切片获取某天的公告"""
    if ALL_ANNOUNCEMENTS.empty:
        return pd.DataFrame()
    try:
        # 公告日期列名为 info_date
        date_col = 'info_date'
        if date_col not in ALL_ANNOUNCEMENTS.columns:
            return pd.DataFrame()

        dt = pd.Timestamp(date_str)
        # info_date 可能是字符串或datetime，统一比较
        ann_dates = pd.to_datetime(ALL_ANNOUNCEMENTS[date_col])
        mask = ann_dates.dt.date == dt.date()
        return ALL_ANNOUNCEMENTS[mask].copy()
    except:
        return pd.DataFrame()


def calculate_return_series_mem(stock_code, info_date_str, hold_days=120):
    """从内存计算收益率序列"""
    try:
        # 获取下一交易日
        next_trade_date = get_next_trading_date_mem(info_date_str, n=1)
        if next_trade_date is None:
            return None
        next_trade_date_str = pd.Timestamp(next_trade_date).strftime('%Y-%m-%d')

        # 获取hold_days个交易日后的日期
        end_date = get_next_trading_date_mem(next_trade_date_str, n=hold_days)
        if end_date is None:
            return None
        end_date_str = pd.Timestamp(end_date).strftime('%Y-%m-%d')

        # 从内存获取后复权开盘价序列
        price_series = get_post_open_series_mem(stock_code, next_trade_date_str, end_date_str)
        if price_series is None or len(price_series) < 2:
            return None

        # 基准价：T+1开盘价
        base_price = price_series.iloc[0]
        if base_price is None or base_price <= 0:
            return None

        # 计算收益率序列
        actual_days = min(hold_days, len(price_series) - 1)
        returns = []
        for i in range(1, actual_days + 1):
            day_price = price_series.iloc[i]
            if day_price is None or day_price <= 0:
                returns.append(None)
            else:
                ret = (day_price / base_price) - 1.0
                returns.append(round(ret, 4))

        while len(returns) < hold_days:
            returns.append(None)

        return returns
    except Exception as e:
        logger.error(f"计算收益率序列失败 {stock_code}: {e}")
        return None


def _is_valid_value(v):
    if v is None or v == '' or v == 0 or v == '0':
        return False
    try:
        return float(v) > 0
    except (ValueError, TypeError):
        return False


# ============================================================================
# 已处理日期管理
# ============================================================================
def load_processed_dates():
    global processed_dates
    if os.path.exists(PROCESSED_DATES_FILE):
        try:
            with open(PROCESSED_DATES_FILE, 'r', encoding='utf-8') as f:
                processed_dates = set(json.load(f))
        except:
            processed_dates = set()

    # 从increase_events目录恢复已处理日期
    if os.path.exists(SAVE_PATH):
        for f in os.listdir(SAVE_PATH):
            if f.endswith('.json'):
                date_part = f[:8]
                if len(date_part) == 8:
                    date_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                    processed_dates.add(date_str)


def save_processed_dates():
    with processed_dates_lock:
        dates_copy = sorted(list(processed_dates))
    with open(PROCESSED_DATES_FILE, 'w', encoding='utf-8') as f:
        json.dump(dates_copy, f, ensure_ascii=False, indent=2)


# ============================================================================
# 事件任务
# ============================================================================
@dataclass
class EventTask:
    date_str: str
    stock_code: str
    stock_ann: pd.DataFrame


# ============================================================================
# 第一级：日期预处理线程（LLM筛选）
# ============================================================================
def process_date_pipeline(date_str: str, event_queue: queue.Queue):
    """单个日期的处理流水线：切片公告→关键词筛选→LLM筛选→分组→产出事件"""
    tid = threading.current_thread().ident
    try:
        # 检查是否已处理
        with processed_dates_lock:
            if date_str in processed_dates:
                return (date_str, "skipped", 0)

        # 切片当天公告
        day_announcements = get_announcements_for_date_mem(date_str)
        if day_announcements.empty:
            with processed_dates_lock:
                processed_dates.add(date_str)
            save_processed_dates()
            return (date_str, "no_announcements", 0)

        # 关键词预筛选
        if 'title' not in day_announcements.columns:
            with processed_dates_lock:
                processed_dates.add(date_str)
            save_processed_dates()
            return (date_str, "no_title", 0)

        pre_filtered = keyword_filter_announcements(day_announcements)
        if pre_filtered.empty:
            with processed_dates_lock:
                processed_dates.add(date_str)
            save_processed_dates()
            return (date_str, "no_keyword_match", 0)

        # 排除已处理的公司
        date_str_compact = date_str.replace('-', '')
        existing_files = [f for f in os.listdir(SAVE_PATH)
                         if f.startswith(date_str_compact) and f.endswith('.json')] if os.path.exists(SAVE_PATH) else []
        processed_stocks = set()
        for f in existing_files:
            parts = f.replace('.json', '').split('_')
            if len(parts) == 2:
                processed_stocks.add(parts[1])

        if processed_stocks:
            if 'order_book_id' in pre_filtered.columns:
                stock_codes_6 = pre_filtered['order_book_id'].str[:6]
                pre_filtered = pre_filtered[~stock_codes_6.isin(processed_stocks)].copy()
            if pre_filtered.empty:
                with processed_dates_lock:
                    processed_dates.add(date_str)
                save_processed_dates()
                return (date_str, "all_done", 0)

        logger.info(f"  [日期线程{tid}] {date_str}: 初筛{len(pre_filtered)}条，LLM筛选...")

        # LLM精确筛选
        filtered_ann = filter_increase_announcements(pre_filtered, FILTER_ANNOUNCEMENT_PROMPT)
        if filtered_ann is None or filtered_ann.empty:
            with processed_dates_lock:
                processed_dates.add(date_str)
            save_processed_dates()
            return (date_str, "llm_no_match", 0)

        # 按公司分组
        if 'order_book_id' in filtered_ann.columns:
            stock_groups = filtered_ann.groupby('order_book_id')
        elif 'media' in filtered_ann.columns:
            stock_groups = filtered_ann.groupby('media')
        else:
            with processed_dates_lock:
                processed_dates.add(date_str)
            save_processed_dates()
            return (date_str, "no_stock_col", 0)

        num_companies = len(stock_groups)
        logger.info(f"  [日期线程{tid}] {date_str}: LLM筛出{num_companies}家公司，提交事件队列")

        discovered = 0
        for stock_code, stock_ann in stock_groups:
            event_queue.put(EventTask(
                date_str=date_str,
                stock_code=stock_code,
                stock_ann=stock_ann.copy(),
            ))
            discovered += 1

        with counter_lock:
            global total_events_found
            total_events_found += discovered

        with processed_dates_lock:
            processed_dates.add(date_str)
        save_processed_dates()

        return (date_str, f"ok_{discovered}", discovered)

    except Exception as e:
        logger.error(f"  [日期线程{tid}] {date_str}: 预处理异常: {e}")
        return (date_str, f"error_{type(e).__name__}", 0)


# ============================================================================
# 第二级：公司事件处理线程（LLM分析+内存计算+保存）
# ============================================================================
def event_worker(worker_id: int, event_queue: queue.Queue, stop_event: threading.Event):
    """公司事件处理消费者：PDF下载→LLM分析→内存切片计算→保存"""
    while not stop_event.is_set() or not event_queue.empty():
        try:
            task = event_queue.get(timeout=2)
        except queue.Empty:
            continue

        if task is None:
            event_queue.task_done()
            return

        date_str = task.date_str
        stock_code = task.stock_code
        stock_ann = task.stock_ann

        try:
            logger.info(f"    [事件线程{worker_id}] {date_str} {stock_code} 开始处理")

            # 1. 下载PDF并提取文本
            all_texts = []
            for _, row in stock_ann.iterrows():
                title = row.get('title', '')
                link = row.get('announcement_link', '')
                if not link:
                    continue
                pdf_content = download_pdf(link)
                if pdf_content:
                    text = extract_text_from_pdf(pdf_content)
                    if text:
                        all_texts.append(text)

            if not all_texts:
                all_texts = stock_ann['title'].tolist()

            combined_text = "\n\n--- 公告分割线 ---\n\n".join(all_texts) if len(all_texts) > 1 else (all_texts[0] if all_texts else "")

            # 2. LLM结构化分析
            structured_result = analyze_announcement_structured(combined_text, STRUCTURED_ANALYSIS_PROMPT)
            if structured_result is None:
                structured_result = analyze_announcement_structured(
                    stock_ann['title'].tolist()[0] if len(stock_ann) > 0 else "",
                    STRUCTURED_ANALYSIS_PROMPT
                )
            if structured_result is None:
                with counter_lock:
                    global total_events_failed
                    total_events_failed += 1
                event_queue.task_done()
                continue

            if isinstance(structured_result, list):
                structured_result = structured_result[0] if len(structured_result) > 0 else None
                if structured_result is None:
                    with counter_lock:
                        total_events_failed += 1
                    event_queue.task_done()
                    continue

            # 3. 必填字段校验
            if not structured_result.get("增持主体"):
                with counter_lock:
                    total_events_failed += 1
                event_queue.task_done()
                continue

            amount_lower = structured_result.get("增持金额下限")
            quantity_lower = structured_result.get("增持数量下限")
            if not _is_valid_value(amount_lower) and not _is_valid_value(quantity_lower):
                with counter_lock:
                    total_events_failed += 1
                event_queue.task_done()
                continue

            # 4. 从内存获取价格、股本、收益率（零rqdatac调用）
            close_price = get_close_price_mem(stock_code, date_str)
            if close_price is None:
                logger.warning(f"    [事件线程{worker_id}] {stock_code}: 无法获取收盘价")
                with counter_lock:
                    total_events_failed += 1
                event_queue.task_done()
                continue

            target_amount = calculate_target_amount(structured_result, close_price)
            if target_amount is None or target_amount <= 0:
                with counter_lock:
                    total_events_failed += 1
                event_queue.task_done()
                continue

            total_shares = get_total_shares_mem(stock_code, date_str)
            total_mv = float(close_price) * float(total_shares) if total_shares else None
            market_cap_ratio = target_amount / total_mv if (total_mv and target_amount) else None

            return_series = calculate_return_series_mem(stock_code, date_str, hold_days=120)

            if return_series and len(return_series) > 0:
                specific_returns = get_specific_period_returns(return_series, periods=[1, 3, 5, 10, 22, 60, 90, 120])
            else:
                specific_returns = {}
                return_series = []

            logger.info(f"    [事件线程{worker_id}] {stock_code}: 收盘价={close_price:.2f} 目标金额={target_amount:,.0f}")

            # 5. 构建最终数据
            final_data = {
                "公告日期": date_str,
                "股票代码": stock_code,
                "公告当日收盘价": close_price,
                "增持目标金额": target_amount,
                "总市值": total_mv,
                "增持占总市值比例": market_cap_ratio,
                "增持主体": structured_result.get("增持主体"),
                "增持标的": structured_result.get("增持标的"),
                "增持金额下限": structured_result.get("增持金额下限"),
                "增持金额上限": structured_result.get("增持金额上限"),
                "增持数量下限": structured_result.get("增持数量下限"),
                "增持数量上限": structured_result.get("增持数量上限"),
                "增持价格上限": structured_result.get("增持价格上限"),
                "增持价格下限": structured_result.get("增持价格下限"),
                "增持目的": structured_result.get("增持目的"),
                "增持期限(月)": structured_result.get("增持期限"),
                "资金来源": structured_result.get("资金来源"),
                "增持方式": structured_result.get("增持方式"),
                "不减持承诺(月)": structured_result.get("不减持承诺月份"),
                "收益率序列": [round(r, 4) if r is not None else None for r in return_series],
                "1日收益率": specific_returns.get("1日收益率"),
                "3日收益率": specific_returns.get("3日收益率"),
                "5日收益率": specific_returns.get("5日收益率"),
                "10日收益率": specific_returns.get("10日收益率"),
                "22日收益率": specific_returns.get("22日收益率"),
                "60日收益率": specific_returns.get("60日收益率"),
                "90日收益率": specific_returns.get("90日收益率"),
                "120日收益率": specific_returns.get("120日收益率"),
            }

            # 6. 保存
            with file_lock:
                save_increase_event(final_data, SAVE_PATH, date_str, stock_code)

            with counter_lock:
                global total_events_processed
                total_events_processed += 1

            logger.info(f"    [事件线程{worker_id}] ✓ {stock_code} 已保存 (成功:{total_events_processed} 失败:{total_events_failed})")

        except Exception as e:
            logger.error(f"    [事件线程{worker_id}] {stock_code}: 异常 {type(e).__name__}: {e}")
            with counter_lock:
                total_events_failed += 1
        finally:
            event_queue.task_done()


# ============================================================================
# 主程序
# ============================================================================
logger.remove()
logger.add(sys.stdout, level="INFO")

logger.info("=" * 60)
logger.info("高管/大股东增持事件分析系统 - 全量预加载版")
logger.info(f"日期线程: {NUM_DATE_WORKERS} | 事件线程: {NUM_EVENT_WORKERS}")
logger.info("=" * 60)

# 阶段1: 预加载所有数据
preload_all_data()

# 阶段2: 加载已处理日期
load_processed_dates()
logger.info(f"\n已处理日期: {len(processed_dates)} 个")

# 确定待处理日期
pending_dates = []
for td in TRADING_DATES_LIST:
    date_str = pd.Timestamp(td).strftime('%Y-%m-%d')
    with processed_dates_lock:
        if date_str not in processed_dates:
            pending_dates.append(date_str)
logger.info(f"待处理日期: {len(pending_dates)} 个")

if not pending_dates:
    logger.info("所有日期已处理，无需运行")
    sys.exit(0)

# 阶段3: 启动事件处理线程
event_queue: queue.Queue = queue.Queue(maxsize=500)
stop_event = threading.Event()
event_threads = []

for wid in range(NUM_EVENT_WORKERS):
    t = threading.Thread(target=event_worker, args=(wid + 1, event_queue, stop_event), daemon=True)
    t.start()
    event_threads.append(t)

logger.info(f"\n阶段2: 两级并行处理（纯LLM多线程，零rqdatac调用）")
logger.info(f"  日期预处理线程池: {NUM_DATE_WORKERS}线程")
logger.info(f"  公司事件处理线程: {NUM_EVENT_WORKERS}线程")

# 阶段4: 启动日期预处理线程池
completed_dates = 0
start_time = time.time()

with ThreadPoolExecutor(max_workers=NUM_DATE_WORKERS) as date_executor:
    future_to_date = {
        date_executor.submit(process_date_pipeline, date_str, event_queue): date_str
        for date_str in pending_dates
    }
    total_pending = len(future_to_date)

    for future in as_completed(future_to_date):
        date_str = future_to_date[future]
        try:
            ds, status, n = future.result()
        except Exception as e:
            logger.error(f"  日期 {date_str}: 线程异常 {e}")
            ds, status, n = date_str, f"thread_error", 0

        completed_dates += 1
        elapsed = (time.time() - start_time) / 60
        speed = completed_dates / elapsed if elapsed > 0 else 0
        remaining_min = (total_pending - completed_dates) / speed if speed > 0 else 0

        with counter_lock:
            done = total_events_processed
            fail = total_events_failed
            found = total_events_found

        logger.info(f"\n📊 进度: {completed_dates}/{total_pending} 日期 ({completed_dates/total_pending*100:.1f}%) | "
                     f"发现{found} | 成功{done} | 失败{fail} | "
                     f"速度{speed:.1f}日/分 | 预计剩余{remaining_min:.0f}分钟")

logger.info("\n✅ 所有日期预处理完成，等待事件队列清空...")

# 等待事件队列清空
event_queue.join()

# 停止事件线程
for _ in range(NUM_EVENT_WORKERS):
    event_queue.put(None)
stop_event.set()
for t in event_threads:
    t.join(timeout=10)

# 完成
logger.info(f"\n{'='*60}")
logger.info(f"🎉 全部处理完成！")
logger.info(f"  共处理交易日: {len(TRADING_DATES_LIST)} (其中待处理 {total_pending})")
logger.info(f"  发现增持预告: {total_events_found} 条")
logger.info(f"  成功处理事件: {total_events_processed} 个")
logger.info(f"  失败事件: {total_events_failed} 个")
logger.info(f"  保存路径: {SAVE_PATH}")
logger.info(f"{'='*60}")
