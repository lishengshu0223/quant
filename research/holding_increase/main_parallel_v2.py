"""
高管/大股东增持事件分析系统 - 两级全局并行版
第一级（10线程）：跨日期并行 - 每天获取公告、关键词筛选、LLM筛选、分组
第二级（10线程）：跨日期全局并行 - 所有日期的公司事件统一排队处理
总并发20线程，充分利用夜间流量
"""
import os
import sys
import time
import json
import threading
import queue
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, List, Tuple

from loguru import logger

import rqdatac
rqdatac.init()

# 导入项目模块
from prompt import FILTER_ANNOUNCEMENT_PROMPT, STRUCTURED_ANALYSIS_PROMPT
from func_tools import (
    get_all_announcements_for_date,
    filter_increase_announcements,
    analyze_announcement_structured,
    download_pdf,
    extract_text_from_pdf,
    get_stock_price,
    get_total_market_cap,
    calculate_target_amount,
    calculate_return_series,
    get_specific_period_returns,
    save_increase_event,
    keyword_filter_announcements,
    RQDATAC_LOCK,
)

# 配置
SEARCH_START_DATE = "2023-01-01"
SEARCH_END_DATE = "2026-07-29"
SAVE_PATH = r"F:\quant\research\holding_increase\increase_events"
TEST_MODE = False
TEST_DATE = "2024-06-28"
PROCESSED_DATES_FILE = r"F:\quant\research\holding_increase\processed_dates.json"
NUM_DATE_WORKERS = 5    # 日期预处理线程数（rqdatac有HTTP连接限制，不要太多）
NUM_EVENT_WORKERS = 10  # 公司事件处理线程数（LLM调用独立，可以更多）

os.makedirs(SAVE_PATH, exist_ok=True)

# 全局线程锁
file_lock = threading.Lock()
counter_lock = threading.Lock()
date_progress_lock = threading.Lock()

# 计数器
total_events_found = 0
total_events_processed = 0
total_events_failed = 0

# 已处理日期集合（线程安全）
processed_dates: set = set()
processed_dates_lock = threading.Lock()


def load_processed_dates():
    """加载已处理日期列表"""
    if os.path.exists(PROCESSED_DATES_FILE):
        try:
            with open(PROCESSED_DATES_FILE, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except:
            return set()
    return set()


def save_processed_dates_locked():
    """线程安全保存已处理日期"""
    with processed_dates_lock:
        dates_copy = sorted(list(processed_dates))
    with open(PROCESSED_DATES_FILE, 'w', encoding='utf-8') as f:
        json.dump(dates_copy, f, ensure_ascii=False, indent=2)


def mark_date_processed(date_str):
    """标记日期已处理并保存"""
    with processed_dates_lock:
        processed_dates.add(date_str)
    save_processed_dates_locked()


def is_date_processed(date_str):
    """检查日期是否已处理"""
    with processed_dates_lock:
        return date_str in processed_dates


def _is_valid_value(v):
    if v is None or v == '' or v == 0 or v == '0':
        return False
    try:
        return float(v) > 0
    except (ValueError, TypeError):
        return False


@dataclass
class EventTask:
    """公司事件处理任务（由日期线程产出，公司线程消费）"""
    date_str: str
    stock_code: str
    stock_ann: pd.DataFrame  # 该公司该日的公告DataFrame


# ============================================================================
# 第一级：日期预处理线程函数
# ============================================================================
def process_date_pipeline(date_str: str, code_list: List[str], event_queue: "queue.Queue[Optional[EventTask]]") -> Tuple[str, str, int]:
    """
    单个日期的处理流水线：获取公告 → 关键词筛选 → LLM筛选 → 分组 → 产出公司事件任务
    返回: (date_str, status, num_events_discovered)
    """
    tid = threading.current_thread().ident
    try:
        # Step 1: 检查是否已处理（双重保险）
        if is_date_processed(date_str):
            return (date_str, "skipped_processed", 0)

        # Step 2: 获取全市场当天公告
        logger.info(f"  [日期线程{tid}] {date_str}: 获取公告...")
        announcements = get_all_announcements_for_date(date_str, stock_list=code_list)

        # 返回None表示全部批次失败，不标记processed，下次可重试
        if announcements is None:
            return (date_str, "announcements_all_failed", 0)

        if announcements.empty:
            mark_date_processed(date_str)
            return (date_str, "no_announcements", 0)

        # Step 3: 关键词预筛选
        if 'title' not in announcements.columns:
            mark_date_processed(date_str)
            return (date_str, "no_title_field", 0)

        pre_filtered = keyword_filter_announcements(announcements)
        if pre_filtered.empty:
            mark_date_processed(date_str)
            return (date_str, "no_keyword_match", 0)

        # Step 3.5: 排除已处理的公司（检查JSON文件）
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
                mark_date_processed(date_str)
                return (date_str, "all_companies_done", 0)

        logger.info(f"  [日期线程{tid}] {date_str}: 初筛{len(pre_filtered)}条，开始LLM筛选...")

        # Step 4: LLM精确筛选（单次LLM调用，占主要耗时）
        filtered_ann = filter_increase_announcements(pre_filtered, FILTER_ANNOUNCEMENT_PROMPT)

        if filtered_ann is None or filtered_ann.empty:
            mark_date_processed(date_str)
            return (date_str, "llm_no_match", 0)

        # Step 5: 按公司分组
        if 'order_book_id' in filtered_ann.columns:
            stock_groups = filtered_ann.groupby('order_book_id')
        elif 'media' in filtered_ann.columns:
            stock_groups = filtered_ann.groupby('media')
        else:
            mark_date_processed(date_str)
            return (date_str, "no_stock_code_col", 0)

        num_companies = len(stock_groups)
        logger.info(f"  [日期线程{tid}] {date_str}: LLM筛选出{num_companies}家公司，提交到事件队列")

        # Step 6: 将每家公司作为独立任务提交到事件队列
        discovered_count = 0
        for stock_code, stock_ann in stock_groups:
            event_queue.put(EventTask(
                date_str=date_str,
                stock_code=stock_code,
                stock_ann=stock_ann.copy(),
            ))
            discovered_count += 1

        with counter_lock:
            global total_events_found
            total_events_found += discovered_count

        # 标记日期已处理（注意：公司事件还在队列中，但日期层面已经产出完毕）
        mark_date_processed(date_str)

        return (date_str, f"ok_{discovered_count}_companies", discovered_count)

    except Exception as e:
        logger.error(f"  [日期线程{tid}] {date_str}: 预处理异常: {e}")
        # 异常日期不标记为已处理，下次可重试
        return (date_str, f"error_{type(e).__name__}", 0)


# ============================================================================
# 第二级：公司事件处理线程函数
# ============================================================================
def event_worker(worker_id: int, event_queue: "queue.Queue[Optional[EventTask]]", stop_event: threading.Event):
    """
    公司事件处理消费者：
    从事件队列取出任务，处理（PDF下载、LLM结构化分析、计算、保存）
    收到None则退出
    """
    logger.info(f"  [事件线程{worker_id}] 启动，等待任务...")

    while not stop_event.is_set() or not event_queue.empty():
        try:
            task = event_queue.get(timeout=2)
        except queue.Empty:
            continue

        # 哨兵值：None 表示停止
        if task is None:
            event_queue.task_done()
            logger.info(f"  [事件线程{worker_id}] 收到停止信号，退出")
            return

        date_str = task.date_str
        stock_code = task.stock_code
        stock_ann = task.stock_ann

        try:
            logger.info(f"    [事件线程{worker_id}] {date_str} {stock_code} 开始处理")

            # Step 1: 下载PDF并提取文本
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

            if len(all_texts) > 1:
                combined_text = "\n\n--- 公告分割线 ---\n\n".join(all_texts)
            else:
                combined_text = all_texts[0] if all_texts else ""

            # Step 2: LLM结构化分析（主要耗时）
            structured_result = analyze_announcement_structured(combined_text, STRUCTURED_ANALYSIS_PROMPT)
            if structured_result is None:
                structured_result = analyze_announcement_structured(
                    stock_ann['title'].tolist()[0] if len(stock_ann) > 0 else "",
                    STRUCTURED_ANALYSIS_PROMPT
                )
            if structured_result is None:
                logger.warning(f"    [事件线程{worker_id}] {stock_code}: 结构化分析失败")
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

            # Step 3: 必填字段校验
            if not structured_result.get("增持主体"):
                with counter_lock:
                    total_events_failed += 1
                event_queue.task_done()
                continue

            amount_lower = structured_result.get("增持金额下限")
            quantity_lower = structured_result.get("增持数量下限")
            if not _is_valid_value(amount_lower) and not _is_valid_value(quantity_lower):
                logger.warning(f"    [事件线程{worker_id}] {stock_code}: 金额/数量下限无效")
                with counter_lock:
                    total_events_failed += 1
                event_queue.task_done()
                continue

            # Step 4: 获取价格、市值、收益率（rqdatac调用在函数内部已加锁）
            close_price = get_stock_price(stock_code, date_str)
            if close_price is None:
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

            total_mv = get_total_market_cap(stock_code, date_str)
            return_series = calculate_return_series(stock_code, date_str, hold_days=120)

            market_cap_ratio = None
            if total_mv and target_amount:
                market_cap_ratio = target_amount / total_mv

            # Step 5: 特定周期收益率
            if return_series is not None and len(return_series) > 0:
                specific_returns = get_specific_period_returns(
                    return_series,
                    periods=[1, 3, 5, 10, 22, 60, 90, 120]
                )
            else:
                specific_returns = {}
                return_series = []

            # Step 6: 构建最终数据
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

            # Step 7: 保存（加锁）
            with file_lock:
                save_increase_event(final_data, SAVE_PATH, date_str, stock_code)

            with counter_lock:
                global total_events_processed
                total_events_processed += 1

            logger.info(f"    [事件线程{worker_id}] ✓ {stock_code} 已保存 (成功:{total_events_processed}, 失败:{total_events_failed})")

        except Exception as e:
            logger.error(f"    [事件线程{worker_id}] {stock_code}: 处理异常 {type(e).__name__}: {e}")
            with counter_lock:
                total_events_failed += 1
        finally:
            event_queue.task_done()


# ============================================================================
# 主流程
# ============================================================================
logger.remove()
logger.add(sys.stdout, level="INFO")

logger.info("=" * 60)
logger.info("高管/大股东增持事件分析系统 - 两级全局并行版")
logger.info(f"日期线程: {NUM_DATE_WORKERS} | 事件线程: {NUM_EVENT_WORKERS}")
logger.info("=" * 60)

# Step 1: 获取股票代码
logger.info("\n[Step 1] 获取全部A股股票代码...")
code_list = rqdatac.all_instruments(type='CS', date=SEARCH_END_DATE)['order_book_id'].tolist()
logger.info(f"获取到 {len(code_list)} 只A股股票")

# Step 2: 获取交易日列表
if TEST_MODE:
    trading_dates = [pd.Timestamp(TEST_DATE)]
    logger.info(f"\n[Step 2] 测试模式，仅处理 {TEST_DATE}")
else:
    logger.info(f"\n[Step 2] 获取交易日列表 {SEARCH_START_DATE} ~ {SEARCH_END_DATE}...")
    trading_dates = rqdatac.get_trading_dates(start_date=SEARCH_START_DATE, end_date=SEARCH_END_DATE)
    logger.info(f"共 {len(trading_dates)} 个交易日")

# Step 3: 加载已处理日期
processed_dates = load_processed_dates()
logger.info(f"已加载 {len(processed_dates)} 个已处理日期")

# 过滤未处理日期
pending_dates = []
for td in trading_dates:
    date_str = pd.Timestamp(td).strftime('%Y-%m-%d')
    if not is_date_processed(date_str):
        pending_dates.append(date_str)
logger.info(f"待处理日期: {len(pending_dates)} 个")

# Step 4: 创建事件队列 + 启动事件处理线程
event_queue: "queue.Queue[Optional[EventTask]]" = queue.Queue(maxsize=500)  # 队列有界防爆内存
stop_event = threading.Event()
event_threads = []

for wid in range(NUM_EVENT_WORKERS):
    t = threading.Thread(target=event_worker, args=(wid + 1, event_queue, stop_event), daemon=True)
    t.start()
    event_threads.append(t)

logger.info(f"\n[Step 3] 两级线程池启动：")
logger.info(f"  日期预处理线程池: {NUM_DATE_WORKERS}线程")
logger.info(f"  公司事件处理线程: {NUM_EVENT_WORKERS}线程")

# Step 5: 启动日期预处理线程池
completed_dates = 0
start_time = time.time()

with ThreadPoolExecutor(max_workers=NUM_DATE_WORKERS) as date_executor:
    future_to_date = {
        date_executor.submit(process_date_pipeline, date_str, code_list, event_queue): date_str
        for date_str in pending_dates
    }
    total_pending = len(future_to_date)

    for future in as_completed(future_to_date):
        date_str = future_to_date[future]
        try:
            ds, status, n = future.result()
        except Exception as e:
            logger.error(f"  日期 {date_str}: 线程异常 {e}")
            ds, status, n = date_str, f"thread_error_{type(e).__name__}", 0

        completed_dates += 1
        elapsed = (time.time() - start_time) / 60
        if completed_dates > 0:
            speed = completed_dates / elapsed if elapsed > 0 else 0
            remaining_min = (total_pending - completed_dates) / speed if speed > 0 else 0
        else:
            remaining_min = 0

        with counter_lock:
            done = total_events_processed
            fail = total_events_failed
            found = total_events_found

        logger.info(f"\n📊 进度: {completed_dates}/{total_pending} 日期 ({completed_dates/total_pending*100:.1f}%) | "
                     f"发现{found} | 成功{done} | 失败{fail} | "
                     f"速度{speed:.1f}日/分 | 预计剩余{remaining_min:.0f}分钟")

logger.info("\n✅ 所有日期预处理完成，等待事件队列清空...")

# Step 6: 等待事件队列清空
event_queue.join()

# 发送停止信号给事件线程
for _ in range(NUM_EVENT_WORKERS):
    event_queue.put(None)

stop_event.set()
for t in event_threads:
    t.join(timeout=10)

# ============================================================================
# 完成
# ============================================================================
logger.info(f"\n{'='*60}")
logger.info(f"🎉 全部处理完成！")
logger.info(f"  共处理交易日: {len(trading_dates)} (其中待处理 {total_pending})")
logger.info(f"  发现增持预告: {total_events_found} 条")
logger.info(f"  成功处理事件: {total_events_processed} 个")
logger.info(f"  失败事件: {total_events_failed} 个")
logger.info(f"  保存路径: {SAVE_PATH}")
logger.info(f"{'='*60}")
