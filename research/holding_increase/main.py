"""
高管/大股东增持事件分析系统
主程序 - 获取全市场公告，筛选高管/大股东增持预告事件
"""
import os
import sys
import time
import json
import pandas as pd

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
    check_event_exists,
    keyword_filter_announcements,
    convert_to_json_serializable,
)

# 配置
SEARCH_START_DATE = "2023-01-01"
SEARCH_END_DATE = "2026-07-29"
SAVE_PATH = r"F:\quant\research\holding_increase\increase_events"
TEST_MODE = False  # 正式模式：运行全量日期范围
TEST_DATE = "2024-06-28"  # 测试日期（仅TEST_MODE=True时使用）
PROCESSED_DATES_FILE = r"F:\quant\research\holding_increase\processed_dates.json"

# 创建保存目录
os.makedirs(SAVE_PATH, exist_ok=True)


def load_processed_dates():
    """加载已处理日期列表"""
    if os.path.exists(PROCESSED_DATES_FILE):
        try:
            with open(PROCESSED_DATES_FILE, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except:
            return set()
    return set()


def save_processed_dates(processed_dates):
    """保存已处理日期列表"""
    with open(PROCESSED_DATES_FILE, 'w', encoding='utf-8') as f:
        json.dump(sorted(list(processed_dates)), f, ensure_ascii=False, indent=2)

# 配置日志到控制台（先移除默认handler避免重复输出）
logger.remove()
logger.add(sys.stdout, level="INFO")

logger.info("=" * 60)
logger.info("高管/大股东增持事件分析系统")
logger.info("=" * 60)

# ============================================================================
# Step 1: 获取全部A股股票代码
# ============================================================================
logger.info("\n[Step 1] 获取全部A股股票代码...")
code_list = rqdatac.all_instruments(type='CS', date=SEARCH_END_DATE)
code_list = code_list['order_book_id'].tolist()
logger.info(f"获取到 {len(code_list)} 只A股股票")

# ============================================================================
# Step 2: 获取交易日列表
# ============================================================================
if TEST_MODE:
    trading_dates = [pd.Timestamp(TEST_DATE)]
    logger.info(f"\n[Step 2] 测试模式，仅处理 {TEST_DATE}")
else:
    logger.info(f"\n[Step 2] 获取交易日列表 {SEARCH_START_DATE} ~ {SEARCH_END_DATE}...")
    trading_dates = rqdatac.get_trading_dates(
        start_date=SEARCH_START_DATE,
        end_date=SEARCH_END_DATE
    )
    logger.info(f"共 {len(trading_dates)} 个交易日")

# ============================================================================
# Step 3: 按交易日循环处理
# ============================================================================
total_events_found = 0
total_events_processed = 0
processed_dates = load_processed_dates()
logger.info(f"已加载 {len(processed_dates)} 个已处理日期")

for date_idx, trade_date in enumerate(trading_dates):
    date_str = pd.Timestamp(trade_date).strftime('%Y-%m-%d')

    # 跳过已处理的日期
    if date_str in processed_dates:
        continue

    logger.info(f"\n{'='*60}")
    logger.info(f"[日期 {date_idx+1}/{len(trading_dates)}] 处理 {date_str}")
    logger.info(f"{'='*60}")

    try:
        # Step 3.2: 获取全市场当天公告
        logger.info(f"  获取全市场公告...")
        announcements = get_all_announcements_for_date(date_str, stock_list=code_list)

        if announcements.empty:
            logger.info(f"  {date_str} 无公告，跳过")
            processed_dates.add(date_str)
            save_processed_dates(processed_dates)
            continue

        logger.info(f"  共 {len(announcements)} 条公告")

        # Step 3.3: 关键词预筛选
        logger.info(f"  关键词预筛选...")
        if 'title' not in announcements.columns:
            logger.warning(f"  公告数据无title字段，跳过")
            processed_dates.add(date_str)
            save_processed_dates(processed_dates)
            continue

        pre_filtered = keyword_filter_announcements(announcements)

        if pre_filtered.empty:
            logger.info(f"  无增持相关公告，跳过")
            processed_dates.add(date_str)
            save_processed_dates(processed_dates)
            continue

        logger.info(f"  初步筛选出 {len(pre_filtered)} 条疑似增持公告")

        # Step 3.3.5: 排除已处理的公司（直接比较已保存JSON文件中的股票代码）
        date_str_compact = date_str.replace('-', '')
        existing_files = [f for f in os.listdir(SAVE_PATH)
                         if f.startswith(date_str_compact) and f.endswith('.json')] if os.path.exists(SAVE_PATH) else []
        processed_stocks = set()
        for f in existing_files:
            # 文件名格式: YYYYmmdd_xxxxxx.json
            parts = f.replace('.json', '').split('_')
            if len(parts) == 2:
                processed_stocks.add(parts[1])

        if processed_stocks:
            logger.info(f"  已保存 {len(processed_stocks)} 个事件: {processed_stocks}")
            # 从预筛选结果中排除已处理公司的公告
            if 'order_book_id' in pre_filtered.columns:
                stock_codes_6 = pre_filtered['order_book_id'].str[:6]
                before_count = len(pre_filtered)
                pre_filtered = pre_filtered[~stock_codes_6.isin(processed_stocks)].copy()
                logger.info(f"  排除已处理后: {before_count} -> {len(pre_filtered)} 条")

            if pre_filtered.empty:
                logger.info(f"  所有公司已处理，跳过")
                processed_dates.add(date_str)
                save_processed_dates(processed_dates)
                continue

        # Step 3.4: 大模型精确筛选
        logger.info(f"  LLM精确筛选...")
        filtered_ann = filter_increase_announcements(pre_filtered, FILTER_ANNOUNCEMENT_PROMPT)

        if filtered_ann is None or filtered_ann.empty:
            logger.info(f"  LLM筛选后无增持预告公告，跳过")
            processed_dates.add(date_str)
            save_processed_dates(processed_dates)
            continue

        logger.info(f"  LLM筛选确定 {len(filtered_ann)} 条增持预告公告")
        logger.info(f"  筛选后列: {filtered_ann.columns.tolist()}")

        total_events_found += len(filtered_ann)

        # Step 3.5: 按公司分组处理
        if 'order_book_id' in filtered_ann.columns:
            logger.info(f"  使用 order_book_id 列分组")
            stock_groups = filtered_ann.groupby('order_book_id')
        elif 'media' in filtered_ann.columns:
            logger.info(f"  使用 media 列分组")
            stock_groups = filtered_ann.groupby('media')
        else:
            logger.error(f"  无法确定股票代码列，跳过")
            processed_dates.add(date_str)
            save_processed_dates(processed_dates)
            continue

        logger.info(f"  涉及 {len(stock_groups)} 家公司")

        for stock_code, stock_ann in stock_groups:
            try:
                logger.info(f"\n    处理 {stock_code} ({len(stock_ann)} 条公告)")

                # Step 3.5.1: 下载PDF并提取文本
                all_texts = []
                for _, row in stock_ann.iterrows():
                    title = row.get('title', '')
                    link = row.get('announcement_link', '')

                    if not link:
                        logger.warning(f"      无公告链接: {title[:40]}...")
                        continue

                    # 下载PDF
                    pdf_content = download_pdf(link)
                    if pdf_content:
                        text = extract_text_from_pdf(pdf_content)
                        if text:
                            logger.info(f"      ✓ 提取文本: {len(text)} 字符 - {title[:40]}...")
                            all_texts.append(text)
                        else:
                            logger.warning(f"      未能提取文本")
                    else:
                        logger.warning(f"      下载失败")

                if not all_texts:
                    logger.warning(f"      无公告文本，使用标题作为兜底")
                    all_texts = stock_ann['title'].tolist()

                # 合并文本
                if len(all_texts) > 1:
                    combined_text = "\n\n--- 公告分割线 ---\n\n".join(all_texts)
                else:
                    combined_text = all_texts[0] if all_texts else ""

                # Step 3.5.2: 大模型结构化分析
                logger.info(f"      大模型结构化分析...")
                structured_result = analyze_announcement_structured(combined_text, STRUCTURED_ANALYSIS_PROMPT)

                if structured_result is None:
                    logger.error(f"      结构化分析失败，使用标题文本重试")
                    structured_result = analyze_announcement_structured(
                        stock_ann['title'].tolist()[0] if len(stock_ann) > 0 else "",
                        STRUCTURED_ANALYSIS_PROMPT
                    )

                if structured_result is None:
                    logger.error(f"      结构化分析失败，跳过")
                    continue

                # 如果LLM返回的是列表，取第一个元素
                if isinstance(structured_result, list):
                    if len(structured_result) > 0:
                        structured_result = structured_result[0]
                    else:
                        logger.warning(f"      结构化分析返回空列表，跳过")
                        continue

                logger.info(f"      ✓ 结构化分析完成")
                logger.info(f"      结果: {json.dumps(structured_result, ensure_ascii=False)[:200]}...")

                # Step 3.5.2.5: 必填字段校验
                # 增持主体必须存在
                if not structured_result.get("增持主体"):
                    logger.warning(f"      缺少必填字段[增持主体]，跳过")
                    continue

                # 金额下限和数量下限至少有一个必须存在且有效
                amount_lower = structured_result.get("增持金额下限")
                quantity_lower = structured_result.get("增持数量下限")

                def _is_valid_value(v):
                    """检查值是否有效（非空、非0、非None）"""
                    if v is None or v == '' or v == 0 or v == '0':
                        return False
                    try:
                        return float(v) > 0
                    except (ValueError, TypeError):
                        return False

                if not _is_valid_value(amount_lower) and not _is_valid_value(quantity_lower):
                    logger.warning(f"      缺少必填字段[增持金额下限]和[增持数量下限]，两者均无效，跳过")
                    logger.info(f"        金额下限={amount_lower}, 数量下限={quantity_lower}")
                    continue

                # Step 3.5.3: 获取价格和计算指标
                logger.info(f"      获取价格和计算指标...")

                # 获取收盘价
                close_price = get_stock_price(stock_code, date_str)
                if close_price is None:
                    logger.error(f"      无法获取收盘价，跳过")
                    continue

                logger.info(f"      收盘价: {close_price:.4f}")

                # 计算增持目标金额
                target_amount = calculate_target_amount(structured_result, close_price)
                if target_amount is None or target_amount <= 0:
                    logger.warning(f"      无法计算增持目标金额，跳过")
                    continue
                logger.info(f"      增持目标金额: {target_amount:,.0f} 元")

                # 获取总市值
                total_mv = get_total_market_cap(stock_code, date_str)
                market_cap_ratio = None
                if total_mv and target_amount:
                    market_cap_ratio = target_amount / total_mv
                    logger.info(f"      总市值: {total_mv:,.0f}, 增持占比: {market_cap_ratio*100:.4f}%")

                # 计算收益率序列
                logger.info(f"      计算收益率序列...")
                return_series = calculate_return_series(stock_code, date_str, hold_days=120)

                if return_series is not None and len(return_series) > 0:
                    specific_returns = get_specific_period_returns(
                        return_series,
                        periods=[1, 3, 5, 10, 22, 60, 90, 120]
                    )
                    logger.info(f"      ✓ 收益率序列计算完成")
                else:
                    specific_returns = {}
                    return_series = []
                    logger.warning(f"      收益率序列计算失败")

                # Step 3.5.4: 构建并保存最终数据
                final_data = {
                    # 基础信息（排在最前）
                    "公告日期": date_str,
                    "股票代码": stock_code,

                    # 增持金额相关
                    "公告当日收盘价": close_price,
                    "增持目标金额": target_amount,

                    # 总市值占比
                    "总市值": total_mv,
                    "增持占总市值比例": market_cap_ratio,

                    # 结构化分析结果
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

                    # 收益率序列
                    "收益率序列": [round(r, 4) if r is not None else None for r in return_series],

                    # 特定周期收益率
                    "1日收益率": specific_returns.get("1日收益率"),
                    "3日收益率": specific_returns.get("3日收益率"),
                    "5日收益率": specific_returns.get("5日收益率"),
                    "10日收益率": specific_returns.get("10日收益率"),
                    "22日收益率": specific_returns.get("22日收益率"),
                    "60日收益率": specific_returns.get("60日收益率"),
                    "90日收益率": specific_returns.get("90日收益率"),
                    "120日收益率": specific_returns.get("120日收益率"),
                }

                # 保存JSON文件
                file_path = save_increase_event(final_data, SAVE_PATH, date_str, stock_code)
                total_events_processed += 1

                logger.info(f"      ✓ 已保存: {os.path.basename(file_path)}")

            except Exception as e:
                logger.error(f"      处理 {stock_code} 时发生异常，跳过该事件: {e}")
                continue

        # 当天所有公司处理完毕，标记日期为已处理
        processed_dates.add(date_str)
        save_processed_dates(processed_dates)

    except Exception as e:
        logger.error(f"  处理日期 {date_str} 时发生异常，跳过该日期: {e}")
        # 等待30秒后继续下一个日期，避免连续异常
        import time as _time
        _time.sleep(30)
        continue

# ============================================================================
# 完成
# ============================================================================
logger.info(f"\n{'='*60}")
logger.info(f"处理完成！")
logger.info(f"  共处理 {len(trading_dates)} 个交易日")
logger.info(f"  发现 {total_events_found} 条增持预告")
logger.info(f"  成功处理 {total_events_processed} 个事件")
logger.info(f"  保存路径: {SAVE_PATH}")
logger.info(f"{'='*60}")
