import os
import re
import json
import time
import requests
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime, timedelta

from openai import OpenAI

import rqdatac
rqdatac.init()

from func_tools import analyze_with_llm, download_pdf, extract_text_from_pdf, filter_incentive_announcements, prioritize_announcements, convert_to_json_serializable
from prompt import ANALYSIS_PROMPT_TEMPLATE

try:
    DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
except KeyError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from config import DEEPSEEK_API_KEY

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"

# 搜索日期范围(查找该时间段内第一个有股权激励公告的交易日)
SEARCH_START_DATE = "2023-01-01"
SEARCH_END_DATE = "2026-07-01"

SAVE_PATH = r"F:\quant\research\equity_incentive\briefing_json"

# 创建保存目录
os.makedirs(SAVE_PATH, exist_ok=True)

# Step 1: 获取全部A股股票代码
print("\n[Step 1] 获取全部A股股票代码...")
code_list = rqdatac.all_instruments(type='CS', date=SEARCH_END_DATE)
code_list = code_list['order_book_id'].tolist()
print(f"  获取到 {len(code_list)} 只A股股票")


# ----------------------------------------------------------------
# Step 2: 获取交易日列表
# ----------------------------------------------------------------
print(f"\n[Step 2] 获取交易日列表 {SEARCH_START_DATE} ~ {SEARCH_END_DATE}...")
trading_dates = rqdatac.get_trading_dates(
    start_date=SEARCH_START_DATE,
    end_date=SEARCH_END_DATE
)
print(f"  共 {len(trading_dates)} 个交易日")

# 用于追踪处理进度
processed_count = 0
total_events = 0

for trade_date in trading_dates:
    date_str = pd.Timestamp(trade_date).strftime('%Y-%m-%d')
    print(f"\n[Step 3] 检查交易日: {date_str}")
    
    incentive_data = pd.read_parquet(
        os.path.join(r"F:\quant\research\equity_incentive\incentive_plan", f"{pd.Timestamp(trade_date).strftime('%Y%m%d')}.parquet")
    )
    if incentive_data.empty:
        print(f"  当日无股权激励公告")
        continue
    
    print(f"  ✓ 在 {date_str} 找到 {len(incentive_data)} 条激励数据!")
    total_events += len(incentive_data)
    
    for idx, (_, row) in enumerate(incentive_data.iterrows()):
        print(row)
        # 确保info_date是字符串类型
        order_book_id = row["order_book_id"]
        info_date = row["info_date"]
        if hasattr(info_date, 'strftime'):
            info_date_str = info_date.strftime('%Y-%m-%d')
        else:
            info_date_str = str(info_date)
        file_name = f"{info_date_str.replace('-', '')}_{order_book_id[:6]}.json"
        json_path = os.path.join(SAVE_PATH, file_name)
        
        # 跳过已处理的文件
        if os.path.exists(json_path):
            print(f"  [{idx+1}/{len(incentive_data)}] 文件: {file_name} 已存在，跳过")
            processed_count += 1
            continue
        
        print(f"\n  [{idx+1}/{len(incentive_data)}] 处理股票: {order_book_id}, 公告日期: {info_date_str}")
        
        try:
            # 获取基础数据
            pre_trading_date = rqdatac.get_previous_trading_date(info_date_str)
            price_data = rqdatac.get_price(
                order_book_id, 
                pre_trading_date, 
                pre_trading_date, 
                fields=['close'], 
                adjust_type='none'
            )
            if price_data is not None and len(price_data) > 0:
                pre_close_price = price_data['close'].values[0]
            else:
                print(f"  无法获取股票价格数据，跳过")
                continue

            staff_data = rqdatac.get_staff_count(order_book_id, end_date=pre_trading_date)
            if staff_data is not None and len(staff_data) > 0:
                employee_num = staff_data["staff_count"].iloc[-1]
            else:
                print(f"  无法获取员工数量数据，使用默认值")
                employee_num = None
            shares_num = row['shares_num']
            incentive_price = row['incentive_price']
            incentive_mode = row['incentive_mode']

            # 获取公告数据
            print(f"  获取公告数据...")
            announcements = rqdatac.get_announcement(
                order_book_ids=[order_book_id],
                start_date=info_date_str,
                end_date=info_date_str
            )
            filtered_ann = filter_incentive_announcements(announcements)
            # 按优先级筛选，减少token消耗
            prioritized_ann = prioritize_announcements(filtered_ann)
            
            # 下载公告PDF并提取文本
            print(f"  下载公告PDF并提取文本...")
            all_texts = []
            download_success_count = 0
            download_fail_count = 0
            
            if prioritized_ann is not None and not prioritized_ann.empty:
                for j, (_, row) in enumerate(prioritized_ann.iterrows()):
                    title = row.get('title', '')
                    link = row.get('announcement_link', '')
                    print(f"    [{j+1}/{len(prioritized_ann)}] {title[:50]}...")
                    print(f"      链接: {link[:100]}...")

                    pdf_content = download_pdf(link)
                    if pdf_content:
                        text = extract_text_from_pdf(pdf_content)
                        if text:
                            print(f"      提取文本: {len(text)} 字符")
                            all_texts.append(text)
                            download_success_count += 1
                        else:
                            print(f"      未能提取文本")
                            download_fail_count += 1
                    else:
                        print(f"      下载失败")
                        download_fail_count += 1
            else:
                print("    无相关公告可下载")
            
            # 检查是否有文本数据
            if not all_texts:
                print("  警告: 未获取到任何公告文本")
                print("  跳过此事件，等待后续手动处理上交所PDF下载问题")
                # 记录下载失败的事件
                fail_record = {
                    "公告日": info_date_str,
                    "股票代码": order_book_id,
                    "状态": "PDF下载失败",
                    "失败链接数": download_fail_count,
                    "激励数量": shares_num,
                    "激励价格": incentive_price,
                    "激励模式": incentive_mode,
                }
                fail_file = os.path.join(SAVE_PATH, f"{info_date_str.replace('-', '')}_{order_book_id[:6]}_FAIL.json")
                with open(fail_file, "w", encoding="utf-8") as f:
                    json.dump(fail_record, f, ensure_ascii=False, indent=2)
                print(f"  已记录失败事件: {fail_file}")
                continue

            # LLM分析公告内容（带重试机制）
            print(f"\n  [Step 4] LLM分析公告内容...")
            client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
            
            max_retries = 3
            llm_result = None
            for retry in range(max_retries):
                try:
                    llm_result = analyze_with_llm(
                        info_date_str=info_date_str,
                        texts=all_texts, 
                        prompt=ANALYSIS_PROMPT_TEMPLATE, 
                        model_name=DEEPSEEK_MODEL, 
                        client=client
                    )
                    # 检查解析结果是否完整：必须包含考核科目且不为空
                    if llm_result is not None:
                        exam_subjects = llm_result.get("考核科目", [])
                        if exam_subjects and isinstance(exam_subjects, list) and len(exam_subjects) > 0:
                            print(f"  ✓ LLM解析成功，提取到 {len(exam_subjects)} 个考核科目")
                            break
                        else:
                            print(f"  LLM解析结果不完整：考核科目为空，第 {retry+1}/{max_retries} 次重试...")
                    else:
                        print(f"  LLM解析失败，第 {retry+1}/{max_retries} 次重试...")
                    time.sleep(2)
                except Exception as e:
                    print(f"  LLM调用异常，第 {retry+1}/{max_retries} 次重试: {e}")
                    time.sleep(3)
            
            if llm_result is None:
                print("  LLM分析失败，跳过此事件")
                continue
            
            # 计算衍生指标
            if llm_result["参与人数"] is not None and employee_num > 0:
                llm_result["激励人数占总员工比例"] = round(llm_result["参与人数"] / employee_num, 6)
                if llm_result["核心技术/业务骨干人数"] is not None and llm_result["核心技术/业务骨干人数"] > 0 and llm_result["参与人数"] > 0:
                    llm_result["骨干人数占激励人数比例"] = round(llm_result["核心技术/业务骨干人数"] / llm_result["参与人数"], 6)
                else:
                    llm_result["骨干人数占激励人数比例"] = None
            else:
                llm_result["激励人数占总员工比例"] = None
                llm_result["骨干人数占激励人数比例"] = None

            llm_result["折价率"] = round(incentive_price / pre_close_price, 4)
            if llm_result["参与人数"] is not None and incentive_price is not None and shares_num is not None:
                llm_result["人均激励金额"] = round((incentive_price * shares_num) / llm_result["参与人数"], 4)
            else:
                llm_result["人均激励金额"] = None
            
            # 处理考核科目数据
            incentive_items = llm_result.get("考核科目", [])
            new_incentive_items = []
            
            if incentive_items and isinstance(incentive_items, list):
                for i, incentive_item in enumerate(incentive_items):
                    try:
                        financial_item = incentive_item.get("财务科目名", "")
                        expectation_item = incentive_item.get("一致预期名", "")
                        values = incentive_item.get("数值", [])
                        
                        if not financial_item or not expectation_item:
                            print(f"    跳过第{i+1}个考核科目: 缺少必要字段")
                            continue
                            
                        if ("未找到" in financial_item) or ("未找到" in expectation_item):
                            print(f"    跳过第{i+1}个考核科目: 字段未找到")
                            new_incentive_items.append(incentive_item)
                            continue
                            
                        new_incentive_item = {
                            "财务科目名": financial_item,
                            "一致预期名": expectation_item,
                            "数值": values,
                        }
                        
                        if len(values) == 2:
                            # 增长率型: [基准年, 增速]
                            year, growth = values
                            print(f"    处理第{i+1}个考核科目: {financial_item}, 基准年={year}, 增速={growth}")
                            
                            try:
                                # 获取基准期财务数据
                                base_value = None
                                fin_result = rqdatac.get_pit_financials_ex(
                                    order_book_id, financial_item, f"{year}q4", f"{year}q4", info_date_str
                                )
                                
                                if fin_result is not None and not fin_result.empty:
                                    base_value = fin_result[financial_item].values[0]
                                    if base_value is None or np.isnan(base_value):
                                        base_value = None
                                
                                # 如果财务数据为空，尝试用一致预期
                                if (base_value is None) and int(info_date_str.split('-')[1]) <= 4:
                                    print(f"      尝试使用一致预期作为基准值...")
                                    # 根据一致预期名推断基准年的一致预期字段
                                    base_exp_item = expectation_item
                                    
                                    try:
                                        cons_base_result = rqdatac.consensus.get_factor(
                                            order_book_id, base_exp_item, pre_trading_date, pre_trading_date
                                        )
                                        if cons_base_result is not None and not cons_base_result.empty:
                                            base_value = cons_base_result[base_exp_item].values[0]
                                            if base_value is None or np.isnan(base_value):
                                                base_value = None
                                    except:
                                        pass
                                
                                if base_value is None:
                                    raise ValueError("无法获取基准期数据")
                                
                                target_value = round(base_value * (1 + float(growth)), 2)
                                
                                # 获取一致预期数据（增加容错）
                                expectation_value = None
                                cons_result = None
                                
                                # 尝试多种日期范围
                                date_ranges = [
                                    (info_date_str, info_date_str),
                                    (pre_trading_date, pre_trading_date),
                                ]
                                
                                for start_dt, end_dt in date_ranges:
                                    try:
                                        cons_result = rqdatac.consensus.get_factor(
                                            order_book_id, expectation_item, start_dt, end_dt
                                        )
                                        if cons_result is not None and not cons_result.empty:
                                            expectation_value = cons_result[expectation_item].values[0]
                                            if expectation_value is not None and not np.isnan(expectation_value):
                                                break
                                    except:
                                        continue
                                
                                if expectation_value is None or np.isnan(expectation_value):
                                    raise ValueError("无法获取一致预期数据")
                                
                                achievement_rate = round(target_value / expectation_value, 4)
                                
                                new_incentive_item.update({
                                    "基准期": year,
                                    "基准期财务数值": round(float(base_value), 2),
                                    "目标值": target_value,
                                    "一致预期数值": round(float(expectation_value), 2),
                                    "实现率": achievement_rate,
                                })
                                print(f"      ✓ 基准值={base_value:,.2f}, 目标值={target_value:,.2f}, 一致预期={expectation_value:,.2f}, 实现率={achievement_rate:.2%}")
                                
                            except Exception as e:
                                print(f"      ✗ 财务数据获取失败: {e}")
                                new_incentive_item["数据获取失败"] = str(e)
                                
                        elif len(values) == 1:
                            # 具体数值型: [目标值]
                            target_value = values[0]
                            print(f"    处理第{i+1}个考核科目: {financial_item}, 目标值={target_value}")
                            
                            try:
                                # 获取一致预期数据（增加容错）
                                expectation_value = None
                                
                                # 尝试多种日期范围
                                date_ranges = [
                                    (info_date_str, info_date_str),
                                    (pre_trading_date, pre_trading_date),
                                ]
                                
                                for start_dt, end_dt in date_ranges:
                                    try:
                                        cons_result = rqdatac.consensus.get_factor(
                                            order_book_id, expectation_item, start_dt, end_dt
                                        )
                                        if cons_result is not None and not cons_result.empty:
                                            expectation_value = cons_result[expectation_item].values[0]
                                            if expectation_value is not None and not np.isnan(expectation_value):
                                                break
                                    except:
                                        continue
                                
                                if expectation_value is None or np.isnan(expectation_value):
                                    raise ValueError("无法获取一致预期数据")
                                
                                achievement_rate = round(target_value / expectation_value, 4)
                                
                                new_incentive_item.update({
                                    "目标值": target_value,
                                    "一致预期数值": round(float(expectation_value), 2),
                                    "实现率": achievement_rate,
                                })
                                print(f"      ✓ 目标值={target_value:,.2f}, 一致预期={expectation_value:,.2f}, 实现率={achievement_rate:.2%}")
                                
                            except Exception as e:
                                print(f"      ✗ 一致预期数据获取失败: {e}")
                                new_incentive_item["数据获取失败"] = str(e)
                                
                        else:
                            print(f"    跳过第{i+1}个考核科目: 数值格式不支持，当前格式: {values}")
                            
                        new_incentive_items.append(new_incentive_item)
                        
                    except Exception as e:
                        print(f"    处理第{i+1}个考核科目异常: {e}")
                        new_incentive_items.append(incentive_item)
            
            llm_result["考核科目"] = new_incentive_items
            llm_result["公告日"] = info_date_str
            llm_result["股票代码"] = order_book_id
            llm_result["激励数量"] = shares_num
            llm_result["激励价格"] = incentive_price
            llm_result["激励模式"] = incentive_mode
            llm_result["前一交易日收盘价"] = pre_close_price
            llm_result["员工总数"] = employee_num
            
            # 转换为JSON可序列化格式
            llm_result = convert_to_json_serializable(llm_result)
            
            # 保存JSON
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(llm_result, f, ensure_ascii=False, indent=2)
            
            print(f"\n  ✓ 已保存: {file_name}")
            processed_count += 1
            
            # 每处理完一个事件后暂停一下，避免请求过快
            time.sleep(1)
            
        except Exception as e:
            print(f"  处理 {order_book_id} 时发生错误: {e}")
            import traceback
            # 将错误信息写入文件
            error_file = os.path.join(SAVE_PATH, f"ERROR_{info_date_str.replace('-', '')}_{order_book_id[:6]}.txt")
            with open(error_file, "w", encoding="utf-8") as f:
                f.write(f"股票代码: {order_book_id}\n")
                f.write(f"公告日期: {info_date_str}\n")
                f.write(f"错误信息: {e}\n")
                f.write("\n完整错误栈:\n")
                traceback.print_exc(file=f)
            print(f"  错误详情已保存到: {error_file}")
            continue

print(f"\n{'='*60}")
print(f"处理完成!")
print(f"总股权激励事件数: {total_events}")
print(f"已处理并保存: {processed_count}")
print(f"{'='*60}")
