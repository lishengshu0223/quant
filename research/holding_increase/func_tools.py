import os
import re
import json
import time
import threading
import requests
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime, timedelta

from loguru import logger

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

import rqdatac

# 本地API（无连接数限制，线程安全）
import sys
sys.path.insert(0, r'f:\quant')
from local_api import init as _local_init, get_price as _local_get_price, get_next_trading_date as _local_get_next_td
_local_init()

# rqdatac非线程安全，且有连接数限制，所有rqdatac调用统一加锁
RQDATAC_LOCK = threading.Lock()

# 阿里百炼 Token Plan 配置
try:
    DASHSCOPE_API_KEY = os.environ["DASHSCOPE_API_KEY"]
except KeyError:
    from config import DASHSCOPE_API_KEY

DASHSCOPE_BASE_URL = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
# 优先使用qwen3.8-max-preview，不可用时回退到qwen3.6-flash
DASHSCOPE_MODEL_PRIMARY = "qwen3.8-max-preview"
DASHSCOPE_MODEL_FALLBACK = "qwen3.6-flash"
DASHSCOPE_MODEL = DASHSCOPE_MODEL_PRIMARY  # 默认使用主模型

# 配置日志
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
logger.add(
    os.path.join(log_dir, "holding_increase_{time:YYYYMMDD}.log"),
    rotation="1 day",
    retention="7 days",
    level="DEBUG",
)


def fix_json_format(result):
    """修复常见的JSON格式问题"""
    if not result:
        return result

    result = result.strip()

    # 移除markdown代码块标记
    if result.startswith('```json'):
        result = result[7:]
        if result.endswith('```'):
            result = result[:-3]
    elif result.startswith('```'):
        result = result[3:]
        if result.endswith('```'):
            result = result[:-3]

    result = result.strip()

    # 修复括号问题 - 将圆括号转换为方括号
    result = re.sub(r'\(', '[', result)
    result = re.sub(r'\)', ']', result)

    # 修复字符串中的转义问题
    result = re.sub(r'\\([^nrtbf\\"])', r'\\\1', result)

    # 确保引号正确
    result = re.sub(r"'([^']+)'", r'"\1"', result)

    return result.strip()


def convert_to_json_serializable(obj):
    """将对象转换为JSON可序列化格式"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif isinstance(obj, (datetime, pd.Timestamp)):
        return obj.strftime('%Y-%m-%d')
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


def call_llm(messages, model_name=None, temperature=0.0, max_tokens=4096):
    """
    调用阿里百炼大模型接口
    优先使用主模型(DASHSCOPE_MODEL_PRIMARY)，失败后回退到备用模型(DASHSCOPE_MODEL_FALLBACK)
    """
    # 确定要尝试的模型列表
    if model_name is not None:
        # 显式指定了模型，只使用该模型
        models_to_try = [model_name]
    else:
        # 默认：先试主模型，失败后试备用模型
        models_to_try = [DASHSCOPE_MODEL_PRIMARY, DASHSCOPE_MODEL_FALLBACK]

    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }

    max_retries = 5
    last_error = None

    for current_model in models_to_try:
        payload = {
            "model": current_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.7,
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{DASHSCOPE_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=300
                )
                response.raise_for_status()

                result = response.json()
                content = result["choices"][0]["message"]["content"]

                if current_model != DASHSCOPE_MODEL:
                    logger.info(f"LLM调用成功(模型: {current_model})")
                return content

            except requests.exceptions.HTTPError as e:
                last_error = f"HTTP错误: {e.response.status_code} - {e.response.text}"
                logger.warning(f"LLM调用失败(模型:{current_model}, 第{attempt+1}次): {last_error}")

                # 404/400类错误说明模型不可用，直接换模型
                if e.response.status_code in (400, 404):
                    logger.warning(f"模型 {current_model} 不可用，尝试切换备用模型")
                    break

                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                    continue

            except requests.exceptions.SSLError as e:
                last_error = f"SSL错误: {e}"
                logger.warning(f"LLM调用SSL错误(模型:{current_model}, 第{attempt+1}次): {e}")

                if attempt < max_retries - 1:
                    time.sleep(10 * (attempt + 1))
                    continue

            except requests.exceptions.Timeout as e:
                last_error = f"请求超时: {e}"
                logger.warning(f"LLM调用超时(模型:{current_model}, 第{attempt+1}次): {last_error}")

                if attempt < max_retries - 1:
                    wait_time = 15 * (attempt + 1)
                    logger.info(f"等待{wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue

            except Exception as e:
                last_error = f"调用异常: {e}"
                logger.warning(f"LLM调用失败(模型:{current_model}, 第{attempt+1}次): {last_error}")

                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue

        # 当前模型所有重试失败，尝试下一个模型
        if current_model != models_to_try[-1]:
            logger.warning(f"模型 {current_model} 全部重试失败，切换到备用模型 {models_to_try[-1]}")

    raise Exception(f"LLM调用失败，所有模型均已尝试。最后错误: {last_error}")


def decode_sse_cookie(arg1):
    """解码上交所反爬cookie"""
    pos_list = [0xf, 0x23, 0x1d, 0x18, 0x21, 0x10, 0x1, 0x26, 0xa, 0x9, 0x13, 0x1f, 0x28, 0x1b, 0x16, 0x17, 0x19, 0xd, 0x6, 0xb, 0x27, 0x12, 0x14, 0x8, 0xe, 0x15, 0x20, 0x1a, 0x2, 0x1e, 0x7, 0x4, 0x11, 0x5, 0x3, 0x1c, 0x22, 0x25, 0xc, 0x24]
    mask = '3000176000856006061501533003690027800375'

    output_list = [''] * len(pos_list)
    for i in range(len(arg1)):
        for j in range(len(pos_list)):
            if pos_list[j] == i + 1:
                output_list[j] = arg1[i]

    arg2 = ''.join(output_list)

    arg3 = ''
    for i in range(0, min(len(arg2), len(mask)), 2):
        str_char = int(arg2[i:i+2], 16)
        mask_char = int(mask[i:i+2], 16)
        xor_char = str_char ^ mask_char
        xor_char_hex = hex(xor_char)[2:]
        if len(xor_char_hex) == 1:
            xor_char_hex = '0' + xor_char_hex
        arg3 += xor_char_hex

    return arg3


def download_pdf(url, timeout=30):
    """下载PDF文件，支持上交所特殊处理"""
    headers_list = [
        {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/pdf',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        },
        {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Referer': 'https://www.sse.com.cn/',
        },
    ]

    # 非上交所链接：直接下载
    if 'sse.com.cn' not in url:
        for headers in headers_list:
            try:
                response = requests.get(url, timeout=timeout, headers=headers, stream=True, allow_redirects=True)
                response.raise_for_status()

                content = response.content
                if content[:4] == b'%PDF':
                    return content

                content_type = response.headers.get('Content-Type', '')
                if 'pdf' in content_type.lower():
                    return content
            except:
                continue

    # 上交所链接：特殊处理
    if 'sse.com.cn' in url:
        try:
            import urllib.parse

            session = requests.Session()
            session.headers.update(headers_list[0])

            # 先访问上交所首页获取初始cookies
            try:
                session.get('https://www.sse.com.cn/', timeout=10)
            except:
                pass

            # 访问PDF链接
            response = session.get(url, timeout=timeout, allow_redirects=True)
            content = response.content

            if content[:4] == b'%PDF':
                return content

            # 检查JS反爬
            html_content = response.text
            if 'acw_sc__v2' in html_content or 'document.location.reload' in html_content:
                arg1_match = re.search(r"var\s+arg1\s*=\s*['\"]([^'\"]+)['\"]", html_content)
                if arg1_match:
                    arg1 = arg1_match.group(1)
                    cookie_value = decode_sse_cookie(arg1)
                    session.cookies.set('acw_sc__v2', cookie_value)

                    response2 = session.get(url, timeout=timeout, allow_redirects=True)
                    if response2.content[:4] == b'%PDF':
                        return response2.content

            # 尝试static.sse.com.cn
            parsed = urllib.parse.urlparse(url)
            if parsed.path.endswith('.pdf'):
                path = parsed.path.lstrip('/')
                static_url = f"http://static.sse.com.cn/{path}"
                try:
                    response3 = session.get(static_url, timeout=15, allow_redirects=True)
                    if response3.content[:4] == b'%PDF':
                        return response3.content
                except:
                    pass

        except Exception as e:
            logger.warning(f"上交所PDF下载处理失败: {e}")

    return None


def extract_text_from_pdf(pdf_content, max_chars=500000):
    """从PDF二进制内容中提取文本"""
    if not pdf_content:
        return ""

    # 策略1: 使用pypdf/PyPDF2
    try:
        reader = PdfReader(BytesIO(pdf_content))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
                if len(text) > max_chars:
                    text = text[:max_chars]
                    break
        if text:
            return text
    except Exception as e:
        logger.warning(f"解析PDF策略1失败: {e}")

    # 策略2: 尝试使用pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(BytesIO(pdf_content)) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
                    if len(text) > max_chars:
                        text = text[:max_chars]
                        break
            if text:
                return text
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"解析PDF策略2(pdfplumber)失败: {e}")

    # 策略3: 尝试直接读取文本
    try:
        text = pdf_content.decode('utf-8', errors='ignore')
        if len(text) > 100 and 'PDF' in text[:100]:
            return text[:max_chars]
    except:
        pass

    return ""


def filter_increase_announcements(announcements_df, filter_prompt):
    """
    使用大模型筛选高管/大股东增持预告公告
    返回筛选后的DataFrame
    """
    if announcements_df is None or announcements_df.empty:
        return pd.DataFrame()

    # 提取公告标题列表
    titles = announcements_df['title'].tolist()
    titles_text = "\n".join([f"{i+1}. {t}" for i, t in enumerate(titles)])

    # 构建消息
    messages = [
        {"role": "system", "content": "你是一个专业的证券分析师，擅长筛选上市公司公告。你必须只返回合法的JSON格式。"},
        {"role": "user", "content": filter_prompt.format(titles=titles_text)}
    ]

    # 调用LLM筛选
    try:
        llm_result = call_llm(messages, temperature=0.0)

        # 修复JSON格式
        llm_result = fix_json_format(llm_result)

        parsed = json.loads(llm_result)

        # 解析筛选结果
        is_increase = parsed.get("is_increase_announcement", False)
        if not is_increase:
            logger.info(f"公告筛选结果: 无高管增持预告")
            return pd.DataFrame()

        # 获取LLM返回的相关公告索引
        relevant_indices = parsed.get("relevant_indices", [])
        if not relevant_indices:
            # 如果没有指定索引，返回全部（兜底）
            logger.info(f"公告筛选结果: 有增持预告，但未指定具体公告，返回全部")
            return announcements_df

        # 根据索引筛选（索引从1开始）
        valid_indices = [i-1 for i in relevant_indices if 1 <= i <= len(announcements_df)]
        filtered = announcements_df.iloc[valid_indices]

        logger.info(f"公告筛选结果: 有高管增持预告, 筛选出{len(filtered)}条")
        return filtered

    except Exception as e:
        logger.error(f"LLM公告筛选失败: {e}")
        # 如果LLM筛选失败，使用关键词进行兜底筛选
        return keyword_filter_announcements(announcements_df)


def keyword_filter_announcements(announcements_df):
    """
    使用关键词兜底筛选高管/大股东增持公告
    """
    increase_keywords = [
        '增持', '高管增持', '股东增持', '拟增持', '计划增持',
        '预增持', '增持方案', '增持计划',
    ]

    # 排除股权激励相关
    exclude_keywords = [
        '股权激励', '激励计划', '员工持股', '股权激励计划',
        '限制性股票', '股票期权',
    ]

    title_col = 'title'
    mask = announcements_df[title_col].str.contains('|'.join(increase_keywords), na=False, regex=True)

    # 排除股权激励
    exclude_mask = announcements_df[title_col].str.contains('|'.join(exclude_keywords), na=False, regex=True)

    filtered = announcements_df[mask & ~exclude_mask]

    logger.info(f"关键词筛选: 原始{len(announcements_df)}条 -> 保留{len(filtered)}条")
    return filtered


def analyze_announcement_structured(text, analysis_prompt):
    """
    对公告文本进行结构化分析
    """
    # 限制文本长度
    max_text_len = 200000
    if len(text) > max_text_len:
        text = text[:max_text_len]
        logger.warning(f"公告文本过长，已截断至{max_text_len}字符")

    messages = [
        {"role": "system", "content": "你是一个专业的证券分析师，擅长分析高管增持公告。你必须只返回合法的JSON格式，不要包含任何其他文字。"},
        {"role": "user", "content": analysis_prompt.format(text=text)}
    ]

    max_retries = 3
    for attempt in range(max_retries):
        try:
            llm_result = call_llm(messages, temperature=0.0)

            # 修复JSON格式
            llm_result = fix_json_format(llm_result)

            parsed = json.loads(llm_result)
            return parsed

        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败 (第{attempt+1}次): {e}")
            time.sleep(2)
            if attempt < max_retries - 1:
                continue

        except Exception as e:
            logger.warning(f"LLM分析失败 (第{attempt+1}次): {e}")
            time.sleep(2)
            if attempt < max_retries - 1:
                continue

    logger.error(f"公告结构化分析失败，已重试{max_retries}次")
    return None


def get_stock_price(order_book_id, date_str, adjust_type='none'):
    """
    获取股票收盘价（使用本地API，无连接数限制）
    """
    try:
        price_data = _local_get_price(
            order_book_id,
            start_date=date_str,
            end_date=date_str,
            fields=['close'],
            adjust_type=adjust_type
        )
        if price_data is not None and len(price_data) > 0:
            return float(price_data['close'].iloc[0])
        return None
    except Exception as e:
        logger.error(f"获取股票价格失败 {order_book_id} {date_str}: {e}")
        return None


def get_total_market_cap(order_book_id, date_str):
    """
    获取公司总市值（收盘价 * 总股本）
    收盘价用本地API，总股本用rqdatac（重试3次）
    """
    try:
        # 获取收盘价（本地API）
        price_data = _local_get_price(
            order_book_id,
            start_date=date_str,
            end_date=date_str,
            fields=['close']
        )
        if price_data is None or price_data.empty:
            logger.warning(f"获取价格失败: {order_book_id}")
            return None

        close_price = price_data['close'].iloc[0]

        # 获取总股本（rqdatac，加锁+重试）
        for retry in range(3):
            try:
                with RQDATAC_LOCK:
                    shares_data = rqdatac.get_shares(
                        order_book_id,
                        start_date=date_str,
                        end_date=date_str
                    )
                break
            except Exception as e:
                if retry < 2:
                    time.sleep(2 + retry * 2)
                    continue
                logger.error(f"获取总股本失败(重试{retry+1}次) {order_book_id}: {e}")
                return None

        if shares_data is None or shares_data.empty:
            logger.warning(f"获取总股本失败: {order_book_id}")
            return None

        total_shares = shares_data['total'].iloc[0]

        # 计算市值
        market_cap = float(close_price) * float(total_shares)
        logger.info(f"总市值计算: {close_price:.2f} × {total_shares:,.0f} = {market_cap:,.0f}")
        return market_cap

    except Exception as e:
        logger.error(f"获取总市值失败 {order_book_id}: {e}")
        return None


def calculate_target_amount(structured_data, close_price):
    """
    计算增持目标金额
    """
    amount = structured_data.get('增持金额下限')
    amount_upper = structured_data.get('增持金额上限')
    quantity = structured_data.get('增持数量下限')
    quantity_upper = structured_data.get('增持数量上限')

    # 如果有金额数据
    if amount is not None and amount != 0:
        if amount_upper is not None and amount_upper != 0:
            # 有上下限取均值
            return (float(amount) + float(amount_upper)) / 2
        else:
            return float(amount)

    # 如果只有数量数据，用收盘价转换
    if quantity is not None and quantity != 0 and close_price:
        if quantity_upper is not None and quantity_upper != 0:
            # 数量有上下限取均值
            avg_quantity = (float(quantity) + float(quantity_upper)) / 2
            return avg_quantity * close_price
        else:
            return float(quantity) * close_price

    return None


def calculate_return_series(order_book_id, info_date_str, hold_days=120):
    """
    计算增持公告后的收益率序列（使用本地API，无连接数限制）
    """
    try:
        # 获取下一交易日（本地API）
        next_trade_date = _local_get_next_td(info_date_str)
        if next_trade_date is None:
            logger.error(f"无法获取下一交易日")
            return None

        next_trade_date_str = pd.Timestamp(next_trade_date).strftime('%Y-%m-%d')

        # 获取hold_days个交易日后的日期（本地API）
        end_date = _local_get_next_td(next_trade_date, n=hold_days)
        if end_date is None:
            logger.error(f"无法获取{hold_days}个交易日后的日期")
            return None

        end_date_str = pd.Timestamp(end_date).strftime('%Y-%m-%d')

        # 获取后复权开盘价序列（本地API）
        price_data = _local_get_price(
            order_book_id,
            start_date=next_trade_date_str,
            end_date=end_date_str,
            fields=['open'],
            adjust_type='post'
        )

        # price_data需要至少 hold_days+1 天（基准日 + hold_days个交易日）
        if price_data is None or len(price_data) < 2:
            logger.error(f"获取价格数据不足: {len(price_data) if price_data is not None else 0}条")
            return None

        # 以下一交易日(T+1)开盘价为基准（即买入价）
        base_price = price_data['open'].iloc[0]
        if base_price is None or base_price <= 0:
            logger.error(f"基准价格无效: {base_price}")
            return None

        # 计算hold_days个交易日的收益率序列
        # returns[0] = T+2/T+1 - 1（1日收益）
        # returns[1] = T+3/T+1 - 1（2日收益）
        # ...
        # returns[119] = T+121/T+1 - 1（120日收益）
        actual_days = min(hold_days, len(price_data) - 1)
        returns = []
        for i in range(1, actual_days + 1):
            day_price = price_data['open'].iloc[i]
            if day_price is None or day_price <= 0:
                returns.append(None)
            else:
                ret = (day_price / base_price) - 1.0
                returns.append(round(ret, 4))

        # 不足的天数用None补齐
        while len(returns) < hold_days:
            returns.append(None)

        logger.info(f"收益率序列计算完成: 共{len(returns)}天有效{sum(1 for r in returns if r is not None)}天")
        return returns

    except Exception as e:
        logger.error(f"计算收益率序列失败: {e}")
        return None


def get_specific_period_returns(return_series, periods=[1, 3, 5, 10, 22, 60, 90, 120]):
    """
    提取特定周期的收益率
    注意: return_series[0]是1日收益，return_series[1]是2日收益...
    所以第N日的收益率在索引N-1的位置
    """
    specific_returns = {}
    for period in periods:
        idx = period - 1
        if 0 <= idx < len(return_series) and return_series[idx] is not None:
            specific_returns[f"{period}日收益率"] = return_series[idx]
        else:
            specific_returns[f"{period}日收益率"] = None

    return specific_returns


def save_increase_event(data, save_dir, info_date_str, order_book_id):
    """
    保存增持事件为JSON文件
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 生成文件名: YYYYmmdd_xxxxxx.json
    file_name = f"{info_date_str.replace('-', '')}_{order_book_id[:6]}.json"
    file_path = os.path.join(save_dir, file_name)

    # 转换为JSON可序列化格式
    json_data = convert_to_json_serializable(data)

    # 保存JSON
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    logger.info(f"已保存: {file_name}")
    return file_path


def check_event_exists(save_dir, info_date_str, order_book_id):
    """
    检查事件是否已存在（避免重复处理）
    """
    file_name = f"{info_date_str.replace('-', '')}_{order_book_id[:6]}.json"
    file_path = os.path.join(save_dir, file_name)
    return os.path.exists(file_path)


def get_all_announcements_for_date(date_str, stock_list=None, batch_size=500):
    """
    获取全市场指定日期的所有公告（分批获取，避免API限制）
    
    Args:
        date_str: 日期字符串，格式 'YYYY-MM-DD'
        stock_list: 股票代码列表，如果为None则获取全部A股
        batch_size: 每批获取的股票数量
        
    Returns:
        DataFrame，包含所有公告
    """
    if stock_list is None:
        # 获取全部A股
        with RQDATAC_LOCK:
            instruments = rqdatac.all_instruments(type='CS', date=date_str)
        stock_list = instruments['order_book_id'].tolist()
    
    # 分批
    batches = [stock_list[i:i+batch_size] for i in range(0, len(stock_list), batch_size)]
    
    all_announcements = []
    failed_batches = 0
    
    for batch_idx, batch in enumerate(batches):
        if (batch_idx + 1) % 5 == 0:
            logger.info(f"  获取公告进度: {batch_idx+1}/{len(batches)} 批")
        
        # 每个批次最多重试3次
        success = False
        for retry in range(3):
            try:
                with RQDATAC_LOCK:
                    batch_ann = rqdatac.get_announcement(
                        order_book_ids=batch,
                        start_date=date_str,
                        end_date=date_str
                    )
                if batch_ann is not None and not batch_ann.empty:
                    all_announcements.append(batch_ann)
                success = True
                break
            except Exception as e:
                if "connection number exceeds" in str(e) or "timeout" in str(e).lower():
                    # 连接超限或超时，等待更长
                    time.sleep(3 + retry * 2)
                else:
                    time.sleep(1 + retry)
                if retry == 2:
                    failed_batches += 1
                    logger.warning(f"  批次 {batch_idx+1} 获取失败(重试{retry+1}次): {e}")
        
        # 限制API频率
        if (batch_idx + 1) % 10 == 0 and not success:
            time.sleep(1.0)
    
    if all_announcements:
        # 重置索引，将order_book_id从索引转为列
        reset_announcements = [df.reset_index() for df in all_announcements]
        combined = pd.concat(reset_announcements, ignore_index=True)
        logger.info(f"全市场公告获取完成: {len(combined)} 条, 失败{failed_batches}批")
        return combined
    else:
        if failed_batches > 0 and failed_batches >= len(batches):
            # 全部批次失败 → 返回None表示异常，调用方不标记日期为已处理
            logger.warning(f"全市场公告获取全部失败 ({failed_batches}/{len(batches)}批)，跳过该日期")
            return None
        # 确实没有公告（失败批次少，或零失败）→ 返回空DF
        logger.info(f"全市场公告获取完成: 0 条, 失败{failed_batches}批")
        return pd.DataFrame()
