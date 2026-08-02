import os
import re
import json
import time
import requests
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime, timedelta

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

from openai import OpenAI


def decode_sse_cookie(arg1):
    """解码上交所反爬cookie（与extract_increase_amount.py一致）"""
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
    """下载PDF文件，增加上交所特殊处理和多种下载策略"""
    headers_list = [
        {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/pdf',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1',
        },
        {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Referer': 'https://www.sse.com.cn/',
            'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'no-cors',
            'Sec-Fetch-Site': 'same-site',
        },
        {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Accept': 'application/pdf',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        },
    ]
    
    # 策略1: 直接下载（非上交所链接）
    if 'sse.com.cn' not in url:
        for headers in headers_list:
            try:
                response = requests.get(url, timeout=timeout, headers=headers, stream=True, allow_redirects=True)
                response.raise_for_status()
                
                content = response.content
                if content[:4] == b'%PDF':
                    print(f"  ✓ 直接下载成功")
                    return content
                    
                content_type = response.headers.get('Content-Type', '')
                if 'pdf' in content_type.lower():
                    return content
            except:
                continue
    
    # 策略2: 上交所特殊处理 - 使用完整会话管理
    if 'sse.com.cn' in url:
        try:
            import urllib.parse
            
            session = requests.Session()
            session.headers.update(headers_list[0])
            
            # 步骤1: 先访问上交所首页获取初始cookies
            try:
                home_response = session.get('https://www.sse.com.cn/', timeout=10)
                print(f"  获取首页cookies成功")
            except:
                print(f"  获取首页cookies失败，继续尝试")
            
            # 步骤2: 访问PDF链接（可能会被重定向到static.sse.com.cn）
            response = session.get(url, timeout=timeout, allow_redirects=True)
            content = response.content
            
            # 检查是否是PDF
            if content[:4] == b'%PDF':
                print(f"  ✓ 上交所PDF下载成功")
                return content
            
            # 步骤3: 如果不是PDF，检查是否需要处理JS反爬
            html_content = response.text
            if 'acw_sc__v2' in html_content or 'document.location.reload' in html_content:
                # 提取arg1参数
                arg1_match = re.search(r"var\s+arg1\s*=\s*['\"]([^'\"]+)['\"]", html_content)
                if arg1_match:
                    arg1 = arg1_match.group(1)
                    print(f"  检测到上交所JS反爬，正在生成cookie...")
                    
                    # 使用破解反爬手段生成cookie（与extract_increase_amount.py一致）
                    cookie_value = decode_sse_cookie(arg1)
                    print(f"  生成cookie: acw_sc__v2={cookie_value}")
                    
                    # 添加cookie后重新请求
                    session.cookies.set('acw_sc__v2', cookie_value)
                    
                    response2 = session.get(url, timeout=timeout, allow_redirects=True)
                    if response2.content[:4] == b'%PDF':
                        print(f"  ✓ 上交所反爬破解成功，PDF下载成功")
                        return response2.content
                    else:
                        print(f"  设置cookie后仍未获取到PDF")
            
            # 策略3: 尝试构建上交所PDF的直接下载链接
            parsed = urllib.parse.urlparse(url)
            
            # 格式1: static.sse.com.cn
            if parsed.path.endswith('.pdf'):
                path = parsed.path.lstrip('/')
                static_url = f"http://static.sse.com.cn/{path}"
                try:
                    print(f"  尝试直接请求static.sse.com.cn: {static_url[:80]}...")
                    response = session.get(static_url, timeout=15, allow_redirects=True)
                    content = response.content
                    if content[:4] == b'%PDF':
                        print(f"  ✓ static.sse.com.cn下载成功")
                        return content
                except:
                    pass
            
            # 策略4: 使用curl命令下载
            try:
                import subprocess
                import tempfile
                
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
                    temp_path = f.name
                
                curl_cmd = f'curl -s -L -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" "{url}" -o "{temp_path}"'
                subprocess.run(curl_cmd, shell=True, timeout=30)
                
                if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                    with open(temp_path, 'rb') as f:
                        content = f.read()
                    os.unlink(temp_path)
                    if content[:4] == b'%PDF':
                        print(f"  ✓ curl命令下载成功")
                        return content
                else:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
            except Exception as e:
                print(f"  curl命令下载失败: {e}")
            
        except Exception as e:
            print(f"  上交所下载处理失败: {e}")
    
    print(f"  下载失败: 所有策略均无法获取PDF")
    return None


def extract_text_from_pdf(pdf_content, max_chars=500000):
    """从PDF二进制内容中提取文本，增加多种解析策略和长度限制"""
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
        print(f"  解析PDF策略1失败: {e}")
    
    # 策略2: 尝试使用pdfplumber（如果安装了）
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
                print("  使用pdfplumber解析成功")
                return text
    except ImportError:
        pass
    except Exception as e:
        print(f"  解析PDF策略2(pdfplumber)失败: {e}")
    
    # 策略3: 尝试直接读取文本（某些PDF可能是纯文本编码）
    try:
        text = pdf_content.decode('utf-8', errors='ignore')
        if len(text) > 100 and 'PDF' in text[:100]:
            return text[:max_chars]
    except Exception as e:
        pass
    
    print(f"  所有解析策略均失败")
    return ""


def filter_incentive_announcements(announcements_df):
    """
    筛选与股权激励相关的公告。
    使用宽松的正则匹配策略:宁可保留不相干公告,也不漏掉相关公告。
    """
    if announcements_df is None or announcements_df.empty:
        return announcements_df

    incentive_keywords = [
        '激励', '期权', '限制性股票', '股票增值权',
        '股权激励', '员工持股', '绩效考核', '业绩考核',
        '激励计划', '行权', '授予',
        '激励对象', '解锁', '归属',
        '激励基金', '激励方案',
    ]

    pattern = '|'.join(incentive_keywords)

    mask = announcements_df['title'].str.contains(pattern, na=False, regex=True)
    filtered = announcements_df[mask].copy()

    print(f"  [公告筛选] 原始{len(announcements_df)}条 -> 保留{len(filtered)}条")
    return filtered


def prioritize_announcements(announcements_df):
    """
    优先选择关键公告，减少token消耗。
    按优先级排序：草案 > 摘要 > 考核管理办法 > 其他
    """
    if announcements_df is None or announcements_df.empty:
        return announcements_df
    
    priority_keywords = [
        ('草案', 5),       # 优先级最高
        ('摘要', 4),       # 摘要
        ('考核管理办法', 3), # 考核办法
        ('激励对象名单', 2), # 名单
        ('法律意见', 1),    # 法律意见
        ('自查表', 0),     # 自查表
    ]
    
    def get_priority(title):
        for keyword, priority in priority_keywords:
            if keyword in title:
                return priority
        return -1
    
    announcements_df['priority'] = announcements_df['title'].apply(get_priority)
    # 按优先级降序排序，取前5条
    prioritized = announcements_df.sort_values('priority', ascending=False).head(5)
    
    print(f"  [公告优先级] 原始{len(announcements_df)}条 -> 优先{len(prioritized)}条")
    return prioritized


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
    
    # 修复括号问题 - 将圆括号转换为方括号（处理元组格式）
    result = re.sub(r'\(', '[', result)
    result = re.sub(r'\)', ']', result)
    
    # 修复字符串中的转义问题
    result = re.sub(r'\\([^nrtbf\\"])', r'\\\1', result)
    
    # 确保引号正确
    result = re.sub(r"'([^']+)'", r'"\1"', result)
    
    return result.strip()


def validate_llm_fields(parsed):
    """
    校验LLM返回的字段是否符合要求，并进行规范化处理。
    返回校验后的结果和是否需要重试的标志。
    """
    required_fields = [
        "考核年限", "考核等级数量", "是否排除实控人/5%股东",
        "参与人数", "核心技术/业务骨干人数", "高管人数",
        "高管激励份额比例", "所用于激励的股票来源", "考核条件",
        "考核科目"
    ]
    
    # 检查是否缺少必要字段
    missing_fields = []
    for field in required_fields:
        if field not in parsed:
            missing_fields.append(field)
            parsed[field] = None
    
    if missing_fields:
        print(f"  LLM返回缺少字段: {missing_fields}")
    
    # 校验考核科目格式
    exam_subjects = parsed.get("考核科目", [])
    if not isinstance(exam_subjects, list):
        print(f"  考核科目格式错误: 应为列表，实际类型为 {type(exam_subjects)}")
        parsed["考核科目"] = []
        return parsed, True  # 需要重试
    
    # 校验每个考核科目的字段
    valid_subjects = []
    has_invalid_subject = False
    
    for i, subject in enumerate(exam_subjects):
        if not isinstance(subject, dict):
            print(f"  第{i+1}个考核科目格式错误: 应为字典")
            has_invalid_subject = True
            continue
        
        # 检查必要字段
        required_subject_fields = ["财务科目名", "一致预期名", "数值"]
        missing_subject_fields = []
        for sf in required_subject_fields:
            if sf not in subject:
                missing_subject_fields.append(sf)
        
        if missing_subject_fields:
            print(f"  第{i+1}个考核科目缺少字段: {missing_subject_fields}")
            has_invalid_subject = True
            continue
        
        # 检查数值格式
        values = subject.get("数值", [])
        if not isinstance(values, list):
            print(f"  第{i+1}个考核科目数值格式错误: 应为列表")
            has_invalid_subject = True
            continue
        
        valid_subjects.append(subject)
    
    parsed["考核科目"] = valid_subjects
    
    # 判断是否需要重试
    need_retry = len(missing_fields) > 2 or (has_invalid_subject and len(valid_subjects) == 0)
    
    return parsed, need_retry


def analyze_with_llm(info_date_str: str, texts: list[str], prompt: str, model_name: str, client: OpenAI):
    """
    调用DeepSeek LLM分析公告内容，提取结构化激励信息。
    增加重试机制、字段校验和错误处理，确保解析结果稳定。
    """
    # 合并文本，限制总长度
    max_total_chars = 600000  # 降低上限，约45万token，避免超限
    combined_text = ""
    for i, text in enumerate(texts):
        # 检查单篇文本是否异常大（可能是解析错误）
        if len(text) > 150000:
            print(f"  警告: 第{i+1}篇文本过长({len(text)}字符)，截断处理")
            text = text[:150000]
        
        if len(combined_text) + len(text) + 100 > max_total_chars:
            remaining = max_total_chars - len(combined_text)
            combined_text += "\n\n--- 公告分割线 ---\n\n" + text[:remaining]
            print(f"  文本总长度已达上限，截断处理")
            break
        combined_text += "\n\n--- 公告分割线 ---\n\n" + text
    
    if not combined_text.strip():
        print("  公告文本为空，跳过LLM分析")
        return None
    
    full_prompt = f"当前公告日期是{info_date_str}\n\n" + prompt
    full_prompt += combined_text

    print(f"  总输入长度: {len(full_prompt)} 字符")

    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            print(f"  正在调用DeepSeek分析... (第{attempt+1}/{max_retries}次)")
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "你是一个专业的金融分析师，擅长分析股权激励公告。你必须只返回合法的JSON格式，不要包含任何其他文字。"},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.0,
                max_tokens=8000,
                top_p=0.1,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            result = response.choices[0].message.content
            
            if not result:
                print(f"  LLM返回空内容，第{attempt+1}次重试...")
                time.sleep(2)
                continue
            
            print(f"  LLM响应长度: {len(result)} 字符")
            
            # 清理和修复JSON格式
            result = fix_json_format(result)
            
            # 尝试解析JSON
            try:
                parsed = json.loads(result)
                print(f"  LLM解析成功")
                
                # 校验和规范化字段
                validated, need_retry = validate_llm_fields(parsed)
                
                if need_retry and attempt < max_retries - 1:
                    print(f"  LLM返回字段不完整，准备重试...")
                    time.sleep(3)
                    continue
                
                return validated
                
            except json.JSONDecodeError as e:
                print(f"  JSON解析失败: {e}")
                print(f"  原始响应前500字符: {result[:500]}")
                
                # 尝试进一步修复
                try:
                    # 移除多余的逗号
                    result = re.sub(r',\s*([}\]])', r'\1', result)
                    # 修复未闭合的引号
                    result = re.sub(r'"([^"]*)$', r'"\1"', result)
                    parsed = json.loads(result)
                    print(f"  JSON修复后解析成功")
                    
                    # 校验和规范化字段
                    validated, need_retry = validate_llm_fields(parsed)
                    
                    if need_retry and attempt < max_retries - 1:
                        print(f"  LLM返回字段不完整，准备重试...")
                        time.sleep(3)
                        continue
                    
                    return validated
                except json.JSONDecodeError:
                    last_error = f"JSON解析失败: {e}"
                    print(f"  第{attempt+1}次解析失败，准备重试...")
                    time.sleep(3)
                    continue
                    
        except Exception as e:
            last_error = f"调用DeepSeek失败: {e}"
            print(f"  {last_error}")
            
            # 检查是否是token超限错误
            if "maximum context length" in str(e).lower() or "tokens" in str(e).lower() and ("limit" in str(e).lower() or "exceed" in str(e).lower()):
                print(f"  Token超限错误，尝试减少文本长度后重试...")
                # 进一步减少文本长度
                max_total_chars = int(max_total_chars * 0.7)
                combined_text = combined_text[:max_total_chars]
                full_prompt = f"当前公告日期是{info_date_str}\n\n" + prompt
                full_prompt += combined_text
                print(f"  调整后总输入长度: {len(full_prompt)} 字符")
                
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
            
            if attempt < max_retries - 1:
                print(f"  第{attempt+1}次调用失败，准备重试...")
                time.sleep(5)
                continue
    
    print(f"  LLM分析失败，已重试{max_retries}次")
    print(f"  最后错误: {last_error}")
    return None


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
