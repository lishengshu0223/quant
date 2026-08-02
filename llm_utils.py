"""
项目级 LLM 调用工具模块

封装 DeepSeek 和 阿里百炼(DashScope) 两个大模型 API 的调用，
各子项目通过 sys.path.insert(0, r'f:\\quant') 后直接 import 使用。

用法:
    import sys
    sys.path.insert(0, r'f:\\quant')
    from llm_utils import call_deepseek, call_dashscope, extract_json

    # 调用 DeepSeek
    result = call_deepseek("分析这段文本...", system_prompt="你是金融分析师")

    # 调用阿里百炼
    result = call_dashscope("分析这段文本...", system_prompt="你是证券分析师")

    # 从返回结果中提取 JSON
    data = extract_json(result)
"""

import os
import re
import json
import time

import requests
from dotenv import load_dotenv
from loguru import logger

# 加载项目根目录 .env
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

# ============ 配置 ============

# DeepSeek 配置
try:
    DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
except KeyError:
    from config import DEEPSEEK_API_KEY

DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

# 阿里百炼 DashScope 配置
try:
    DASHSCOPE_API_KEY = os.environ["DASHSCOPE_API_KEY"]
except KeyError:
    from config import DASHSCOPE_API_KEY

DASHSCOPE_BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL",
    "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)
DASHSCOPE_MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen3.6-flash")
# 备用模型（主模型不可用时自动切换）
DASHSCOPE_MODEL_FALLBACK = "qwen3.6-flash"


# ============ DeepSeek 调用 ============

def call_deepseek(
    user_content,
    system_prompt="你是一个专业的金融分析师。",
    model=None,
    temperature=0.0,
    max_tokens=8000,
    top_p=0.1,
    max_retries=5,
    timeout=300,
):
    """
    调用 DeepSeek 大模型（OpenAI 兼容格式，使用 requests）

    Args:
        user_content: 用户消息内容
        system_prompt: 系统提示词
        model: 模型名称，默认使用环境变量 DEEPSEEK_MODEL
        temperature: 温度参数
        max_tokens: 最大输出 token 数
        top_p: top_p 参数
        max_retries: 最大重试次数
        timeout: 请求超时秒数

    Returns:
        str: 模型回复内容

    Raises:
        Exception: 所有重试均失败时抛出
    """
    model = model or DEEPSEEK_MODEL
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{DEEPSEEK_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            if not content or not content.strip():
                raise ValueError("模型返回内容为空")
            return content

        except requests.exceptions.HTTPError as e:
            last_error = f"HTTP错误: {e.response.status_code} - {e.response.text[:300]}"
            logger.warning(f"DeepSeek调用失败(第{attempt+1}次): {last_error}")
        except requests.exceptions.Timeout:
            last_error = "请求超时"
            logger.warning(f"DeepSeek调用超时(第{attempt+1}次)")
        except requests.exceptions.SSLError as e:
            last_error = f"SSL错误: {e}"
            logger.warning(f"DeepSeek SSL错误(第{attempt+1}次)")
        except Exception as e:
            last_error = f"调用异常: {e}"
            logger.warning(f"DeepSeek调用异常(第{attempt+1}次): {last_error}")

        if attempt < max_retries - 1:
            wait = 5 * (attempt + 1)
            time.sleep(wait)

    raise Exception(f"DeepSeek调用失败，已重试{max_retries}次。最后错误: {last_error}")


# ============ 阿里百炼 DashScope 调用 ============

def call_dashscope(
    user_content,
    system_prompt="你是一个专业的证券分析师。",
    model=None,
    temperature=0.0,
    max_tokens=4096,
    top_p=0.7,
    max_retries=5,
    timeout=300,
    enable_fallback=True,
):
    """
    调用阿里百炼 DashScope 大模型（OpenAI 兼容格式）

    支持主/备模型自动切换：主模型不可用(400/404)时自动回退到备用模型。

    Args:
        user_content: 用户消息内容
        system_prompt: 系统提示词
        model: 模型名称，默认使用环境变量 DASHSCOPE_MODEL
        temperature: 温度参数
        max_tokens: 最大输出 token 数
        top_p: top_p 参数
        max_retries: 每个模型的最大重试次数
        timeout: 请求超时秒数
        enable_fallback: 是否启用备用模型回退

    Returns:
        str: 模型回复内容

    Raises:
        Exception: 所有模型和重试均失败时抛出
    """
    primary_model = model or DASHSCOPE_MODEL
    if enable_fallback and primary_model != DASHSCOPE_MODEL_FALLBACK:
        models_to_try = [primary_model, DASHSCOPE_MODEL_FALLBACK]
    else:
        models_to_try = [primary_model]

    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }

    last_error = None
    for current_model in models_to_try:
        payload = {
            "model": current_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }

        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    f"{DASHSCOPE_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                if not content or not content.strip():
                    raise ValueError("模型返回内容为空")
                if current_model != primary_model:
                    logger.info(f"DashScope使用备用模型成功: {current_model}")
                return content

            except requests.exceptions.HTTPError as e:
                code = e.response.status_code if e.response is not None else -1
                last_error = f"HTTP错误: {code} - {e.response.text[:300] if e.response else ''}"
                logger.warning(f"DashScope调用失败(模型:{current_model}, 第{attempt+1}次): {last_error}")
                # 模型不可用，直接切换
                if code in (400, 404):
                    break
            except requests.exceptions.Timeout:
                last_error = "请求超时"
                logger.warning(f"DashScope调用超时(模型:{current_model}, 第{attempt+1}次)")
            except requests.exceptions.SSLError as e:
                last_error = f"SSL错误: {e}"
                logger.warning(f"DashScope SSL错误(模型:{current_model}, 第{attempt+1}次)")
            except Exception as e:
                last_error = f"调用异常: {e}"
                logger.warning(f"DashScope调用异常(模型:{current_model}, 第{attempt+1}次): {last_error}")

            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                time.sleep(wait)

        # 当前模型失败，尝试下一个
        if current_model != models_to_try[-1]:
            logger.warning(f"模型 {current_model} 失败，切换到备用模型 {models_to_try[-1]}")

    raise Exception(f"DashScope调用失败，所有模型均已尝试。最后错误: {last_error}")


# ============ 通用 messages 接口 ============

def call_deepseek_messages(
    messages,
    model=None,
    temperature=0.0,
    max_tokens=8000,
    top_p=0.1,
    max_retries=5,
    timeout=300,
):
    """
    调用 DeepSeek，接受完整的 messages 列表（适合多轮对话或自定义消息结构）

    Args:
        messages: OpenAI 格式的消息列表 [{"role": ..., "content": ...}, ...]
        其余参数同 call_deepseek

    Returns:
        str: 模型回复内容
    """
    model = model or DEEPSEEK_MODEL
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{DEEPSEEK_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            if not content or not content.strip():
                raise ValueError("模型返回内容为空")
            return content
        except Exception as e:
            last_error = str(e)
            logger.warning(f"DeepSeek messages调用失败(第{attempt+1}次): {last_error}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))

    raise Exception(f"DeepSeek调用失败，已重试{max_retries}次。最后错误: {last_error}")


def call_dashscope_messages(
    messages,
    model=None,
    temperature=0.0,
    max_tokens=4096,
    top_p=0.7,
    max_retries=5,
    timeout=300,
    enable_fallback=True,
):
    """
    调用阿里百炼 DashScope，接受完整的 messages 列表

    Args:
        messages: OpenAI 格式的消息列表 [{"role": ..., "content": ...}, ...]
        其余参数同 call_dashscope

    Returns:
        str: 模型回复内容
    """
    primary_model = model or DASHSCOPE_MODEL
    if enable_fallback and primary_model != DASHSCOPE_MODEL_FALLBACK:
        models_to_try = [primary_model, DASHSCOPE_MODEL_FALLBACK]
    else:
        models_to_try = [primary_model]

    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }

    last_error = None
    for current_model in models_to_try:
        payload = {
            "model": current_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }

        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    f"{DASHSCOPE_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                if not content or not content.strip():
                    raise ValueError("模型返回内容为空")
                return content
            except requests.exceptions.HTTPError as e:
                code = e.response.status_code if e.response is not None else -1
                last_error = f"HTTP {code}"
                if code in (400, 404):
                    break
            except Exception as e:
                last_error = str(e)
                logger.warning(f"DashScope messages调用失败(模型:{current_model}, 第{attempt+1}次): {last_error}")

            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))

        if current_model != models_to_try[-1]:
            logger.warning(f"模型 {current_model} 失败，切换备用模型")

    raise Exception(f"DashScope调用失败。最后错误: {last_error}")


# ============ JSON 提取工具 ============

def extract_json(text):
    """
    从模型回复中健壮地提取 JSON 对象/数组

    支持:
    - 直接 JSON 文本
    - ```json ... ``` 代码块包裹
    - 文本中嵌入的 JSON（括号配平扫描）
    - 尾随逗号修复

    Args:
        text: 模型返回的原始文本

    Returns:
        dict 或 list: 解析后的 JSON 对象

    Raises:
        ValueError: 无法解析时抛出
    """
    if not text:
        raise ValueError("输入文本为空")

    text = text.strip()

    # 1. 直接解析
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2. 提取 ```json ... ``` 代码块
    m = re.search(r"```(?:json)?\s*([\{\[].*?[\}\]])\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            # 尝试修复尾随逗号
            fixed = re.sub(r",\s*([}\]])", r"\1", m.group(1))
            try:
                return json.loads(fixed)
            except Exception:
                pass

    # 3. 括号配平扫描（支持 {} 和 []）
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        if start < 0:
            continue
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == start_char:
                    depth += 1
                elif ch == end_char:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i + 1]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                            try:
                                return json.loads(fixed)
                            except Exception:
                                break

    # 4. 最后尝试修复整个文本
    try:
        return json.loads(re.sub(r",\s*([}\]])", r"\1", text))
    except Exception:
        pass

    raise ValueError(f"无法从模型输出中解析出JSON, 原文前200字: {text[:200]}")
