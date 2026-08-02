"""
本地化因子挖掘项目 - LLM 客户端

参考 research/holding_increase/func_tools.py 的阿里 token plan 调用方式:
- requests 直接请求 OpenAI 兼容端点
- 主模型 qwen3.8-max-preview(开启深度思考, 预算适度), 失败回退 qwen3.6-flash
- 递增等待重试
- 健壮的 JSON 提取
"""

import json
import re
import time

import requests

from . import console
from .config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL


class LLMError(Exception):
    pass


def call_llm(messages: list, cfg) -> dict:
    """
    调用大模型, 返回 {"content": 正式回复, "thinking": 思考内容, "model": 实际模型}
    """
    if not DASHSCOPE_API_KEY:
        raise LLMError("未找到 DASHSCOPE_API_KEY, 请检查 f:\\quant\\.env")

    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }

    models = [cfg.model_primary, cfg.model_fallback]
    last_error = None

    for model in models:
        is_primary = (model == cfg.model_primary)
        payload = {
            "model": model,
            "messages": messages,
            "temperature": cfg.temperature,
            "top_p": 0.8,
            "max_tokens": cfg.max_tokens,
        }
        # 主模型开启深度思考; 备用flash模型不传思考参数以保证兼容
        if is_primary and cfg.enable_thinking:
            payload["enable_thinking"] = True
            payload["thinking_budget"] = cfg.thinking_budget

        for attempt in range(cfg.llm_max_retry):
            try:
                resp = requests.post(
                    f"{DASHSCOPE_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=cfg.llm_timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                message = data["choices"][0]["message"]
                content = message.get("content") or ""
                thinking = message.get("reasoning_content") or ""
                if not content.strip():
                    raise LLMError("模型返回内容为空")
                return {"content": content, "thinking": thinking, "model": model}

            except requests.exceptions.HTTPError as e:
                code = e.response.status_code if e.response is not None else -1
                body = e.response.text[:300] if e.response is not None else ""
                last_error = f"HTTP {code}: {body}"
                console.log(f"    [LLM] 模型{model} 第{attempt+1}次调用失败: {last_error}")
                if code in (400, 404):  # 模型不可用, 直接换模型
                    break
            except (requests.exceptions.Timeout, requests.exceptions.SSLError) as e:
                last_error = f"网络异常: {e}"
                console.log(f"    [LLM] 模型{model} 第{attempt+1}次调用失败: {last_error}")
            except LLMError as e:
                last_error = str(e)
                console.log(f"    [LLM] 模型{model} 第{attempt+1}次调用失败: {last_error}")
            except Exception as e:
                last_error = f"未知异常: {e}"
                console.log(f"    [LLM] 模型{model} 第{attempt+1}次调用失败: {last_error}")

            if attempt < cfg.llm_max_retry - 1:
                wait = 8 * (attempt + 1)
                console.log(f"    [LLM] {wait}秒后重试...")
                time.sleep(wait)

        console.log(f"    [LLM] 模型 {model} 全部重试失败, 切换备用模型...")

    raise LLMError(f"所有模型均调用失败。最后错误: {last_error}")


def extract_json(text: str) -> dict:
    """从模型回复中健壮地提取 JSON 对象"""
    text = text.strip()

    # 1. 直接解析
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2. 提取 ```json ... ``` 代码块
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # 3. 括号配平扫描第一个完整 JSON 对象
    start = text.find("{")
    if start >= 0:
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
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i + 1]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            # 4. 修复尾随逗号后再试
                            fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                            try:
                                return json.loads(fixed)
                            except Exception:
                                break
        # 5. 最后尝试修复整个文本中的尾随逗号
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", text))
        except Exception:
            pass

    raise ValueError(f"无法从模型输出中解析出JSON, 原文前200字: {text[:200]}")
