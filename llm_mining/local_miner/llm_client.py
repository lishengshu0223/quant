"""
本地化因子挖掘项目 - LLM 客户端

双提供商支持:
- 主模型: DeepSeek deepseek-v4-flash(固定, OpenAI兼容端点 https://api.deepseek.com)
- 备用模型: 阿里百炼 qwen3.6-flash(DashScope兼容端点, 仅DeepSeek完全不可用时兜底)
- 递增等待重试 + 健壮的 JSON 提取
"""

import json
import re
import time

import requests

from . import console
from .config import (
    DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,
    OPENCODE_API_KEY, OPENCODE_BASE_URL,
)


class LLMError(Exception):
    pass


def _provider_route(model: str, cfg, is_primary: bool = True) -> tuple:
    """返回 (provider名, base_url, api_key, 是否OpenAI兼容无思考参数)。
    cfg.llm_provider=opencode 时仅主模型走 OpenCode 网关(备用模型按名前缀回退到百炼/DeepSeek);
    其余 provider 显式指定则固定走对应端点; auto=按模型名前缀。"""
    provider = (cfg.llm_provider or "auto").lower()
    if provider == "opencode" and is_primary:
        return ("opencode", OPENCODE_BASE_URL, OPENCODE_API_KEY, True)
    if provider == "deepseek":
        return ("deepseek", DEEPSEEK_BASE_URL, DEEPSEEK_API_KEY, True)
    if provider == "dashscope":
        return ("dashscope", DASHSCOPE_BASE_URL, DASHSCOPE_API_KEY, False)
    # auto: 按模型名前缀
    if str(model).lower().startswith("deepseek"):
        return ("deepseek", DEEPSEEK_BASE_URL, DEEPSEEK_API_KEY, True)
    return ("dashscope", DASHSCOPE_BASE_URL, DASHSCOPE_API_KEY, False)


def call_llm(messages: list, cfg) -> dict:
    """
    调用大模型, 返回 {"content": 正式回复, "thinking": 思考内容, "model": 实际模型}

    通道策略(用户硬性要求):
    - llm_provider == "opencode": 唯一通道 = Opencode 网关(主模型 deepseek-v4-flash)。
      任何暂时性故障(网络/限流/超时/空内容/HTTP 4xx5xx)一律在网关通道内无限等待重试,
      直到调用成功为止; 绝不切换到 DeepSeek 直连或百炼兜底。
      重试等待时间按 8 秒起步递增, 封顶 300 秒, 避免无意义的高频轮询。
    - 其余 provider: 按原逻辑多通道依次尝试(主模型直连 -> 备用模型), 每通道内递增等待重试。
    """
    opencode_only = (cfg.llm_provider or "auto").lower() == "opencode"
    if opencode_only:
        # 只保留 Opencode 网关一个通道, 失败就等, 直到成功
        models = [(cfg.model_primary, True)]
    else:
        models = [(cfg.model_primary, False)]
        models.append((cfg.model_fallback, False))
    last_error = None

    for idx, (model, via_opencode) in enumerate(models):
        provider, base_url, api_key, no_thinking = _provider_route(model, cfg, is_primary=via_opencode)
        if not api_key:
            console.log(f"    [LLM] 模型{model}({provider}) 缺少API Key, 跳过。")
            last_error = "缺少API Key"
            continue
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": cfg.temperature,
            "top_p": 0.8,
            "max_tokens": cfg.max_tokens,
        }
        # 深度思考参数仅 DashScope(百炼)支持; DeepSeek/OpenCode 为 OpenAI 兼容端点, 不传该参数
        if (not no_thinking) and cfg.enable_thinking:
            payload["enable_thinking"] = True
            payload["thinking_budget"] = cfg.thinking_budget

        for attempt in range(cfg.llm_max_retry if not opencode_only else 10**9):
            try:
                resp = requests.post(
                    f"{base_url}/chat/completions",
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
                console.log(f"    [LLM] 模型{model}({provider}) 第{attempt+1}次调用失败: {last_error}")
                # 模型不可用/配额耗尽(429 quota): 重试也不会成功, 直接换通道
                if not opencode_only and (code in (400, 404) or (code == 429 and "quota" in last_error.lower())):
                    break
            except (requests.exceptions.Timeout, requests.exceptions.SSLError) as e:
                last_error = f"网络异常: {e}"
                console.log(f"    [LLM] 模型{model}({provider}) 第{attempt+1}次调用失败: {last_error}")
            except LLMError as e:
                last_error = str(e)
                console.log(f"    [LLM] 模型{model}({provider}) 第{attempt+1}次调用失败: {last_error}")
            except Exception as e:
                last_error = f"未知异常: {e}"
                console.log(f"    [LLM] 模型{model}({provider}) 第{attempt+1}次调用失败: {last_error}")

            if opencode_only:
                # 用户硬性要求: 链接暂时失误就等待直到成功, 不切通道
                wait = min(8 * (attempt + 1), 300)
                console.log(f"    [LLM] {provider} 网关暂时不可用, {wait}秒后重试(等待直到成功)...")
                time.sleep(wait)
            elif attempt < cfg.llm_max_retry - 1:
                wait = 8 * (attempt + 1)
                console.log(f"    [LLM] {wait}秒后重试...")
                time.sleep(wait)

        if not opencode_only:
            console.log(f"    [LLM] 通道 {model}({provider}) 全部重试失败, 切换下一通道...")

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
