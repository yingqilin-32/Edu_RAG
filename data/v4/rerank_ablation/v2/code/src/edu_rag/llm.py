"""One bounded DeepSeek call with safe errors; no retries or SQL execution."""

import httpx

from edu_rag.config import ConfigurationError, get_settings


class ModelError(RuntimeError):
    pass


def complete(messages, *, json_mode=False, max_tokens=1024, settings=None, client=None):
    settings = settings if settings is not None else get_settings()
    key = settings.deepseek_api_key.get_secret_value().strip()
    if not key:
        raise ConfigurationError("请在本机 .env 配置 DEEPSEEK_API_KEY。")
    body = {"model": settings.deepseek_model, "messages": messages,
            "thinking": {"type": "disabled"}, "max_tokens": max_tokens, "stream": False}
    if json_mode:
        body.update(response_format={"type": "json_object"}, temperature=0)
    owned = client is None
    if owned:
        client = httpx.Client(timeout=httpx.Timeout(settings.deepseek_timeout, connect=10), follow_redirects=False)
    try:
        response = client.post(settings.deepseek_base_url.rstrip("/") + "/chat/completions",
                               headers={"Authorization": "Bearer " + key}, json=body)
        response.raise_for_status()
        payload = response.json()
        if payload["choices"][0].get("finish_reason") != "stop":
            raise ModelError("模型未完整生成内容，请缩短问题或重试。")
        content = payload["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ModelError("模型没有返回有效内容。")
        usage = payload.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        return content, {name: usage.get(name, 0) for name in ("prompt_tokens", "completion_tokens", "total_tokens")}
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        descriptions = {401: "密钥无效或未授权", 402: "API 余额不足", 403: "API 访问被拒绝", 429: "请求过于频繁"}
        raise ModelError("DeepSeek 调用失败：" + descriptions.get(code, f"服务返回 HTTP {code}") + "。") from None
    except httpx.RequestError:
        raise ModelError("DeepSeek 请求超时或网络不可达，请稍后重试。") from None
    except (KeyError, IndexError, TypeError, ValueError):
        raise ModelError("DeepSeek 返回了无法解析的响应。") from None
    finally:
        if owned:
            client.close()
