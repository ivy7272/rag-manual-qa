# -*- coding: utf-8 -*-
"""自定义 Ollama LLM（继承 LangChain LLM 基类，支持流式与非流式调用）"""

import json as _json
from typing import List, Optional, Any

import requests
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun

from config import OLLAMA_BASE_URL, DEFAULT_TEMPERATURE, DEFAULT_REQUEST_TIMEOUT


class CustomOllamaLLM(LLM):
    """
    自定义 Ollama LLM 类（继承 LangChain 的 LLM 基类）
    支持两种调用方式：
    1. 非流式调用：一次性返回完整回答
    2. 流式调用：逐段返回回答（通过 stream() 方法）
    """
    model_name: str = "deepseek-r1:32b"
    temperature: float = DEFAULT_TEMPERATURE
    base_url: str = OLLAMA_BASE_URL
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT

    @property
    def _llm_type(self) -> str:
        return "custom_ollama_llm_via_requests"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """非流式调用：向 Ollama 发送请求，一次性获取完整回答"""
        api_url = f"{self.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature, "stop": stop},
        }
        try:
            resp = requests.post(api_url, json=payload, timeout=self.request_timeout)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"连接 Ollama 服务失败 ({api_url}): {e}")
        except Exception as e:
            raise RuntimeError(f"调用 Ollama 模型时发生未知错误: {e}")

    def stream(self, prompt: str, stop: Optional[List[str]] = None):
        """
        流式调用：向 Ollama 发送流式请求，逐段返回模型生成的文本
        :yield: 每段生成的文本
        """
        api_url = f"{self.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": self.temperature, "stop": stop},
        }
        with requests.post(api_url, json=payload, stream=True, timeout=self.request_timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = _json.loads(line)
                except Exception:
                    continue
                piece = obj.get("response")
                if piece:
                    yield piece
                if obj.get("done"):
                    break
