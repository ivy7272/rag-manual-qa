# -*- coding: utf-8 -*-
"""统一 LLM 接口：支持 Ollama 和 OpenAI 两种后端，提供一致的调用接口"""

import os
from typing import Optional, List

from langchain_openai import ChatOpenAI

from config import (
    OLLAMA_BASE_URL,
    OPENAI_DEFAULT_BASE_URL,
    DEFAULT_TEMPERATURE,
    DEFAULT_REQUEST_TIMEOUT,
)
from llm.custom_ollama import CustomOllamaLLM


class UnifiedLLM:
    """
    统一 LLM 包装器，对外提供一致的 ._call() 和 .stream() 接口

    支持两种后端：
    - ollama: 本地 Ollama 服务（通过 CustomOllamaLLM）
    - openai: OpenAI 兼容 API（通过 ChatOpenAI，支持自定义 base_url）
    """

    def __init__(
        self,
        provider: str = "ollama",
        model_name: str = "deepseek-r1:32b",
        temperature: float = DEFAULT_TEMPERATURE,
        request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
    ):
        """
        :param provider: "ollama" | "openai"
        :param model_name: 模型名称
        :param temperature: 温度参数
        :param request_timeout: 请求超时时间
        :param openai_api_key: OpenAI API Key（默认从环境变量 OPENAI_API_KEY 获取）
        :param openai_base_url: OpenAI API 地址（默认 https://api.openai.com/v1）
        :param ollama_base_url: Ollama 服务地址（默认使用 config.OLLAMA_BASE_URL）
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.temperature = temperature

        if self.provider == "ollama":
            self._llm = CustomOllamaLLM(
                model_name=model_name,
                temperature=temperature,
                base_url=ollama_base_url or OLLAMA_BASE_URL,
                request_timeout=request_timeout,
            )
        elif self.provider == "openai":
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
            base_url = openai_base_url or OPENAI_DEFAULT_BASE_URL
            self._llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                openai_api_key=api_key,
                base_url=base_url,
                request_timeout=request_timeout,
            )
        else:
            raise ValueError(f"不支持的 LLM 后端: {provider}，可选: ollama, openai")

    def _call(self, prompt: str) -> str:
        """
        非流式调用：一次性返回完整回答

        :param prompt: 提示词文本
        :return: 模型生成的完整回答
        """
        if self.provider == "ollama":
            return self._llm._call(prompt)
        else:
            # ChatOpenAI.invoke() 返回 AIMessage，需要取 .content
            response = self._llm.invoke(prompt)
            return response.content or ""

    def stream(self, prompt: str):
        """
        流式调用：逐段返回模型生成的文本

        :param prompt: 提示词文本
        :yield: 每段生成的文本（str）
        """
        if self.provider == "ollama":
            yield from self._llm.stream(prompt)
        else:
            # ChatOpenAI.stream() 返回 AIMessageChunk 迭代器
            for chunk in self._llm.stream(prompt):
                if chunk.content:
                    yield chunk.content

    @property
    def _llm_type(self) -> str:
        return f"unified_llm_{self.provider}"


def create_llm(
    provider: str = "ollama",
    model_name: str = "",
    openai_api_key: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    ollama_base_url: Optional[str] = None,
    **kwargs,
) -> UnifiedLLM:
    """
    工厂函数：根据后端类型创建 UnifiedLLM 实例

    :param provider: "ollama" | "openai"
    :param model_name: 模型名称
    :param openai_api_key: OpenAI API Key
    :param openai_base_url: OpenAI API 地址
    :param ollama_base_url: Ollama 服务地址（默认使用 config.OLLAMA_BASE_URL）
    :param kwargs: 其他参数传递给 UnifiedLLM
    :return: UnifiedLLM 实例
    """
    return UnifiedLLM(
        provider=provider,
        model_name=model_name,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        ollama_base_url=ollama_base_url,
        **kwargs,
    )
