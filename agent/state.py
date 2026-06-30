# -*- coding: utf-8 -*-
"""LangGraph Agent 状态定义"""

from typing import List, Dict, Any, TypedDict, Optional

from langchain_core.documents import Document


class AgentState(TypedDict):
    """LangGraph Agent 的全局状态"""

    # === 用户输入 ===
    question: str                              # 用户当前问题
    chat_history: List[Dict[str, str]]         # 对话历史 [{"role":"user","content":...}, ...]

    # === 知识库上下文 ===
    kb_name: str                               # 当前选中的知识库名称
    selected_model: str                        # 当前选中的 LLM 模型
    llm_provider: str                          # LLM 后端: "ollama" | "openai"
    openai_api_key: str                        # OpenAI API Key（Ollama 时可为空）
    openai_base_url: str                       # OpenAI API Base URL
    ollama_base_url: str                       # Ollama 服务地址

    # === 检索结果 ===
    retrieved_docs: List[Document]             # 检索到的文档块（向量库回表后的完整内容）
    top_items: List[Dict[str, Any]]            # 检索到的 index.json 条目
    hybrid_scores: Dict[str, float]            # 混合检索评分 {chunk_uid: score}

    # === 路由 ===
    route: str                                 # "llm" | "rag"

    # === 重试控制（ReAct 循环） ===
    needs_retry: bool                          # 是否需要重写查询后重试
    retry_count: int                           # 当前重试次数
    max_retries: int                           # 最大重试次数（默认 2）
    search_keywords: str                       # 重写后的搜索关键词

    # === 输出 ===
    answer: str                                # 最终回答
    status_message: str                        # 状态提示信息

    # === 检索参数 ===
    top_k: int                                 # Top-K 数量
    fuzzy_th: float                            # 模糊匹配阈值
    min_cn_len: int                            # 中文短词最小长度
    require_all: bool                          # ALL 模式

    # === 图片映射 ===
    image_maps: Dict[str, Dict[str, str]]      # 图片映射 {文档基名: {图片ID: 路径}}


def create_initial_state(
    question: str,
    kb_name: str,
    selected_model: str,
    llm_provider: str = "ollama",
    openai_api_key: str = "",
    openai_base_url: str = "",
    ollama_base_url: str = "",
    top_k: int = 3,
    fuzzy_th: float = 0.55,
    min_cn_len: int = 2,
    require_all: bool = False,
    image_maps: Optional[Dict[str, Dict[str, str]]] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> AgentState:
    """创建初始 Agent 状态"""
    return AgentState(
        question=question,
        chat_history=chat_history or [],
        kb_name=kb_name,
        selected_model=selected_model,
        llm_provider=llm_provider,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        ollama_base_url=ollama_base_url,
        retrieved_docs=[],
        top_items=[],
        hybrid_scores={},
        route="rag",               # 默认走 RAG
        needs_retry=False,
        retry_count=0,
        max_retries=2,
        search_keywords="",
        answer="",
        status_message="",
        top_k=top_k,
        fuzzy_th=fuzzy_th,
        min_cn_len=min_cn_len,
        require_all=require_all,
        image_maps=image_maps or {},
    )
