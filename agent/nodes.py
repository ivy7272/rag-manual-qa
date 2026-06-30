# -*- coding: utf-8 -*-
"""LangGraph Agent 节点函数：路由、检索、评估、重写、生成"""

import re
from typing import Dict, Any

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from config import GRADE_CONTEXT_MAX_LEN, CHAT_HISTORY_MSG_MAX_LEN
from agent.state import AgentState
from retrieval.hybrid_search import hybrid_search
from retrieval.keyword_search import search_index_entries
from knowledge_base.vector_store import fetch_docs_by_chunk_uids
from knowledge_base.index_manager import load_index_entries
from llm.unified_llm import create_llm


# ==================== 提示词模板 ====================

RAG_PROMPT_TEMPLATE = """
你是一位专业的用户手册分析专家和问题解决助手，你能够根据用户的问题准确的帮助他们查询到用户手册中相关的内容。
下面提供的是依据混合检索定位到的最相关分区。请基于这些块回答用户问题。

【使用要求】
1. 必须以证据为基础回答；
2. 若依据不足，请明确指出并建议应该查找的目录标签；
3. 在答案末尾列出参考的目录标签（一级 > 二级 > 最小）；
4. 如果对话历史中有上下文（如"第二种"、"刚才说的"等指代），请结合历史理解用户意图。

---
{chat_history}
---
【参考分区内容】：
{context}
---
【用户问题】：
{question}

【请给出专业分析与结论】："""


GRADE_PROMPT_TEMPLATE = """你是一个检索质量评估专家。请判断以下检索到的文档块是否与用户问题相关。

【对话历史】（帮助你理解用户问题的上下文，特别是追问和指代）：
{chat_history}
---
用户问题：{question}

检索到的内容：
{context}

请回答：
1. 这些内容是否足以回答用户问题？（是/否）
2. 如果不足，缺少什么关键信息？
3. 如果不足，建议用什么关键词重新检索？（给出3-5个关键词，用逗号分隔）

请严格按以下格式回答：
相关性: [是/否]
缺少信息: [如果足够则写"无"]
建议关键词: [如果需要重试则写关键词，否则写"无"]
"""

REWRITE_PROMPT_TEMPLATE = """你是一个查询优化专家。用户的问题未能检索到足够相关的内容。

【对话历史】（帮助你理解用户意图和上下文）：
{chat_history}
---
原始问题：{question}
之前使用的关键词：{old_keywords}
检索评估反馈：{grade_feedback}

请生成3-5个更精准的检索关键词（用逗号分隔），这些关键词应该：
1. 更贴近用户手册中可能使用的专业术语
2. 避免过于宽泛或模糊的词汇
3. 考虑同义词和缩写

只输出关键词列表，用逗号分隔，不要输出其他内容。"""

CHAT_SYSTEM_PROMPT = """你是一个『大模型用户手册智能助手』，
负责友好地回答用户的打招呼、身份询问、使用方式等通用问题。
不要主动提到自己背后的具体模型或服务器，只说自己是"智能助手"即可。

如果对话历史中有之前的交流，请自然地参考上下文，保持对话连贯性。"""


# ==================== 工具函数 ====================

def format_chat_history(messages: list, max_rounds: int = 5) -> str:
    """
    将 Streamlit 的 chat messages 格式化为 Prompt 中的对话历史文本

    :param messages: [{"role":"user/assistant", "content":"..."}, ...]
    :param max_rounds: 最多保留的对话轮次（每轮含 user + assistant）
    :return: 格式化后的历史文本，若无历史则返回空提示
    """
    if not messages:
        return "【对话历史】\n（无历史对话，这是本次会话的第一个问题）"

    # 取最近 N 轮（每轮 2 条：user + assistant）
    max_messages = max_rounds * 2
    recent = messages[-max_messages:]

    lines = ["【对话历史】"]
    for msg in recent:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        # 截断过长的单条消息
        if len(content) > CHAT_HISTORY_MSG_MAX_LEN:
            content = content[:CHAT_HISTORY_MSG_MAX_LEN] + " ..."
        label = "用户" if role == "user" else "助手"
        lines.append(f"{label}: {content}")

    return "\n".join(lines)


# ==================== 节点函数 ====================

def route_node(state: AgentState) -> AgentState:
    """
    路由节点：判断走 RAG 检索还是纯 LLM 闲聊
    增强：识别对话历史中的上下文，判断是否为追问/澄清

    路由决策矩阵：
    | 条件 | 路由 |
    |------|------|
    | 明显闲聊/身份声明（你好/我是谁/我叫...） | LLM |
    | 有历史 + 短句 + 含指代词（那/详细/第二种...） | RAG（追问） |
    | 有历史 + 短句 + 无指代词（我是马云/你是谁） | LLM（新话题） |
    | 长句（>6字） | RAG |
    | 无历史 + 短句 | LLM |
    """
    question = state["question"]
    q = (question or "").lower().strip()
    chat_history = state.get("chat_history", [])

    # 闲聊 / 身份声明模式
    small_talk_patterns = [
        "你好", "您好", "在吗",
        "你是谁", "who are you",
        "hello", "hi", "hey",
        "我是谁", "我叫", "我是", "我叫",
        "谢谢", "感谢", "再见", "拜拜",
    ]
    is_small_talk = any(p in q for p in small_talk_patterns)

    # 对话上下文感知
    has_history = len(chat_history) >= 2
    is_too_short = len(q) <= 6

    # 有历史上下文的短问题：区分"追问"与"新话题"
    if has_history and is_too_short:
        anaphora_patterns = [
            # 指代词 → 明确指向上一轮内容
            "那", "这个", "那个", "它", "这",
            "第二种", "第一种", "第三种", "上面", "前面", "刚才",
            # 追问/深化 → 对上一轮内容的延续
            "详细", "具体", "继续", "接着", "然后",
            "还有", "其他", "另外", "还有吗",
            # 疑问词 → 对上一轮内容的质疑/追问
            "怎么", "如何", "为什么", "步骤", "流程", "方法",
        ]
        is_followup = any(p in q for p in anaphora_patterns)
        if is_followup:
            state["route"] = "rag"
            state["status_message"] = "🔍 检测到追问，结合对话历史启动检索..."
            return state
        else:
            # 有历史但无指代 → 可能是新话题/身份声明/无关闲聊
            # 如"我是马云"、"我是谁"、"今天天气" — 不应走 RAG
            state["route"] = "llm"
            state["status_message"] = "💬 检测到新话题或通用对话，直接回答..."
            return state

    if is_small_talk or (is_too_short and not has_history):
        state["route"] = "llm"
        state["status_message"] = "💬 检测到通用对话，直接回答..."
    else:
        state["route"] = "rag"
        state["status_message"] = "🔍 检测到专业问题，启动混合检索..."

    return state


def retrieve_node(state: AgentState) -> AgentState:
    """
    检索节点：执行混合检索（关键词 + 向量）+ 向量库回表
    如果 retry_count > 0 且 search_keywords 非空，则使用重写后的关键词

    向量检索优先使用 Chroma ANN 索引（O(log n)），
    当向量库不可用时回退到手动 O(n) 余弦相似度计算。
    """
    question = state["question"]
    # 如果是重试，使用重写后的关键词拼接
    search_query = question
    if state["retry_count"] > 0 and state.get("search_keywords"):
        search_query = f"{question} {state['search_keywords']}"
        state["status_message"] = f"🔄 第 {state['retry_count']} 次重试检索，使用扩展关键词..."

    # 加载索引
    entries = load_index_entries(state["kb_name"])
    if not entries:
        state["status_message"] = "⚠️ 知识库索引为空，无法检索。"
        state["top_items"] = []
        state["retrieved_docs"] = []
        return state

    # 初始化向量库（用于 Chroma ANN 索引加速向量检索）
    vs = None
    try:
        from knowledge_base.vector_store import build_vectorstore_for_kb
        from embeddings.manager import create_embeddings
        embeddings = create_embeddings()
        vs = build_vectorstore_for_kb(embeddings, state["kb_name"])
    except Exception:
        vs = None  # 回退到手动余弦相似度

    # 混合检索
    top_items = hybrid_search(
        entries=entries,
        query=search_query,
        top_k=state["top_k"],
        fuzzy_th=state["fuzzy_th"],
        require_all=state.get("require_all", False),
        min_cn_len=state.get("min_cn_len", 2),
        vs=vs,
    )

    state["top_items"] = top_items

    if not top_items:
        state["status_message"] = "⚠️ 未检索到相关内容。"
        state["retrieved_docs"] = []
        return state

    state["status_message"] = f"✅ 检索完成，命中 {len(top_items)} 个相关分区。"

    return state


def grade_node(state: AgentState) -> AgentState:
    """
    评估节点：用 LLM 判断检索结果是否充足
    如果不足且未超过最大重试次数，标记 needs_retry=True
    """
    top_items = state.get("top_items", [])
    if not top_items:
        state["needs_retry"] = True
        state["status_message"] = "⚠️ 无检索结果，将尝试重写查询..."
        state["answer"] = "没有检索到和你问题明显相关的内容。\n可以尝试：减少关键词、降低相似度阈值，或改写问题再试一次。"
        return state

    # 拼接检索内容用于评估
    ctx_parts = []
    for i, item in enumerate(top_items, 1):
        label = item.get("label", "（无标签）")
        kw_text = item.get("keywords_text", "")
        ctx_parts.append(f"[{i}] 目录：{label} | 关键词：{kw_text}")

    context = "\n".join(ctx_parts)

    # 用 LLM 评估相关性
    llm = create_llm(
        provider=state.get("llm_provider", "ollama"),
        model_name=state["selected_model"],
        openai_api_key=state.get("openai_api_key", ""),
        openai_base_url=state.get("openai_base_url", ""),
        ollama_base_url=state.get("ollama_base_url", ""),
    )
    chat_history_str = format_chat_history(state.get("chat_history", []))
    grade_prompt = GRADE_PROMPT_TEMPLATE.format(
        question=state["question"],
        chat_history=chat_history_str,
        context=context[:GRADE_CONTEXT_MAX_LEN],  # 限制长度
    )

    try:
        grade_result = llm._call(grade_prompt)
    except Exception:
        # LLM 调用失败，默认为相关，避免无限重试
        state["needs_retry"] = False
        state["status_message"] = "⚠️ 相关性评估失败，直接生成回答..."
        return state

    # 解析评估结果
    is_relevant = "相关性: 是" in grade_result or "相关性：是" in grade_result
    state["needs_retry"] = not is_relevant and state["retry_count"] < state["max_retries"]

    if state["needs_retry"]:
        state["retry_count"] += 1
        state["status_message"] = f"⚠️ 检索内容相关性不足，准备第 {state['retry_count']} 次重试..."
        # 提取建议关键词
        kw_match = re.search(r'建议关键词[：:]\s*(.+)', grade_result)
        if kw_match:
            state["search_keywords"] = kw_match.group(1).strip()
    else:
        state["status_message"] = "✅ 检索内容相关性达标，开始生成回答..."

    return state


def rewrite_node(state: AgentState) -> AgentState:
    """
    查询重写节点：用 LLM 生成更精准的检索关键词
    """
    llm = create_llm(
        provider=state.get("llm_provider", "ollama"),
        model_name=state["selected_model"],
        openai_api_key=state.get("openai_api_key", ""),
        openai_base_url=state.get("openai_base_url", ""),
        ollama_base_url=state.get("ollama_base_url", ""),
    )
    chat_history_str = format_chat_history(state.get("chat_history", []))
    rewrite_prompt = REWRITE_PROMPT_TEMPLATE.format(
        question=state["question"],
        chat_history=chat_history_str,
        old_keywords=state.get("search_keywords", "无"),
        grade_feedback=state.get("status_message", ""),
    )

    try:
        new_keywords = llm._call(rewrite_prompt)
        state["search_keywords"] = new_keywords.strip()
        state["status_message"] = f"🔄 查询已重写，新关键词: {new_keywords.strip()}"
    except Exception:
        state["search_keywords"] = state["question"]
        state["status_message"] = "⚠️ 查询重写失败，使用原始查询重试..."

    return state


def generate_rag_node(state: AgentState) -> AgentState:
    """
    RAG 生成节点：基于检索到的文档块生成回答（流式输出在 Streamlit 层处理）
    这里生成 prompt，实际的流式输出由 UI 层的 answer_with_rag_stream 处理
    """
    top_items = state.get("top_items", [])

    if not top_items:
        state["answer"] = (
            "没有检索到和你问题明显相关的内容。\n"
            "可以尝试：减少关键词、降低相似度阈值，或改写问题再试一次。"
        )
        state["status_message"] = "⚠️ 无检索结果，无法生成 RAG 回答。"
        return state

    state["status_message"] = "📝 正在基于检索结果生成回答..."
    return state


def generate_chat_node(state: AgentState) -> AgentState:
    """
    闲聊生成节点：纯 LLM 回答，不走检索
    实际的流式输出由 UI 层处理
    """
    state["status_message"] = "💬 正在生成回复..."
    return state
