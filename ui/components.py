# -*- coding: utf-8 -*-
"""Streamlit UI 组件 — v2.0 优化版

优化点:
- 侧边栏重构：配置移至 st.sidebar，expander 分组折叠
- 原生 st.tabs() 替代 st.radio
- 消除双重检索：graph.stream() 的 retrieve_node 结果直接复用
- 停止生成按钮 + 部分回答保留
- 对话管理：清空历史 / 导出 Markdown
- 引用标注优化：内联引用 + 结构化参考面板
- 知识库构建：上传文件模式增加进度条
- 知识库管理：KB 统计信息卡片
- 自定义 CSS 主题
"""

import os
import re
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import numpy as np
import requests
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from config import (
    OLLAMA_BASE_URL, SHOW_K_DEFAULT, OPENAI_DEFAULT_BASE_URL,
    DEFAULT_TEMPERATURE, DEFAULT_OLLAMA_MODEL, SNIPPET_MAX_LEN,
)
from utils.file_utils import list_kbs, delete_kb, kb_dir
from embeddings.manager import create_embeddings, compute_cosine_similarity
from knowledge_base.index_manager import (
    load_index_entries, load_all_kb_summaries, load_kb_image_maps, load_prompt_hints,
)
from knowledge_base.manager import build_knowledge_base_from_files, build_knowledge_base_from_folders
from knowledge_base.vector_store import (
    build_vectorstore_for_kb, fetch_docs_by_chunk_uids,
)
from retrieval.hybrid_search import hybrid_search
from retrieval.keyword_search import search_index_entries
from llm.unified_llm import UnifiedLLM
from agent.state import create_initial_state
from agent.nodes import (
    route_node, retrieve_node, grade_node, rewrite_node,
    generate_rag_node, generate_chat_node,
    RAG_PROMPT_TEMPLATE, CHAT_SYSTEM_PROMPT, format_chat_history,
)
from agent.graph import compile_agent
from memory.checkpointer import create_sqlite_checkpointer, get_or_create_thread_id, load_chat_history, clear_chat_history
from rendering.renderer import render_text_with_images


# ═══════════════════════════════════════════════════════════════
# 缓存编译的 LangGraph 图（避免每条消息重复编译）
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def cached_compile_agent(_checkpointer):
    """缓存编译后的 LangGraph Agent 图（图结构不变，只需编译一次）"""
    return compile_agent(checkpointer=_checkpointer)


# ═══════════════════════════════════════════════════════════════
# 自定义 CSS
# ═══════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
    /* === 侧边栏 === */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fafafa 0%, #ffffff 100%);
    }
    section[data-testid="stSidebar"] .stExpander {
        border: 1px solid #e8e8e8;
        border-radius: 10px;
        margin: 6px 0;
    }
    section[data-testid="stSidebar"] .stExpander > div:first-child {
        font-weight: 600;
    }

    /* === 信息卡片 === */
    .kb-stat-card {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 18px 14px;
        text-align: center;
        transition: box-shadow 0.2s;
    }
    .kb-stat-card:hover {
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }
    .kb-stat-card .stat-number {
        font-size: 1.7rem;
        font-weight: 700;
        color: #1a73e8;
        line-height: 1.2;
    }
    .kb-stat-card .stat-desc {
        font-size: 0.78rem;
        color: #5f6368;
        margin-top: 4px;
    }

    /* === 引用标记 === */
    .cite-tag {
        display: inline-block;
        background: #e8f0fe;
        color: #1a73e8;
        border-radius: 3px;
        padding: 0px 5px;
        font-size: 0.68rem;
        font-weight: 700;
        vertical-align: super;
        margin: 0 1px;
    }

    /* === 按钮 === */
    .stButton > button {
        border-radius: 8px;
        transition: all 0.15s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    /* === 滚动聊天区 === */
    .chat-scroll-container {
        max-height: 55vh;
        overflow-y: auto;
        padding-right: 8px;
    }

    /* === 分割线 === */
    hr.thin-divider {
        border: none;
        border-top: 1px solid #e8e8e8;
        margin: 14px 0;
    }

    /* === KB 摘要框 === */
    .kb-summary-box {
        background: linear-gradient(135deg, #f8fbff 0%, #f0f4ff 100%);
        border-left: 3px solid #1a73e8;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 0.88rem;
        line-height: 1.65;
        color: #333;
        max-height: 200px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-break: break-word;
    }

    /* === Tab 标签 === */
    [data-testid="stTabs"] button {
        font-weight: 500;
        font-size: 0.95rem;
    }
</style>
"""


# ═══════════════════════════════════════════════════════════════
# 缓存函数
# ═══════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def get_available_ollama_models(ollama_url: str = "") -> List[str]:
    """获取 Ollama 服务中可用的模型列表（缓存 5 分钟）"""
    base_url = ollama_url or OLLAMA_BASE_URL
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]
    except Exception:
        return [DEFAULT_OLLAMA_MODEL]


@st.cache_resource(show_spinner="正在加载向量库...")
def init_content_vs_for_kb(kb_name: str):
    """初始化知识库的向量库（缓存实例）"""
    embeddings = create_embeddings()
    return build_vectorstore_for_kb(embeddings, kb_name)


@st.cache_data
def cached_load_all_kb_summaries() -> Dict[str, str]:
    """加载所有知识库总结"""
    return load_all_kb_summaries()


@st.cache_data
def cached_load_kb_image_maps(kb_name: str) -> Dict[str, Dict[str, str]]:
    """加载知识库图片映射"""
    return load_kb_image_maps(kb_name)


@st.cache_data(show_spinner=False)
def cached_load_index_entries(kb_name: str) -> List[Dict[str, Any]]:
    """加载知识库索引条目"""
    return load_index_entries(kb_name)


# ═══════════════════════════════════════════════════════════════
# 上下文构建（从检索结果提取文本）
# ═══════════════════════════════════════════════════════════════

def build_context_from_top_items(
    top_items: List[Dict[str, Any]],
    content_vs,
) -> Tuple[str, List[Document]]:
    """
    从 graph.stream() 的 retrieve_node 结果中：
    1. 回表拉取完整 Document
    2. 拼接成 RAG 上下文文本
    不执行检索 — 检索已由 graph.stream() 完成。
    """
    if not top_items:
        return "(无内容)", []

    uids = [it.get("chunk_uid", "") for it in top_items if it.get("chunk_uid")]
    top_docs = fetch_docs_by_chunk_uids(content_vs, uids)

    ctx_parts = []
    for i, doc in enumerate(top_docs, 1):
        path = doc.metadata.get("label", "（无标签）")
        src = doc.metadata.get("source", "未知来源")
        body_lines = [
            ln for ln in (doc.page_content or "").splitlines()
            if not ln.startswith("[LABEL]") and not ln.startswith("[KEYWORDS]")
        ]
        snippet = "\n".join(body_lines).strip()
        if len(snippet) > SNIPPET_MAX_LEN:
            snippet = snippet[:SNIPPET_MAX_LEN] + " ..."
        ctx_parts.append(f"[{i}] 目录：{path}｜来源：{src}\n{snippet}")

    context = "\n\n".join(ctx_parts) if ctx_parts else "(无内容)"
    return context, top_docs


# ═══════════════════════════════════════════════════════════════
# 流式回答
# ═══════════════════════════════════════════════════════════════

def _create_llm(provider: str, model: str, api_key: str, base_url: str,
                ollama_url: str = "") -> UnifiedLLM:
    """创建 UnifiedLLM 实例的快捷函数"""
    return UnifiedLLM(
        provider=provider,
        model_name=model,
        openai_api_key=api_key,
        openai_base_url=base_url,
        ollama_base_url=ollama_url,
    )


def stream_rag_answer(
    question: str,
    context: str,
    chat_history_str: str,
    llm_provider: str,
    selected_model: str,
    openai_api_key: str,
    openai_base_url: str,
    ollama_base_url: str = "",
) -> str:
    """
    基于已有上下文的 RAG 流式回答（不执行检索，检索由 graph.stream() 完成）
    支持停止生成信号。流式失败时自动回退到非流式调用。
    """
    prompt = RAG_PROMPT_TEMPLATE.format(
        chat_history=chat_history_str,
        context=context,
        question=question,
    )
    llm = _create_llm(llm_provider, selected_model, openai_api_key, openai_base_url, ollama_base_url)

    placeholder = st.empty()
    chunks: List[str] = []
    st.session_state._streaming = True
    st.session_state._partial_answer = ""

    try:
        for piece in llm.stream(prompt):
            if st.session_state.get("_stop_requested"):
                break
            chunks.append(piece)
            st.session_state._partial_answer = "".join(chunks)
            placeholder.markdown("".join(chunks))
    except Exception as e:
        # 流式失败时回退到非流式调用
        try:
            fallback_text = llm._call(prompt)
            if fallback_text:
                placeholder.markdown(fallback_text)
                st.session_state._partial_answer = fallback_text
                return fallback_text
            else:
                placeholder.error(f"模型返回空响应，请检查模型是否可用。\n错误: {e}")
                return ""
        except Exception as e2:
            placeholder.error(f"Ollama 连接失败，请检查服务是否运行。\n流式错误: {e}\n非流式错误: {e2}")
            return ""
    finally:
        st.session_state._streaming = False

    return "".join(chunks)


def stream_chat_answer(
    question: str,
    chat_history_str: str,
    llm_provider: str,
    selected_model: str,
    openai_api_key: str,
    openai_base_url: str,
    ollama_base_url: str = "",
) -> str:
    """纯 LLM 闲聊流式回答（含停止信号支持，流式失败时回退到非流式）"""
    full_prompt = CHAT_SYSTEM_PROMPT + f"\n\n{chat_history_str}\n\n用户：{question}"
    llm = _create_llm(llm_provider, selected_model, openai_api_key, openai_base_url, ollama_base_url)

    placeholder = st.empty()
    chunks: List[str] = []
    st.session_state._streaming = True
    st.session_state._partial_answer = ""

    try:
        for piece in llm.stream(full_prompt):
            if st.session_state.get("_stop_requested"):
                break
            chunks.append(piece)
            st.session_state._partial_answer = "".join(chunks)
            placeholder.markdown("".join(chunks))
    except Exception as e:
        # 流式失败时回退到非流式调用
        try:
            fallback_text = llm._call(full_prompt)
            if fallback_text:
                placeholder.markdown(fallback_text)
                st.session_state._partial_answer = fallback_text
                return fallback_text
            else:
                placeholder.error(f"模型返回空响应，请检查模型是否可用。\n错误: {e}")
                return ""
        except Exception as e2:
            placeholder.error(f"Ollama 连接失败，请检查服务是否运行。\n流式错误: {e}\n非流式错误: {e2}")
            return ""
    finally:
        st.session_state._streaming = False

    return "".join(chunks)


# ═══════════════════════════════════════════════════════════════
# 兼容旧接口：完整检索 + 流式回答（独立调用时使用）
# ═══════════════════════════════════════════════════════════════

def answer_with_rag_stream(
    question: str,
    entries: List[Dict[str, Any]],
    content_vs,
    image_maps: Dict[str, Dict[str, str]],
    selected_model: str,
    top_k: int,
    fuzzy_th: float,
    min_cn_len: int,
    require_all: bool,
    llm_provider: str = "ollama",
    openai_api_key: str = "",
    openai_base_url: str = "",
    ollama_base_url: str = "",
    chat_history_messages: List[Dict[str, str]] | None = None,
) -> tuple:
    """
    [兼容保留] RAG 路径：自行执行检索 + 流式输出。
    在独立调用（非 graph.stream() 流程）时使用。
    """
    # 1. 混合检索
    top_items = hybrid_search(
        entries, question,
        top_k=top_k, fuzzy_th=fuzzy_th,
        require_all=require_all, min_cn_len=min_cn_len,
    )
    if not top_items:
        return (
            "没有检索到和你问题明显相关的内容。\n"
            "可以尝试：减少关键词、降低相似度阈值，或改写问题再试一次。",
            [],
        )

    # 2. 构建上下文
    context, top_docs = build_context_from_top_items(top_items, content_vs)

    # 3. 流式输出
    chat_history_str = format_chat_history(chat_history_messages or [])
    final_text = stream_rag_answer(
        question=question,
        context=context,
        chat_history_str=chat_history_str,
        llm_provider=llm_provider,
        selected_model=selected_model,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        ollama_base_url=ollama_base_url,
    )
    return final_text, top_docs


def answer_with_llm_stream(
    question: str,
    selected_model: str,
    llm_provider: str = "ollama",
    openai_api_key: str = "",
    openai_base_url: str = "",
    ollama_base_url: str = "",
    chat_history_messages: List[Dict[str, str]] | None = None,
) -> str:
    """[兼容保留] 纯 LLM 闲聊流式"""
    history_str = format_chat_history(chat_history_messages or [])
    return stream_chat_answer(
        question=question,
        chat_history_str=history_str,
        llm_provider=llm_provider,
        selected_model=selected_model,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        ollama_base_url=ollama_base_url,
    )


# ═══════════════════════════════════════════════════════════════
# 对话导出
# ═══════════════════════════════════════════════════════════════

def export_chat_markdown(messages: List[Dict[str, Any]], kb_name: str = "") -> str:
    """将对话历史导出为 Markdown 文本"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# 对话记录",
        f"**知识库**: {kb_name or '（未选择）'}  ",
        f"**导出时间**: {now}  ",
        f"**对话轮次**: {len([m for m in messages if m.get('role') == 'user'])}  ",
        "",
        "---",
        "",
    ]
    for msg in messages:
        role = "🧑 用户" if msg.get("role") == "user" else "🤖 助手"
        lines.append(f"### {role}")
        lines.append("")
        lines.append(msg.get("content", "(空)"))
        lines.append("")
        # 如果有引用来源，附加到消息末尾
        if msg.get("sources"):
            lines.append("> **参考来源**:  ")
            for s in msg["sources"]:
                label = s.metadata.get("label", "未知")
                fname = s.metadata.get("source", "未知来源")
                lines.append(f"> - `{fname}` → {label}")
            lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# KB 统计信息（供管理 Tab 使用）
# ═══════════════════════════════════════════════════════════════

def _get_kb_stats(kb_name: str) -> Dict[str, Any]:
    """获取知识库统计信息（含主要内容摘要）"""
    stats = {
        "index_entries": 0,
        "doc_count": 0,
        "size_mb": 0.0,
        "last_built": "未知",
        "summary": "",
        "doc_labels": [],  # 文档标签（目录层级列表）
    }
    kb_path = kb_dir(kb_name)
    if not os.path.isdir(kb_path):
        return stats

    # 索引条目数
    entries = load_index_entries(kb_name)
    stats["index_entries"] = len(entries)

    # 去重文档数 + 目录标签
    sources = set()
    labels = []
    for e in entries:
        src = e.get("source", "")
        label = e.get("label", "")
        if src:
            sources.add(src)
        if label and label not in labels:
            labels.append(label)
    stats["doc_count"] = len(sources) if sources else len(entries)
    stats["doc_labels"] = labels[:20]  # 最多展示 20 个标签

    # 文件夹大小
    total_size = 0
    for dirpath, _dirnames, filenames in os.walk(kb_path):
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            try:
                total_size += os.path.getsize(fp)
            except OSError:
                pass
    stats["size_mb"] = round(total_size / (1024 * 1024), 2)

    # 构建时间
    idx_path = os.path.join(kb_path, "index.json")
    if os.path.exists(idx_path):
        try:
            mtime = os.path.getmtime(idx_path)
            stats["last_built"] = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        except OSError:
            pass

    # 主要内容摘要
    summary_path = os.path.join(kb_path, "summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                stats["summary"] = data.get("summary", "")
        except Exception:
            pass

    return stats


# ═══════════════════════════════════════════════════════════════
# Tab 1: RAG 问答
# ═══════════════════════════════════════════════════════════════

def build_rag_tab():
    """RAG 问答功能界面 — 使用 graph.stream() 自动编排"""
    st.subheader("💬 问答区")

    kbs = list_kbs()
    if not kbs:
        st.warning("还没有任何知识库，请先在『知识库构建』里创建。")
        st.stop()

    # ── 初始化 session state ──
    if "selected_kb" not in st.session_state:
        st.session_state.selected_kb = ""
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
    if "_stop_requested" not in st.session_state:
        st.session_state._stop_requested = False
    if "_streaming" not in st.session_state:
        st.session_state._streaming = False
    if "_partial_answer" not in st.session_state:
        st.session_state._partial_answer = ""
    # 检索参数默认值（防止用户未展开侧边栏检索参数面板时 KeyError）
    if "top_k" not in st.session_state:
        st.session_state.top_k = SHOW_K_DEFAULT
    if "fuzzy_th" not in st.session_state:
        st.session_state.fuzzy_th = 0.55
    if "min_cn_len" not in st.session_state:
        st.session_state.min_cn_len = 2
    if "require_all" not in st.session_state:
        st.session_state.require_all = False
    if "_prev_selected_kb" not in st.session_state:
        st.session_state._prev_selected_kb = ""
    if "_rag_restored" not in st.session_state:
        st.session_state._rag_restored = False

    # ── 已中断的流式生成处理（停止按钮触发的 rerun）──
    if st.session_state.get("_stop_requested") and st.session_state.get("_streaming"):
        partial = st.session_state.get("_partial_answer", "")
        if partial:
            partial += "\n\n> ⚠️ *生成已被用户中断*"
        else:
            partial = "> ⚠️ *生成已被用户中断*"
        st.session_state.rag_messages.append({
            "role": "assistant",
            "content": partial,
            "sources": st.session_state.get("_interrupted_sources", []),
        })
        st.session_state._stop_requested = False
        st.session_state._streaming = False
        st.session_state._partial_answer = ""
        st.rerun()

    # ── 知识库选择 + 操作按钮行 ──
    kb_col, btn_col1, btn_col2, btn_col3 = st.columns([3, 1, 1, 1])
    with kb_col:
        idx = 0
        if st.session_state.selected_kb and st.session_state.selected_kb in kbs:
            idx = kbs.index(st.session_state.selected_kb) + 1
        kb_name = st.selectbox(
            "选择知识库",
            [""] + kbs,
            key="kb_selectbox",
            index=idx,
            label_visibility="collapsed",
        )
        if kb_name:
            st.session_state.selected_kb = kb_name

    # ── KB 切换或首次加载时从 Checkpointer 恢复对话历史 ──
    if st.session_state.selected_kb:
        prev = st.session_state._prev_selected_kb
        cur = st.session_state.selected_kb
        if prev != cur:
            history = load_chat_history(cur)
            st.session_state.rag_messages = history
            st.session_state._prev_selected_kb = cur

    with btn_col1:
        if st.button("🧹 清空对话", width='stretch'):
            st.session_state.rag_messages = []
            st.session_state._stop_requested = False
            st.session_state._streaming = False
            # 同步清除 Checkpointer 中持久化的对话历史，防止刷新后恢复
            if st.session_state.selected_kb:
                clear_chat_history(st.session_state.selected_kb)
            st.rerun()

    with btn_col2:
        if st.button("📥 导出对话", width='stretch',
                     disabled=len(st.session_state.rag_messages) == 0):
            st.session_state._export_data = export_chat_markdown(
                st.session_state.rag_messages,
                st.session_state.selected_kb,
            )
            st.session_state._export_filename = (
                f"chat_{st.session_state.selected_kb}_{datetime.now():%Y%m%d_%H%M}.md"
            )

        if st.session_state.get("_export_data"):
            st.download_button(
                label="⬇️ 下载 Markdown",
                data=st.session_state._export_data,
                file_name=st.session_state.get("_export_filename", "chat_export.md"),
                mime="text/markdown",
                width='stretch',
                key="dl_export",
            )

    with btn_col3:
        # 始终可点击 — 渲染时 _streaming 尚未置 True（流式在后面），
        # 若用 disabled=_streaming 会导致按钮永远不可用
        if st.button("⏹️ 停止生成", width='stretch'):
            st.session_state._stop_requested = True
            st.rerun()

    # ── 加载 KB 数据（延迟校验：闲聊路径不需要 KB）──
    entries: List[Dict[str, Any]] = []
    content_vs = None
    image_maps: Dict[str, Dict[str, str]] = {}
    summaries: Dict[str, str] = cached_load_all_kb_summaries()

    if st.session_state.selected_kb:
        entries = cached_load_index_entries(st.session_state.selected_kb)
        if entries:
            content_vs = init_content_vs_for_kb(st.session_state.selected_kb)
            image_maps = cached_load_kb_image_maps(st.session_state.selected_kb)
        # 注意：不在此处 st.stop()，闲聊路径可绕过 KB 要求

    # ── 聊天消息展示 ──
    chat_area = st.container()
    with chat_area:
        for msg in st.session_state.rag_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg.get("content", ""))
                if msg["role"] == "assistant" and msg.get("sources"):
                    with st.expander("🔎 参考来源", expanded=False):
                        for ii, src in enumerate(msg["sources"][: st.session_state.get("top_k", 3)]):
                            label = src.metadata.get("label", "（无标签）")
                            fname = src.metadata.get("source", "未知来源")
                            st.caption(f"**[{ii + 1}]** `{fname}` · {label}")
                            render_text_with_images(src.page_content, image_maps, source_filename=fname)
                            if ii < len(msg["sources"]) - 1:
                                st.divider()
        # 流式输出锚点
        live_anchor = st.empty()

    # ── 用户输入 ──
    if question := st.chat_input("请在此输入您的问题", key="chat_input_main"):
        st.session_state.rag_messages.append({"role": "user", "content": question})
        st.rerun()

    # ── 处理用户问题（统一入口）──
    if (st.session_state.rag_messages
            and st.session_state.rag_messages[-1]["role"] == "user"):
        user_question = st.session_state.rag_messages[-1]["content"]
        history_before = st.session_state.rag_messages[:-1]

        # ---- 快速路由检查（轻量，无需 KB，无需 LLM）----
        route_check_state = create_initial_state(
            question=user_question,
            kb_name=st.session_state.selected_kb or "",
            selected_model=st.session_state.get("selected_model", DEFAULT_OLLAMA_MODEL),
            chat_history=history_before,
            ollama_base_url=st.session_state.get("ollama_base_url", ""),
        )
        route_result = route_node(route_check_state)
        is_chat = route_result["route"] == "llm"

        with live_anchor.container():
            with st.chat_message("assistant"):

                # ================================================
                # 路径 A：闲聊 / 身份声明 → 直接 LLM（无需 KB）
                # ================================================
                if is_chat:
                    final_text = stream_chat_answer(
                        question=user_question,
                        chat_history_str=format_chat_history(history_before),
                        llm_provider=st.session_state.get("llm_provider", "ollama").lower(),
                        selected_model=st.session_state.get("selected_model", DEFAULT_OLLAMA_MODEL),
                        openai_api_key=st.session_state.get("openai_api_key", ""),
                        openai_base_url=st.session_state.get("openai_base_url", ""),
                        ollama_base_url=st.session_state.get("ollama_base_url", ""),
                    )
                    st.session_state.rag_messages.append({
                        "role": "assistant",
                        "content": final_text,
                        "sources": [],
                    })
                    st.rerun()

                # ================================================
                # 路径 B：RAG 检索 → 需要知识库
                # ================================================

                # B1: 无 KB → 推荐知识库
                if not st.session_state.selected_kb:
                    st.info("💡 检测到知识查询，请先选择或构建知识库：")
                    if summaries:
                        similarities = compute_cosine_similarity(user_question, summaries)
                        top_kbs = sorted(similarities, key=similarities.get, reverse=True)[:5]
                        if top_kbs:
                            selected = st.selectbox("📚 推荐知识库", top_kbs, key="recommended_kb")
                            c_confirm, c_skip = st.columns([1, 2])
                            with c_confirm:
                                if st.button("✅ 确认选择并查询", width='stretch'):
                                    st.session_state.selected_kb = selected
                                    st.rerun()
                            with c_skip:
                                st.caption("或在上方下拉框手动选择其他知识库")
                        else:
                            st.warning("没有匹配的知识库，请手动选择或先构建知识库。")
                    else:
                        st.warning("当前没有任何知识库，请先在『知识库构建』里创建。")
                    st.stop()

                # B2: 有 KB 但索引为空
                if not entries:
                    st.warning(f"知识库『{st.session_state.selected_kb}』索引为空，请先在『知识库构建』里构建。")
                    st.stop()

                # B3: 完整 RAG 流程
                with st.status("🤔 正在分析与检索...", expanded=False) as thinking:
                    # --- 构建初始状态 ---
                    initial_state = create_initial_state(
                        question=user_question,
                        kb_name=st.session_state.selected_kb,
                        selected_model=st.session_state.selected_model,
                        llm_provider=st.session_state.get("llm_provider", "ollama").lower(),
                        openai_api_key=st.session_state.get("openai_api_key", ""),
                        openai_base_url=st.session_state.get("openai_base_url", ""),
                        ollama_base_url=st.session_state.get("ollama_base_url", ""),
                        top_k=st.session_state.top_k,
                        fuzzy_th=st.session_state.fuzzy_th,
                        min_cn_len=st.session_state.min_cn_len,
                        require_all=st.session_state.require_all,
                        image_maps=image_maps,
                        chat_history=history_before,
                    )

                    # --- 编译 Agent（缓存，只编译一次）---
                    checkpointer = create_sqlite_checkpointer()
                    thread_id = get_or_create_thread_id(st.session_state.selected_kb)
                    config = {"configurable": {"thread_id": thread_id}}
                    graph = cached_compile_agent(checkpointer)

                    # --- graph.stream() 自动编排 ---
                    final_state = dict(initial_state)
                    try:
                        for step in graph.stream(initial_state, config):
                            for node_name, node_update in step.items():
                                final_state.update(node_update)
                                msg = node_update.get("status_message", "")
                                if msg:
                                    thinking.write(msg)
                    except Exception as e:
                        thinking.update(label="❌ Agent 执行出错", state="error")
                        st.error(f"错误详情: {e}")
                        st.stop()

                # --- 流式生成 ---
                thinking.update(
                    label="📝 正在生成回答（RAG 检索）...",
                    state="running",
                )

                top_items = final_state.get("top_items", [])
                if top_items:
                    context, top_docs = build_context_from_top_items(top_items, content_vs)
                    st.session_state._interrupted_sources = top_docs
                    final_text = stream_rag_answer(
                        question=user_question,
                        context=context,
                        chat_history_str=format_chat_history(history_before),
                        llm_provider=st.session_state.get("llm_provider", "ollama").lower(),
                        selected_model=st.session_state.selected_model,
                        openai_api_key=st.session_state.get("openai_api_key", ""),
                        openai_base_url=st.session_state.get("openai_base_url", ""),
                        ollama_base_url=st.session_state.get("ollama_base_url", ""),
                    )
                else:
                    final_text = (
                        "没有检索到和你问题明显相关的内容。\n"
                        "可以尝试：减少关键词、降低相似度阈值，或改写问题再试一次。"
                    )
                    top_docs = []

                thinking.update(
                    label=f"✅ 生成完成（路由: RAG，检索到 {len(top_docs)} 条参考）",
                    state="complete",
                )

                # --- 保存回答 ---
                st.session_state.rag_messages.append({
                    "role": "assistant",
                    "content": final_text,
                    "sources": top_docs,
                })

                # --- 持久化到 Checkpointer ---
                try:
                    graph.update_state(config, {
                        "answer": final_text,
                        "chat_history": st.session_state.rag_messages,
                    })
                except Exception:
                    pass

                st.rerun()


# ═══════════════════════════════════════════════════════════════
# Tab 2: 知识库构建
# ═══════════════════════════════════════════════════════════════

def build_kb_construction_tab():
    """知识库构建界面 —— 支持上传文件 / 文件夹批量导入，含进度条"""
    st.subheader("📦 知识库构建（Word / PDF / Markdown → 向量入库）")

    build_mode = st.radio(
        "📂 导入模式",
        ["📤 上传文件", "📁 从本地文件夹批量导入"],
        horizontal=True,
        help="上传文件：手动选择文件上传；文件夹导入：指定本地目录批量导入"
    )

    if build_mode == "📤 上传文件":
        with st.form(key='kb_form_upload'):
            kb_name = st.text_input(
                "知识库名称（英文 / 数字 / 下划线 / 短横线）",
                placeholder="e.g. finance_gl",
                help="不同知识库相互隔离，便于多文档场景管理",
            )
            uploaded_files = st.file_uploader(
                "上传文件（支持 .docx / .pdf / .md）",
                accept_multiple_files=True,
                type=['docx', 'pdf', 'md'],
            )
            c1, c2 = st.columns(2)
            with c1:
                chunk_size = st.number_input("文本块大小", 100, 2000, 600, 50)
            with c2:
                chunk_overlap = st.number_input("文本块重叠", 0, 500, 80, 10)
            submitted = st.form_submit_button("🚀 开始构建", width='stretch')

            if submitted:
                if not kb_name or not kb_name.strip():
                    st.error("请先填写知识库名称。")
                elif not uploaded_files:
                    st.error("请至少上传一个文件。")
                else:
                    build_knowledge_base_from_files(
                        uploaded_files=uploaded_files,
                        kb_name=kb_name.strip(),
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        llm_provider=st.session_state.get("llm_provider", "ollama").lower(),
                        llm_model=st.session_state.get("selected_model", ""),
                        openai_api_key=st.session_state.get("openai_api_key", ""),
                        openai_base_url=st.session_state.get("openai_base_url", ""),
                    )
                    st.cache_resource.clear()
                    st.cache_data.clear()
                    st.success(
                        f"知识库『{kb_name}』构建完成。"
                        f"请切换到「💬 RAG问答」Tab 开始使用。"
                    )

    else:  # 文件夹批量导入
        with st.form(key='kb_form_folders'):
            kb_name = st.text_input(
                "知识库名称（英文 / 数字 / 下划线 / 短横线）",
                placeholder="e.g. finance_gl",
                help="不同知识库相互隔离，便于多文档场景管理",
            )
            folder_paths_input = st.text_area(
                "源文件夹路径（每行一个路径）",
                placeholder="D:\\Documents\\手册1\nD:\\Documents\\手册2\nE:\\知识库\\财务相关",
                help="支持多个文件夹，系统会递归扫描每个文件夹中的 .docx / .pdf / .md 文件",
            )
            c1, c2 = st.columns(2)
            with c1:
                chunk_size = st.number_input("文本块大小", 100, 2000, 600, 50, key="folder_chunk_size")
            with c2:
                chunk_overlap = st.number_input("文本块重叠", 0, 500, 80, 10, key="folder_chunk_overlap")
            submitted = st.form_submit_button("🚀 开始批量构建", width='stretch')

            if submitted:
                if not folder_paths_input.strip():
                    st.error("请至少输入一个文件夹路径。")
                else:
                    folder_paths = [
                        p.strip() for p in folder_paths_input.strip().split("\n")
                        if p.strip()
                    ]
                    valid_paths = []
                    invalid_paths = []
                    for p in folder_paths:
                        if os.path.isdir(p):
                            # 扫描文件
                            for root, _dirs, files in os.walk(p):
                                if any(f.endswith(('.docx', '.pdf', '.md')) for f in files):
                                    valid_paths.append(p)
                                    break
                            else:
                                invalid_paths.append(f"{p} (无支持的文件)")
                        else:
                            invalid_paths.append(p)

                    if invalid_paths:
                        st.warning(f"以下路径无效，已跳过：{', '.join(invalid_paths)}")

                    if valid_paths:
                        build_knowledge_base_from_folders(
                            folder_paths=valid_paths,
                            kb_name=kb_name.strip(),
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            llm_provider=st.session_state.get("llm_provider", "ollama").lower(),
                            llm_model=st.session_state.get("selected_model", ""),
                            openai_api_key=st.session_state.get("openai_api_key", ""),
                            openai_base_url=st.session_state.get("openai_base_url", ""),
                        )
                        st.cache_resource.clear()
                        st.cache_data.clear()
                    else:
                        st.error("没有有效的文件夹路径，请检查输入。")

    st.caption("> 需要清理旧知识库？请切换到「⚙️ 知识库管理」Tab。")


# ═══════════════════════════════════════════════════════════════
# Tab 3: 知识库管理
# ═══════════════════════════════════════════════════════════════

def build_kb_management_tab():
    """知识库管理界面 —— 删除 + 统计信息"""
    st.subheader("⚙️ 知识库管理")

    kbs = list_kbs()
    if not kbs:
        st.info("当前没有可管理的知识库。请先在『知识库构建』里创建。")
        st.stop()

    # ── KB 信息卡片（可展开查看摘要）──
    st.markdown("### 📊 知识库概览")

    for i, kb_name in enumerate(kbs):
        stats = _get_kb_stats(kb_name)
        summary = stats.get("summary", "")
        labels = stats.get("doc_labels", [])

        # 标题行：序号 + 名称 + 关键指标
        with st.expander(
            f"📁 **{kb_name}**  —  {stats['doc_count']} 文档 · "
            f"{stats['index_entries']} 索引块 · "
            f"{stats['size_mb']} MB · "
            f"🕐 {stats['last_built']}",
            expanded=False,
        ):
            col_info, col_summary = st.columns([1, 2])

            with col_info:
                st.markdown("**📋 基本信息**")
                st.markdown(f"- 文档数量：{stats['doc_count']}")
                st.markdown(f"- 索引块数：{stats['index_entries']}")
                st.markdown(f"- 存储大小：{stats['size_mb']} MB")
                st.markdown(f"- 构建时间：{stats['last_built']}")

                if labels:
                    st.markdown("**📑 内容目录**")
                    for label in labels[:10]:
                        st.markdown(f"- {label}")
                    if len(labels) > 10:
                        st.caption(f"… 共 {len(labels)} 个目录")

            with col_summary:
                if summary:
                    st.markdown("**📝 主要内容摘要**")
                    st.markdown(
                        f'<div class="kb-summary-box">{summary}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("暂无摘要（构建知识库时将自动生成）")

            # 快捷操作
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🔍 选择此知识库", key=f"select_kb_{kb_name}"):
                    st.session_state.selected_kb = kb_name
                    st.success(f"已选择知识库『{kb_name}』，请切换到「💬 问答区」Tab")
            with col_b:
                if st.button("📋 查看详细索引", key=f"detail_kb_{kb_name}"):
                    with st.container():
                        entries = load_index_entries(kb_name)
                        for e in entries[:30]:
                            st.caption(
                                f"`{e.get('source', '?')}` → {e.get('label', '?')} "
                                f"({e.get('char_count', 0)} 字符)"
                            )
                        if len(entries) > 30:
                            st.caption(f"… 共 {len(entries)} 条索引")

    st.divider()

    # ── 删除 KB ──
    st.markdown("### 🗑️ 删除知识库")
    kb_to_delete = st.selectbox("选择要删除的知识库", kbs, key="selected_kb_to_delete")

    with st.expander("⚠️ 高危操作说明", expanded=False):
        st.markdown(
            "- 将同时删除：`RAG_doc/<kb>` 与 `rag_processed_data/<kb>` 下所有文件；\n"
            "- 操作**不可恢复**，请谨慎！"
        )

    col_a, col_b = st.columns([0.6, 0.4])
    with col_a:
        confirm_text = st.text_input(
            "请输入要删除的知识库名称以确认：",
            placeholder=kb_to_delete,
        )
    with col_b:
        deep_confirm = st.checkbox("我已了解此操作不可恢复", value=False)

    delete_btn = st.button(
        "🗑️ 删除该知识库",
        type="primary",
        disabled=not (confirm_text == kb_to_delete and deep_confirm),
    )

    if delete_btn:
        with st.status("正在删除知识库文件...", expanded=True) as status:
            ok, logs = delete_kb(kb_to_delete)
            for line in logs:
                st.write(line)
            if ok:
                status.update(label="✅ 删除完成", state="complete")
                st.success(f"知识库『{kb_to_delete}』已删除。")
            else:
                status.update(label="❌ 删除失败", state="error")
                st.error("删除过程中有错误，详见上方日志。")
        st.cache_resource.clear()
        st.cache_data.clear()
        time.sleep(1)
        st.rerun()


# ═══════════════════════════════════════════════════════════════
# 侧边栏构建
# ═══════════════════════════════════════════════════════════════

def build_sidebar():
    """构建侧边栏：模型配置 + 检索参数（分组折叠）"""
    with st.sidebar:
        # ── Logo ──
        try:
            st.image('logo.png', width='stretch')
        except Exception:
            pass

        st.markdown("### ⚙️ 系统配置")

        # ── 模型配置 ──
        with st.expander("🤖 模型配置", expanded=True):
            st.selectbox(
                "LLM 后端",
                ["Ollama", "OpenAI"],
                key="llm_provider",
                help="Ollama: 本地部署；OpenAI: 兼容 API",
            )

            if st.session_state.llm_provider == "Ollama":
                # Ollama 服务地址
                if "ollama_base_url" not in st.session_state:
                    st.session_state.ollama_base_url = OLLAMA_BASE_URL
                st.text_input(
                    "Ollama 服务地址",
                    key="ollama_base_url",
                    value=OLLAMA_BASE_URL,
                    placeholder="http://localhost:11434",
                    help="Ollama 服务的 API 地址（默认使用 config.OLLAMA_BASE_URL）",
                )

                models = get_available_ollama_models(st.session_state.ollama_base_url)
                # 确定默认选中项：优先已选（且仍在列表中），否则用 DEFAULT_OLLAMA_MODEL
                existing = st.session_state.get("selected_model", "")
                if existing and existing in models:
                    default_idx = models.index(existing)
                elif DEFAULT_OLLAMA_MODEL in models:
                    default_idx = models.index(DEFAULT_OLLAMA_MODEL)
                else:
                    default_idx = 0
                col_m, col_r = st.columns([4, 1])
                with col_m:
                    st.selectbox("模型", models, key="selected_model", index=default_idx)
                with col_r:
                    if st.button("🔄", help="刷新模型列表", width='stretch'):
                        st.cache_data.clear()
                        st.rerun()
            else:
                st.text_input(
                    "API Key",
                    type="password",
                    key="openai_api_key",
                    placeholder="sk-...（留空则使用环境变量）",
                )
                st.text_input(
                    "API Base URL",
                    key="openai_base_url",
                    value=OPENAI_DEFAULT_BASE_URL,
                    placeholder="https://api.openai.com/v1",
                    help="支持任何 OpenAI 兼容 API",
                )
                st.text_input(
                    "模型名称",
                    key="selected_model",
                    value="gpt-4o",
                    placeholder="gpt-4o / gpt-4-turbo / deepseek-v3",
                )

            # 兜底
            if "selected_model" not in st.session_state:
                st.session_state.selected_model = DEFAULT_OLLAMA_MODEL

        # ── 检索参数 ──
        with st.expander("🔍 检索参数", expanded=False):
            st.session_state.top_k = st.slider(
                "Top-K 返回数量", 1, 10, SHOW_K_DEFAULT, 1,
                help="检索阶段返回的候选分区数量",
            )
            st.session_state.fuzzy_th = st.slider(
                "模糊阈值", 0.30, 0.90, 0.55, 0.01,
                help="低于此相似度的关键词将被忽略",
            )
            st.session_state.min_cn_len = st.slider(
                "中文短词最小长度", 2, 4, 2, 1,
                help="中文词语的最小字符数",
            )
            st.session_state.require_all = st.checkbox(
                "ALL 模式（所有词必须出现 / 或高相似度兜底）",
                value=False,
            )

            if st.button("↩️ 恢复默认", width='stretch'):
                st.session_state.top_k = SHOW_K_DEFAULT
                st.session_state.fuzzy_th = 0.55
                st.session_state.min_cn_len = 2
                st.session_state.require_all = False
                st.rerun()

        st.divider()

        # ── 当前状态 ──
        with st.expander("📋 当前状态", expanded=False):
            kb = st.session_state.get("selected_kb", "")
            if kb:
                stats = _get_kb_stats(kb)
                st.markdown(f"**知识库**: {kb}")
                st.markdown(f"- 索引条目: {stats['index_entries']}")
                st.markdown(f"- 文档数: {stats['doc_count']}")
                st.markdown(f"- 大小: {stats['size_mb']} MB")
                st.markdown(f"- 构建时间: {stats['last_built']}")
            else:
                st.caption("未选择知识库")

            msg_count = len(st.session_state.get("rag_messages", []))
            user_count = len([
                m for m in st.session_state.get("rag_messages", [])
                if m.get("role") == "user"
            ])
            st.markdown(f"**对话轮次**: {user_count} 问 / {msg_count} 条")


# ═══════════════════════════════════════════════════════════════
# 主 UI 入口
# ═══════════════════════════════════════════════════════════════

def Build_UI():
    """构建 Streamlit 主界面 —— v2.0 优化版"""
    # ── 注入自定义 CSS ──
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── 构建侧边栏 ──
    build_sidebar()

    # ── 顶部标题栏 ──
    col_logo, col_title = st.columns([1, 9])
    with col_logo:
        try:
            st.image('logo.png', width='stretch')
        except Exception:
            st.markdown("📖")
    with col_title:
        st.markdown(
            '<div style="font-size:1.5rem;font-weight:700;margin-top:0.3rem;">'
            '📖 基于大模型的用户手册查询平台'
            '</div>',
            unsafe_allow_html=True,
        )
        st.caption("RAG · LangGraph Agent · 混合检索")

    st.divider()

    # ── 主功能区（原生 Tabs）──
    tab1, tab2, tab3 = st.tabs([
        "💬 RAG 问答",
        "📦 知识库构建",
        "⚙️ 知识库管理",
    ])

    with tab1:
        build_rag_tab()
    with tab2:
        build_kb_construction_tab()
    with tab3:
        build_kb_management_tab()
