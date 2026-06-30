# -*- coding: utf-8 -*-
"""LangGraph Checkpointer 管理：SQLite 持久化 + 线程 ID 管理"""

import sqlite3
import os
from typing import Optional

# SqliteSaver 在 langgraph >= 1.0 中拆分为独立包 langgraph-checkpoint-sqlite
# 优先从新路径导入，失败则尝试旧路径（langgraph < 1.0）
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except (ImportError, ModuleNotFoundError):
    try:
        from langgraph_checkpoint_sqlite import SqliteSaver  # type: ignore[import-untyped,no-redef]
    except (ImportError, ModuleNotFoundError):
        raise ImportError(
            "SqliteSaver 导入失败，请安装: pip install langgraph-checkpoint-sqlite>=2.0.0"
        )

try:
    from langgraph.checkpoint.memory import MemorySaver
except (ImportError, ModuleNotFoundError):
    from langgraph.checkpoint.memory import InMemorySaver as MemorySaver  # type: ignore[import-untyped,no-redef]

from config import BASE_KB_DIR


# ==================== Checkpointer 工厂 ====================

_CHECKPOINTER_INSTANCE: Optional[SqliteSaver] = None
_CHECKPOINTER_CONN: Optional[sqlite3.Connection] = None


def create_sqlite_checkpointer(db_path: str = "") -> SqliteSaver:
    """
    创建或获取 SQLite Checkpointer 单例

    使用 sqlite3 直接连接（非 context manager），适配 Streamlit 长生命周期。
    每次 graph 编译时复用同一连接。

    :param db_path: SQLite 数据库路径，默认存储于 {BASE_KB_DIR}/checkpoints.db
    :return: SqliteSaver 实例
    """
    global _CHECKPOINTER_INSTANCE, _CHECKPOINTER_CONN

    if _CHECKPOINTER_INSTANCE is not None:
        return _CHECKPOINTER_INSTANCE

    if not db_path:
        os.makedirs(BASE_KB_DIR, exist_ok=True)
        db_path = os.path.join(BASE_KB_DIR, "checkpoints.db")

    conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()

    _CHECKPOINTER_INSTANCE = saver
    _CHECKPOINTER_CONN = conn
    return saver


def create_memory_checkpointer() -> MemorySaver:
    """创建内存 Checkpointer（用于开发/测试，不持久化）"""
    return MemorySaver()


# ==================== 线程 ID 管理 ====================

def get_or_create_thread_id(kb_name: str) -> str:
    """
    基于知识库名称生成稳定的 thread_id

    策略：一个知识库 = 一个对话线程。
    这意味着同一 KB 下的所有问题共享同一个对话历史和 LangGraph 状态。

    :param kb_name: 知识库名称
    :return: thread_id 字符串
    """
    return f"kb_{kb_name}"


# ==================== 对话历史恢复 ====================

def load_chat_history(kb_name: str) -> list:
    """
    从 SQLite Checkpointer 中恢复指定知识库的对话历史

    用于 Streamlit 刷新后恢复 st.session_state.rag_messages。
    如果知识库从未有过对话，返回空列表。

    :param kb_name: 知识库名称
    :return: chat_history 列表 [{"role":"user","content":"..."}, ...]
    """
    try:
        saver = create_sqlite_checkpointer()
        thread_id = get_or_create_thread_id(kb_name)
        config = {"configurable": {"thread_id": thread_id}}

        tuple_ = saver.get_tuple(config)
        if tuple_ and tuple_.checkpoint:
            channel_values = tuple_.checkpoint.get("channel_values", {}) or {}
            history = channel_values.get("chat_history", [])
            if history:
                return list(history)
    except Exception:
        pass

    return []


def clear_chat_history(kb_name: str) -> None:
    """
    清除指定知识库在 Checkpointer 中持久化的对话历史

    与 UI 中"清空对话"按钮配合使用，确保刷新后不会恢复已清除的对话。

    :param kb_name: 知识库名称
    """
    try:
        saver = create_sqlite_checkpointer()
        thread_id = get_or_create_thread_id(kb_name)
        saver.delete_thread(thread_id)
    except Exception:
        pass
