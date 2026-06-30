# -*- coding: utf-8 -*-
"""LangGraph 工作流图构建器"""

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes import (
    route_node,
    retrieve_node,
    grade_node,
    rewrite_node,
    generate_rag_node,
    generate_chat_node,
)


def _route_after_decision(state: AgentState) -> str:
    """条件路由：根据 route 字段决定去 LLM 闲聊还是 RAG 检索"""
    if state["route"] == "llm":
        return "generate_chat"
    return "retrieve"


def _route_after_grade(state: AgentState) -> str:
    """条件路由：根据相关性评估决定生成回答还是重写查询"""
    # 如果在 grade 阶段已经设置了 answer（无检索结果情况），直接生成
    if state.get("answer") and "没有检索到" in state.get("answer", ""):
        return "generate_rag"
    if state.get("needs_retry") and state["retry_count"] <= state["max_retries"]:
        return "rewrite"
    return "generate_rag"


def build_agent_graph() -> StateGraph:
    """
    构建 LangGraph Agent 工作流图

    图结构：
        START
          │
    ┌─────▼──────┐
    │   route    │
    └──┬──────┬──┘
       │      │
   llm │      │ rag
       ▼      ▼
  ┌────────┐ ┌──────────┐
  │ chat   │ │ retrieve │
  └───┬────┘ └─────┬────┘
      │            │
      ▼            ▼
     END    ┌──────────┐
            │  grade   │
            └──┬───┬───┘
               │   │
      相关     │   │ 不相关 & retry<max
               ▼   ▼
        ┌──────────┐ ┌─────────┐
        │generate  │ │ rewrite │──┐
        │  _rag    │ └─────────┘  │
        └────┬─────┘      ▲       │
             │            └───────┘
             ▼            (循环回 retrieve)
            END
    """
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("route", route_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("generate_rag", generate_rag_node)
    workflow.add_node("generate_chat", generate_chat_node)

    # 设置入口
    workflow.set_entry_point("route")

    # route → conditional: llm → chat, rag → retrieve
    workflow.add_conditional_edges(
        "route",
        _route_after_decision,
        {
            "generate_chat": "generate_chat",
            "retrieve": "retrieve",
        },
    )

    # retrieve → grade
    workflow.add_edge("retrieve", "grade")

    # grade → conditional: relevant → generate_rag, needs_retry → rewrite
    workflow.add_conditional_edges(
        "grade",
        _route_after_grade,
        {
            "generate_rag": "generate_rag",
            "rewrite": "rewrite",
        },
    )

    # rewrite → retrieve (循环)
    workflow.add_edge("rewrite", "retrieve")

    # generate_rag → END
    workflow.add_edge("generate_rag", END)

    # generate_chat → END
    workflow.add_edge("generate_chat", END)

    return workflow


def compile_agent(checkpointer=None):
    """
    编译 Agent 工作流

    :param checkpointer: LangGraph Checkpointer 实例（如 SqliteSaver / MemorySaver）
                         传入后启用状态持久化和跨会话记忆
    :return: 编译后的 Runnable
    """
    workflow = build_agent_graph()

    # 先编译：get_graph() 只在 CompiledStateGraph 上可用（langgraph >= 1.0）
    compiled = workflow.compile(checkpointer=checkpointer)

    try:
        img = compiled.get_graph().draw_mermaid_png()
        with open("graph_flow.png", "wb") as f:
            f.write(img)
        print("[OK] graph_flow.png saved")
    except Exception as e:
        print(f"[WARN] draw_mermaid_png failed: {e}")

    return compiled
