# -*- coding: utf-8 -*-
"""
大模型用户手册查询交互平台 —— LangGraph Agent 版本

主入口文件：基于 LangGraph 的模块化 RAG 系统
支持多文件夹知识库批量导入、混合检索、Agent 自主推理

v2.0: 侧边栏重构、消除双重检索、停止生成、对话管理、自定义主题
"""

import streamlit as st

# -------------------- Streamlit 基本设置 --------------------
st.set_page_config(
    page_title="用户手册查询平台",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- 主入口 --------------------
from ui.components import Build_UI  # noqa: E402


def main():
    """主函数：初始化界面并启动应用"""
    Build_UI()


if __name__ == '__main__':
    main()
