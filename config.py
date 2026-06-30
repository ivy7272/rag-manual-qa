# -*- coding: utf-8 -*-
"""全局配置常量

敏感配置（OLLAMA_BASE_URL、OPENAI_API_KEY 等）通过 .env 文件或环境变量设置，
不在此文件中硬编码。详见 .env.example。
"""

import os
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件中的环境变量（如不存在则静默跳过）

# ==================== Ollama 服务配置 ====================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ==================== 存储路径配置 ====================
BASE_KB_DIR = "RAG_doc"                # 知识库根目录（每个子目录对应一个独立知识库）
BASE_DATA_DIR = "rag_processed_data"   # 处理后数据根目录（存储图片、处理后的Word文档）

# ==================== 向量库配置 ====================
COLLECTION_NAME = "rag_db"        # Chroma向量库集合名
EMBEDDING_MODEL = "nomic-embed-text:latest"  # 嵌入模型名称

# ==================== 关键词提取参数 ====================
MIN_CHUNK_LEN = 8     # 最小文本长度（低于此长度不提取关键词）
MAX_KW = 12           # 最大关键词数量（每个文本块提取前12个高频关键词）

# ==================== Chunk 文本块大小控制 ====================
MAX_CHUNK_CHARS = 1200    # 单个文本块最大字符数（超过则二次切分，适配 nomic-embed-text ~800 tokens）
MIN_CHUNK_CHARS = 100     # 单个文本块最小字符数（低于此阈值合并到相邻块，避免碎片化）
CHUNK_OVERLAP_CHARS = 200  # 超大块二次切分时的重叠字符数（保证边界处检索不被截断）

# ==================== 检索参数默认值 ====================
SHOW_K_DEFAULT = 3    # 默认Top-K检索数量
DEFAULT_FUZZY_TH = 0.55  # 默认模糊匹配阈值
DEFAULT_MIN_CN_LEN = 2   # 默认中文短词最小长度
RRF_K = 60            # RRF (Reciprocal Rank Fusion) 平滑因子

# ==================== 上下文截断参数 ====================
SNIPPET_MAX_LEN = 800       # 单个文档片段最大字符数（RAG 上下文拼接时截断）
GRADE_CONTEXT_MAX_LEN = 2000  # grade_node 评估时的上下文最大字符数
CHAT_HISTORY_MSG_MAX_LEN = 500  # 对话历史中单条消息最大字符数

# ==================== Agent 参数 ====================
MAX_RETRIES = 2       # LangGraph Agent 最大重试次数（查询重写循环上限）

# ==================== OpenAI API 配置 ====================
OPENAI_DEFAULT_BASE_URL = os.getenv("OPENAI_DEFAULT_BASE_URL", "https://api.deepseek.com")
OPENAI_DEFAULT_MODEL = os.getenv("OPENAI_DEFAULT_MODEL", "deepseek-v4-flash")
# OPENAI_API_KEY 通过环境变量 OPENAI_API_KEY 设置，也可在 UI 中手动输入

# ==================== LLM 默认参数 ====================
DEFAULT_TEMPERATURE = 0.5
DEFAULT_REQUEST_TIMEOUT = 600  # 10分钟
DEFAULT_OLLAMA_MODEL = os.getenv("DEFAULT_OLLAMA_MODEL", "gpt-oss:20b")
