# HandRAG —— 用户手册智能问答平台

基于 **RAG（检索增强生成）** 的用户手册智能问答系统。上传 Word/PDF/Markdown 文档，构建可搜索知识库，通过自然语言提问获取精准答案。

**技术栈**: LangGraph（ReAct Agent） + LangChain（Chroma / ChatOpenAI） + Ollama/OpenAI 双 LLM + Streamlit 界面

## ✨ 核心特性

- **📄 多格式文档解析** — 支持 Word（.docx）、PDF、Markdown，自动提取标题层级、表格和图片
- **🔍 混合检索** — RRF（倒数排名融合）结合关键词匹配 + Chroma 稠密向量相似度检索，召回更精准
- **🤖 ReAct Agent 循环** — 检索 → 相关性评判 → 查询重写 → 再检索（最多 2 轮），自主优化检索质量
- **🔀 双 LLM 提供商** — 统一接口同时支持 Ollama 本地模型 和 OpenAI 兼容 API（DeepSeek 等），UI 中随时切换
- **🖼️ 图片提取与展示** — 自动提取文档中的图片，问答时在答案中引用展示
- **💬 多轮对话** — 基于 LangGraph SqliteSaver 持久化会话历史，支持上下文追问
- **⚡ 流式输出** — LLM 回答实时流式渲染，体验流畅

## 🏗️ 系统架构

```
用户问题
  → route_node: 意图路由（闲聊 / 知识库问答）
  → retrieve_node: 混合检索（关键词 + 向量 → RRF 融合）
  → grade_node: LLM 评判文档相关性
  → [不相关] rewrite_node: LLM 重写查询关键词 → 回到检索
  → [相关] generate: 拼接上下文 → LLM 流式生成答案
```

### 模块结构

```
HandRAG/
├── app.py                          # Streamlit 入口
├── config.py                       # 全局配置常量
├── requirements.txt                # 依赖清单
│
├── agent/                          # LangGraph Agent
│   ├── state.py                    # Agent 状态定义
│   ├── nodes.py                    # 四个核心节点（route/retrieve/grade/rewrite）
│   └── graph.py                    # 图编排 + 编译
│
├── llm/                            # LLM 抽象层
│   ├── custom_ollama.py            # Ollama HTTP 调用封装（继承 LangChain LLM）
│   └── unified_llm.py              # 统一接口（Ollama / OpenAI 双后端）
│
├── embeddings/                     # 嵌入模型管理
│   └── manager.py                  # OllamaEmbeddings 封装
│
├── knowledge_base/                 # 知识库管理
│   ├── manager.py                  # 知识库构建/删除/列表
│   ├── vector_store.py             # Chroma 持久化向量存储
│   └── index_manager.py            # 索引/摘要/提示词生成
│
├── document_processing/            # 文档解析
│   ├── word_parser.py              # Word 文档解析（标题分组 + 表格）
│   ├── word_parser_helpers.py      # Word 解析辅助函数
│   ├── pdf_parser.py               # PDF 解析（书签/自动分段）
│   └── image_processor.py          # 图片提取与占位符替换
│
├── retrieval/                      # 检索模块
│   ├── hybrid_search.py            # 混合检索（RRF 融合）
│   └── keyword_search.py           # 关键词匹配检索
│
├── memory/                         # 状态持久化
│   └── checkpointer.py             # SQLite 检查点（SqliteSaver）
│
├── rendering/                      # 渲染
│   └── renderer.py                 # 答案 Markdown 渲染
│
├── ui/                             # 界面
│   └── components.py               # Streamlit UI 组件 + 流式生成
│
└── utils/                          # 工具函数
    ├── file_utils.py               # 文件操作
    └── text_utils.py               # 文本处理
```

## 🚀 快速开始

### 环境要求

- Python 3.10+
- [Ollama](https://ollama.com/)（用于嵌入模型，需拉取 `nomic-embed-text`）
- （可选）OpenAI 兼容 API Key（如 [DeepSeek](https://platform.deepseek.com/)）

**核心依赖**（详见 [requirements.txt](requirements.txt)）：

| 类别 | 包名 | 用途 |
|------|------|------|
| 框架 | `langgraph` | ReAct Agent 编排 |
| 向量库 | `langchain-chroma` | Chroma 持久化向量存储 |
| LLM | `langchain-openai` | OpenAI 兼容 API 调用 |
| 文档 | `python-docx`, `PyMuPDF` | Word / PDF 解析 |
| 界面 | `streamlit` | Web UI |
| 数值 | `numpy` | 向量相似度计算 |
| HTTP | `requests` | Ollama 模型列表查询 |

### 安装

```bash
# 1. 克隆项目
git clone https://github.com/your-username/HandRAG.git
cd HandRAG

# 2. 创建本地环境变量文件（config.py 中的默认值已可运行，可后续按需修改）
cp .env.example .env
# 编辑 .env，填入你的 Ollama 地址和 API Key

# 3. 安装依赖
pip install -r requirements.txt

# 4. 确保 Ollama 已启动并拉取嵌入模型
ollama pull nomic-embed-text
```

### 启动

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501` 即可使用。

### 使用流程

1. **配置 LLM** — 在侧边栏选择 Ollama（本地）或 OpenAI（云端 API），设置模型名称
2. **上传文档** — 将 Word/PDF/Markdown 文件放入 `RAG_doc/` 下的子文件夹（每个子文件夹 = 一个知识库）
3. **构建知识库** — 在侧边栏点击"构建/刷新知识库"，等待索引完成
4. **开始提问** — 选择知识库，输入问题，获得流式答案

## ⚙️ 配置说明

敏感配置（API Key、服务地址）通过 `.env` 文件管理，`config.py` 从环境变量读取并提供默认值。
编辑 `.env` 即可修改：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 服务地址 |
| `OPENAI_API_KEY` | （空） | OpenAI 兼容 API Key |
| `OPENAI_DEFAULT_BASE_URL` | `https://api.deepseek.com` | OpenAI 兼容 API 地址 |
| `OPENAI_DEFAULT_MODEL` | `deepseek-v4-flash` | 默认 OpenAI 模型 |
| `DEFAULT_OLLAMA_MODEL` | `gpt-oss:20b` | 默认 Ollama 模型 |

以下参数在 `config.py` 中调整：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `EMBEDDING_MODEL` | `nomic-embed-text:latest` | 嵌入模型（仅 Ollama） |
| `MAX_CHUNK_CHARS` | `1200` | 文本块最大字符数 |
| `RRF_K` | `60` | RRF 融合平滑因子 |
| `MAX_RETRIES` | `2` | Agent 重试次数上限 |

## 📋 数据目录

- `RAG_doc/` — 知识库源文档目录（每个子文件夹 = 一个知识库）
- `rag_processed_data/` — 处理后的数据（提取的图片、图片映射等）
- `RAG_doc/checkpoints.db` — 对话历史 SQLite 数据库

## 🛠️ 技术要点

### 混合检索策略

1. **关键词检索**：基于 TF 词频的倒排索引匹配（中文分词 + 模糊匹配）
2. **向量检索**：Chroma ANN 索引（优先）→ 手动余弦相似度（降级）
3. **RRF 融合**：对两路排序结果做倒数排名融合，k=60 平滑

### 容错设计

- LLM 调用失败 → 降级为非流式重试，再失败则返回错误提示
- Chroma 不可用 → 自动降级为 numpy 手动余弦相似度计算
- 评判/重写节点失败 → 跳过重试循环，避免死锁

### Agent 路由策略

基于模式匹配 + 上下文感知的智能路由：
- 识别问候/感谢/自我介绍等闲聊 → 直接 LLM 对话
- 检测指代词（那/这个/第二种…）→ 结合上文判断是否追问
- 其余 → 进入 RAG 检索流程

## 📝 License

MIT License
