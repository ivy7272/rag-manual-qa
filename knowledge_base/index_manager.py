# -*- coding: utf-8 -*-
"""索引管理：index.json、prompt_hints.json、summary.json 的读写"""

import os
import json
from typing import List, Dict, Any

from config import DEFAULT_OLLAMA_MODEL
from langchain_core.documents import Document
from llm.unified_llm import UnifiedLLM
from utils.file_utils import (
    kb_dir, kb_index_json, kb_prompt_json, kb_summary_json, kb_image_meta_dir,
)


# ==================== 索引与提示词写入 ====================

def write_index_and_prompt(kb_name: str, index_entries: List[Dict[str, Any]], content_docs: List[Document]):
    """写入索引文件（index.json）和提示词配置文件（prompt_hints.json）"""
    os.makedirs(kb_dir(kb_name), exist_ok=True)

    with open(kb_index_json(kb_name), "w", encoding="utf-8") as f:
        json.dump(index_entries, f, ensure_ascii=False, indent=2)

    all_labels = [
        (d.metadata or {}).get("label", "")
        for d in content_docs
        if (d.metadata or {}).get("label")
    ]
    uniq_labels = list(dict.fromkeys(all_labels))

    prompt_data = {
        "instruction": "请根据用户输入的问题，从下列目录中选择或提取最相关的术语，作为检索关键词。",
        "labels": uniq_labels,
        "rules": [
            "优先使用目录中出现的层级词语",
            "控制提取的关键词数量在 3~8 个之间",
            "目录层级拼接（如 财务会计 > 会计科目设置 > 计量单位）可作为强信号",
            "去掉无关的虚词或语气词",
        ],
    }

    with open(kb_prompt_json(kb_name), "w", encoding="utf-8") as f:
        json.dump(prompt_data, f, ensure_ascii=False, indent=2)


# ==================== 知识库总结 ====================

def generate_kb_summary(
    content_docs: List[Document],
    provider: str = "ollama",
    model_name: str = "",
    openai_api_key: str = "",
    openai_base_url: str = "",
) -> str:
    """
    使用 LLM 生成知识库的主要内容总结

    :param content_docs: 知识库文档列表
    :param provider: LLM 后端 ("ollama" | "openai")
    :param model_name: 模型名称（为空时使用默认值）
    :param openai_api_key: OpenAI API Key
    :param openai_base_url: OpenAI API Base URL
    :return: 100-200字的中文总结
    """
    from config import OPENAI_DEFAULT_MODEL

    if not model_name:
        model_name = OPENAI_DEFAULT_MODEL if provider == "openai" else DEFAULT_OLLAMA_MODEL

    context = "\n\n".join([(d.page_content or "")[:500] for d in content_docs[:5]])
    prompt = f"请用100-200字概括这个知识库的主要内容：\n\n{context}"

    try:
        llm = UnifiedLLM(
            provider=provider,
            model_name=model_name,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
        )
        return llm._call(prompt)
    except Exception:
        # LLM 不可用时回退到基于前几个文档标签的简单摘要
        labels = []
        for d in content_docs[:10]:
            label = (d.metadata or {}).get("label", "")
            if label and label not in labels:
                labels.append(label)
        if labels:
            return f"该知识库包含以下主要内容：{'；'.join(labels[:8])}等。"
        return f"该知识库共包含 {len(content_docs)} 个内容块。"


def load_all_kb_summaries() -> Dict[str, str]:
    """加载所有知识库的总结（用于知识库推荐）"""
    from utils.file_utils import list_kbs
    summaries: Dict[str, str] = {}
    for kb in list_kbs():
        path = kb_summary_json(kb)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                summaries[kb] = data.get("summary", "")
    return summaries


# ==================== 图片映射加载 ====================

def load_kb_image_maps(kb_name: str) -> Dict[str, Dict[str, str]]:
    """加载知识库的图片映射数据"""
    meta_dir = kb_image_meta_dir(kb_name)
    if not os.path.exists(meta_dir):
        return {}
    maps: Dict[str, Dict[str, str]] = {}
    for fname in os.listdir(meta_dir):
        if not fname.endswith("_image_map.json"):
            continue
        fpath = os.path.join(meta_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            mapping = data.get("mapping", {}) or {}
            orig = data.get("document_original")
            proc = data.get("document_processed")
            if orig:
                maps[orig] = mapping
            if proc:
                maps[proc] = mapping
        except Exception:
            continue
    return maps


# ==================== 索引条目加载 ====================

def load_index_entries(kb_name: str) -> List[Dict[str, Any]]:
    """加载知识库的索引条目（来自 index.json）"""
    path = kb_index_json(kb_name)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompt_hints(kb_name: str) -> Dict[str, Any]:
    """加载知识库的提示词配置（来自 prompt_hints.json）"""
    path = kb_prompt_json(kb_name)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
