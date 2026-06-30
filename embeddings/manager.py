# -*- coding: utf-8 -*-
"""嵌入模型管理器"""

import numpy as np
from typing import Dict

# 优先导入新版，失败则降级
try:
    from langchain_ollama import OllamaEmbeddings
except Exception:
    from langchain_community.embeddings import OllamaEmbeddings  # type: ignore

from config import OLLAMA_BASE_URL, EMBEDDING_MODEL


def create_embeddings() -> OllamaEmbeddings:
    """创建 Ollama 嵌入模型实例"""
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)


def compute_cosine_similarity(query_text: str, candidate_texts: Dict[str, str]) -> Dict[str, float]:
    """
    计算查询文本与候选文本的余弦相似度
    :param query_text: 查询文本
    :param candidate_texts: {key: text} 候选文本字典
    :return: {key: similarity} 相似度字典
    """
    embeddings = create_embeddings()
    query_emb = np.array(embeddings.embed_query(query_text))
    similarities = {}
    for key, text in candidate_texts.items():
        if not text:
            similarities[key] = 0.0
            continue
        cand_emb = np.array(embeddings.embed_query(text))
        norm_query = np.linalg.norm(query_emb)
        norm_cand = np.linalg.norm(cand_emb)
        if norm_query > 0 and norm_cand > 0:
            similarities[key] = float(np.dot(query_emb, cand_emb) / (norm_query * norm_cand))
        else:
            similarities[key] = 0.0
    return similarities
