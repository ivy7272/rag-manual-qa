# -*- coding: utf-8 -*-
"""混合检索：关键词匹配 + 向量语义检索，使用 RRF 融合"""

from typing import List, Dict, Any, Optional

import numpy as np
from langchain_core.documents import Document

from config import RRF_K
from retrieval.keyword_search import search_index_entries
from knowledge_base.vector_store import fetch_docs_by_chunk_uids
from embeddings.manager import create_embeddings


def _rrf_fusion(
    keyword_results: List[Dict[str, Any]],
    vector_results: List[Dict[str, Any]],
    k: int = RRF_K,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion：合并关键词检索和向量检索的结果
    :param keyword_results: 关键词检索结果（index.json 条目）
    :param vector_results: 向量检索结果（index.json 条目）
    :param k: RRF 平滑因子
    :return: 融合后的排序结果
    """
    scores: Dict[str, float] = {}
    uid_to_item: Dict[str, Dict[str, Any]] = {}

    for rank, item in enumerate(keyword_results):
        uid = item.get("chunk_uid", "")
        if uid:
            scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank + 1)
            uid_to_item[uid] = item

    for rank, item in enumerate(vector_results):
        uid = item.get("chunk_uid", "")
        if uid:
            scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank + 1)
            uid_to_item[uid] = item

    sorted_uids = sorted(scores, key=scores.get, reverse=True)
    return [uid_to_item[uid] for uid in sorted_uids if uid in uid_to_item]


def _build_uid_index(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """构建 chunk_uid → index entry 的快速查找表"""
    idx: Dict[str, Dict[str, Any]] = {}
    for item in entries:
        uid = item.get("chunk_uid", "")
        if uid:
            idx[uid] = item
    return idx


def _vector_search_via_index(
    entries: List[Dict[str, Any]],
    query: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    向量语义检索（回退方案）：逐个计算 index entry 与 query 的余弦相似度
    当 Chroma 向量库不可用时使用。时间复杂度 O(n)。
    """
    embeddings = create_embeddings()
    query_emb = np.array(embeddings.embed_query(query))

    vec_scored: List[tuple] = []
    for item in entries:
        label = item.get("label", "")
        kw_text = item.get("keywords_text", "")
        haystack = f"{label} {kw_text}"
        if not haystack.strip():
            continue
        try:
            hay_emb = np.array(embeddings.embed_query(haystack))
            norm_q = np.linalg.norm(query_emb)
            norm_h = np.linalg.norm(hay_emb)
            if norm_q > 0 and norm_h > 0:
                sim = float(np.dot(query_emb, hay_emb) / (norm_q * norm_h))
                vec_scored.append((sim, item))
        except Exception:
            continue

    vec_scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in vec_scored[:top_k]]


def _vector_search_via_chroma(
    vs,
    query: str,
    top_k: int,
    uid_index: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    向量语义检索（Chroma 索引方案）：使用 Chroma 的 ANN 索引进行高效检索
    时间复杂度 O(log n)，适合大规模知识库。
    """
    try:
        docs_with_scores = vs.similarity_search_with_score(query, k=top_k)
    except Exception:
        return []

    results: List[Dict[str, Any]] = []
    for doc, _score in docs_with_scores:
        uid = (doc.metadata or {}).get("chunk_uid", "")
        if uid and uid in uid_index:
            results.append(uid_index[uid])

    return results


def hybrid_search(
    entries: List[Dict[str, Any]],
    query: str,
    top_k: int = 3,
    fuzzy_th: float = 0.55,
    require_all: bool = False,
    min_cn_len: int = 2,
    vs=None,
) -> List[Dict[str, Any]]:
    """
    混合检索：关键词检索 + 向量语义检索 → RRF 融合

    :param entries: index.json 索引条目列表
    :param query: 用户查询
    :param top_k: 返回前 k 个结果
    :param fuzzy_th: 关键词检索模糊阈值
    :param require_all: ALL 模式
    :param min_cn_len: 中文短词最小长度
    :param vs: Chroma 向量库实例（可选）。传入时使用 Chroma ANN 索引进行高效
              向量检索；不传时回退到手动 O(n) 余弦相似度计算
    :return: 融合后的 Top-K 检索结果
    """
    candidate_k = max(top_k * 2, 10)

    # 1. 关键词检索（从 index.json）
    kw_results = search_index_entries(
        entries, query,
        top_k=candidate_k,
        fuzzy_th=fuzzy_th,
        require_all=require_all,
        min_cn_len=min_cn_len,
    )

    # 2. 向量语义检索
    if vs is not None:
        # ✅ 使用 Chroma ANN 索引（高效）
        uid_index = _build_uid_index(entries)
        vec_results = _vector_search_via_chroma(vs, query, candidate_k, uid_index)
    else:
        # ⚠️ 回退：手动 O(n) 余弦相似度（兼容无向量库的调用场景）
        vec_results = _vector_search_via_index(entries, query, candidate_k)

    # 3. RRF 融合
    fused = _rrf_fusion(kw_results, vec_results)

    return fused[:top_k]


def hybrid_retrieve_and_fetch(
    entries: List[Dict[str, Any]],
    query: str,
    vs,
    top_k: int = 3,
    fuzzy_th: float = 0.55,
    require_all: bool = False,
    min_cn_len: int = 2,
) -> List[Document]:
    """
    混合检索 + 向量库回表：一站式检索获取完整文档
    :return: 命中的 Document 列表
    """
    # 混合检索
    top_items = hybrid_search(
        entries, query,
        top_k=top_k, fuzzy_th=fuzzy_th,
        require_all=require_all, min_cn_len=min_cn_len,
    )

    if not top_items:
        return []

    # 向量库回表
    uids = [it.get("chunk_uid", "") for it in top_items]
    return fetch_docs_by_chunk_uids(vs, uids)
