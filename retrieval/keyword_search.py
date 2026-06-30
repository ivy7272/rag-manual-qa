# -*- coding: utf-8 -*-
"""关键词检索：基于 index.json 的关键词匹配搜索"""

from difflib import SequenceMatcher
from typing import List, Dict, Any

from utils.text_utils import normalize_text, tokenize


def seq_ratio(a: str, b: str) -> float:
    """计算两个字符串的相似度（0-1之间）"""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def score_item(item: Dict[str, Any], tokens: List[str], q_norm: str, require_all: bool = False) -> tuple:
    """
    为索引条目打分（评估与查询的相关性）
    :return: (hits, hits_in_label, fuzzy, -length_penalty)
    """
    label = (item.get("label") or "")
    kw = (item.get("keywords_text") or "")
    hay_raw = f"{label} {kw}"
    hay_norm = normalize_text(hay_raw)

    hits = 0
    hits_in_label = 0
    label_norm = normalize_text(label)
    for t in tokens:
        if t in hay_norm:
            hits += 1
            if t in label_norm:
                hits_in_label += 1

    fuzzy = seq_ratio(q_norm, hay_norm)
    length_penalty = len(hay_norm)

    return (hits, hits_in_label, fuzzy, -length_penalty)


def search_index_entries(
    entries: List[Dict[str, Any]],
    query: str,
    top_k: int = 3,
    fuzzy_th: float = 0.55,
    require_all: bool = False,
    min_cn_len: int = 2,
) -> List[Dict[str, Any]]:
    """
    检索索引条目（从 index.json 中查找与查询最相关的文本块）
    """
    tokens, q_norm = tokenize(query, min_cn_len=min_cn_len)
    if not tokens and not q_norm:
        return []

    scored = []
    for it in entries:
        s = score_item(it, tokens, q_norm, require_all=require_all)
        hits, _, fuzzy, _ = s
        if hits > 0 or fuzzy >= fuzzy_th:
            scored.append((s, it))

    scored.sort(key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3]), reverse=True)
    return [it for _, it in scored[:top_k]]
