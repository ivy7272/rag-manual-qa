# -*- coding: utf-8 -*-
"""文本处理工具：关键词提取、文本归一化、令牌化"""

import re
import unicodedata
from collections import Counter
from typing import List

from config import MIN_CHUNK_LEN, MAX_KW


# ==================== 轻量关键词提取 ====================

def cheap_keywords(text: str, topn: int = MAX_KW) -> List[str]:
    """
    轻量关键词提取（基于词频统计，速度快，无需NLP模型）
    提取中文2字以上、英文3字符以上的词，按词频+词长排序
    """
    text = (text or "").strip()
    if not text or len(text) < MIN_CHUNK_LEN:
        return []
    # 去掉图片标签
    text = re.sub(r'<img>.*?</img>', '', text)
    # 过滤掉数字和图片格式
    text = re.sub(r'\b(\d+|[a-zA-Z0-9_-]+\.(?:png|jpg|jpeg|gif|bmp|tiff|svg|webp))\b', '', text)
    # 提取中文关键词（正则匹配2字以上中文字符）
    zh = re.findall(r'[一-鿿]{2,}', text)
    # 提取英文关键词（正则匹配3字符以上字母/数字/下划线）
    en = re.findall(r'[A-Za-z0-9_]{3,}', text)
    toks = zh + en

    cnt = Counter(toks)
    items = sorted(cnt.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)
    kws = [w for w, _ in items]

    seen, uniq = set(), []
    for w in kws:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
        if len(uniq) >= topn:
            break
    return uniq


# ==================== 文本归一化 ====================

PUNCT_REGEX = re.compile(r"[^\w一-鿿]+", re.UNICODE)


def to_halfwidth(s: str) -> str:
    """将全角字符转换为半角（统一文本格式，提升匹配精度）"""
    return unicodedata.normalize("NFKC", s)


def normalize_text(s: str) -> str:
    """文本归一化（全角转半角、小写、去标点、去空格）"""
    s = to_halfwidth(s or "")
    s = s.strip().lower()
    s = PUNCT_REGEX.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def extract_cn_tokens(s: str, min_len: int = 2, max_len: int = 4) -> List[str]:
    """提取中文短词（2-4字）作为检索令牌（提升中文匹配精度）"""
    tokens = []
    for seg in re.findall(r"[一-鿿]+", s or ""):
        L = len(seg)
        if L >= min_len:
            tokens.append(seg)
        for n in range(min_len, min(max_len, L) + 1):
            for i in range(0, L - n + 1):
                tokens.append(seg[i:i + n])
    return tokens


def tokenize(q: str, min_cn_len: int = 2) -> tuple:
    """将查询文本转换为检索令牌（中文短词+英文单词）"""
    q_norm = normalize_text(q)
    basic = [t for t in q_norm.split(" ") if t]
    cn_short = extract_cn_tokens(q, min_len=min_cn_len, max_len=4)
    tokens = basic + cn_short
    seen, out = set(), []
    for t in tokens:
        if t and t not in seen:
            out.append(t)
            seen.add(t)
    return out, q_norm
