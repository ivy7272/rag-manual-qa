# -*- coding: utf-8 -*-
"""渲染模块：文本内容中的图片占位符替换、表格 Markdown 渲染"""

import os
import re
from typing import Dict

import streamlit as st


IMG_TAG_RE = re.compile(r"<img>\s*([^<>\s]+)\s*</img>", re.I)
TABLE_BLOCK_RE = re.compile(r"\[\[TABLE_MD\]\](.*?)\[\[/TABLE_MD\]\]", re.S | re.I)


def render_text_with_images(
    raw_text: str,
    image_maps_by_doc: Dict[str, Dict[str, str]],
    source_filename: str,
):
    """
    渲染文本内容（替换图片占位符为实际图片，识别并渲染 Markdown 表格）
    :param raw_text: 原始文本（包含图片占位符和表格标记）
    :param image_maps_by_doc: 图片映射字典 {文档基名: {图片ID: 路径}}
    :param source_filename: 来源文件名（用于定位图片映射）
    """
    # 提取来源文档的基名
    doc_base = os.path.splitext(os.path.basename(source_filename or ""))[0]
    candidates = [doc_base]
    if doc_base.endswith("_with_placeholders"):
        candidates.append(doc_base.replace("_with_placeholders", ""))

    # 查找对应的图片映射
    image_map: Dict[str, str] = {}
    for key in candidates:
        if key in image_maps_by_doc:
            image_map = image_maps_by_doc[key]
            break
    if not image_map and len(image_maps_by_doc) == 1:
        image_map = next(iter(image_maps_by_doc.values()))

    # 移除标签和关键词前缀
    lines = [
        ln for ln in (raw_text or "").splitlines()
        if not ln.startswith("[LABEL]") and not ln.startswith("[KEYWORDS]")
    ]
    text = "\n".join(lines).strip()

    # 分段处理表格和普通文本
    pos = 0
    for m in TABLE_BLOCK_RE.finditer(text):
        before = text[pos:m.start()]
        _render_text_and_images(before, image_map)
        table_md = m.group(1).strip()
        if table_md:
            st.markdown(table_md)
        pos = m.end()
    tail = text[pos:]
    _render_text_and_images(tail, image_map)


def _render_text_and_images(text: str, image_map: Dict[str, str]):
    """辅助函数：渲染文本和图片（替换占位符）"""
    if not text:
        return
    parts = re.split(r"(<img>.*?</img>)", text, flags=re.S | re.I)
    for part in parts:
        if not part:
            continue
        m = IMG_TAG_RE.match(part.strip())
        if not m:
            if part.strip():
                st.markdown(
                    f"<div style='white-space:pre-wrap'>{part}</div>",
                    unsafe_allow_html=True,
                )
        else:
            img_id = m.group(1).strip()
            path = image_map.get(img_id)
            if path and os.path.exists(path):
                st.image(path, use_container_width=True, caption=img_id)
            else:
                st.warning(f"找不到图片：{img_id}")
