# -*- coding: utf-8 -*-
"""Word 文档解析辅助函数：表格转换、关键词提取、元数据清洗"""

import json
import re
from typing import List, Dict, Any

from docx.text.paragraph import Paragraph
from docx.table import Table as DocxTable
from docx import Document as DocxDocument

from utils.text_utils import cheap_keywords


def sanitize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """清洗元数据（确保所有值可序列化，避免存储到向量库时出错）"""
    cleaned = {}
    for k, v in (meta or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            cleaned[k] = v
        else:
            cleaned[k] = json.dumps(v, ensure_ascii=False)
    return cleaned


def iter_block_items(doc: DocxDocument):
    """
    迭代 Word 文档中的所有块（段落和表格），保持原始顺序
    :yield: Paragraph 或 DocxTable 对象
    """
    parent_elm = doc.element.body
    for child in parent_elm.iterchildren():
        if child.tag.endswith('}p'):
            yield Paragraph(child, doc)
        elif child.tag.endswith('}tbl'):
            yield DocxTable(child, doc)


def word_table_to_markdown(tbl: DocxTable, assume_first_row_header: bool = True) -> str:
    """将 Word 表格转换为 Markdown 格式"""
    rows = []
    for r in tbl.rows:
        cells = [(c.text or "").strip().replace("\n", " ") for c in r.cells]
        rows.append(cells)
    if not rows:
        return ""

    max_cols = max(len(r) for r in rows)
    rows = [r + [""] * (max_cols - len(r)) for r in rows]

    header = rows[0] if assume_first_row_header else [f"col{i + 1}" for i in range(max_cols)]
    body = rows[1:] if assume_first_row_header else rows

    md = []
    md.append("| " + " | ".join(h or f"col{i + 1}" for i, h in enumerate(header)) + " |")
    md.append("| " + " | ".join("---" for _ in range(max_cols)) + " |")
    for r in body:
        md.append("| " + " | ".join(r) + " |")
    return "\n".join(md)


def table_first_col_keywords(tbl: DocxTable) -> List[str]:
    """提取表格第一列内容作为关键词（增强表格内容的检索召回率）"""
    out = []
    if not tbl.rows:
        return out
    for r in tbl.rows[1:]:
        cells = [(c.text or "").strip().replace("\n", " ") for c in r.cells]
        if cells and cells[0]:
            out.append(cells[0])
    return out


def bold_runs_to_keywords(paragraph: Paragraph) -> List[str]:
    """提取段落中加粗的文本作为关键词"""
    bold_texts: List[str] = []
    try:
        for run in paragraph.runs:
            is_bold = False
            if getattr(run, "bold", None):
                is_bold = True
            if hasattr(run, "font") and getattr(run.font, "bold", None):
                is_bold = True
            if is_bold:
                t = (run.text or "").strip()
                if t:
                    bold_texts.append(t)
    except Exception:
        pass
    if not bold_texts:
        return []
    merged = " ".join(bold_texts)
    return cheap_keywords(merged, topn=24)
