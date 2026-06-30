# -*- coding: utf-8 -*-
"""Word 文档解析：按标题层级分割为结构化文本块，保留表格和关键词"""

import os
import re
from typing import List, Dict

from docx import Document as DocxDocument
from docx.text.paragraph import Paragraph
from docx.table import Table as DocxTable
from langchain_core.documents import Document

from utils.text_utils import cheap_keywords
from document_processing.word_parser_helpers import (
    iter_block_items, word_table_to_markdown,
    table_first_col_keywords, bold_runs_to_keywords,
)
from document_processing.pdf_parser import _apply_chunk_size_guard


def parse_docx_to_chunks(file_path: str) -> List[Document]:
    """
    将 Word 文档解析为结构化文本块（按标题层级划分，保留表格和关键词）
    :param file_path: Word 文档路径（处理后的文档，图片已替换为占位符）
    :return: LangChain Document 对象列表
    """
    docs: List[Document] = []
    docx = DocxDocument(file_path)

    current_headings: Dict[int, str] = {}
    buffer: List[str] = []
    table_buf: List[str] = []
    table_kws: List[str] = []
    bold_kws_buf: List[str] = []

    def current_label() -> str:
        """生成当前文本块的标签（标题层级拼接）"""
        return " > ".join(current_headings[i] for i in sorted(current_headings.keys()))

    def flush():
        """将缓存的内容打包为 Document 对象并添加到列表"""
        nonlocal buffer, table_buf, table_kws, bold_kws_buf
        if not current_headings:
            buffer, table_buf, table_kws, bold_kws_buf = [], [], [], []
            return

        label = current_label()
        body = "\n".join(buffer).strip()
        text = body + ("\n".join(table_buf) if table_buf else "")

        kws = cheap_keywords(text)
        kws.extend(table_kws)
        kws.extend(bold_kws_buf)
        kws = list(dict.fromkeys(kws))

        if text.strip():
            # 应用 size guard（超大块二次切分，保证 embedding 质量）
            sub_results = _apply_chunk_size_guard(
                full_text=text,
                label=label,
                uploaded_name=os.path.basename(file_path),
                keywords=kws,
            )
            for doc, _entry in sub_results:
                docs.append(doc)

        buffer, table_buf, table_kws, bold_kws_buf = [], [], [], []

    for block in iter_block_items(docx):
        if isinstance(block, Paragraph):
            style = (block.style.name if block.style else "")
            if re.match(r"^(Heading|标题)\s*\d+$", style, re.I):
                flush()
                lvl = int(re.findall(r"(\d+)", style)[0])
                current_headings[lvl] = (block.text or "").strip()
                for k in list(current_headings.keys()):
                    if k > lvl:
                        del current_headings[k]
            else:
                if current_headings:
                    txt = (block.text or "").strip()
                    if txt:
                        buffer.append(txt)
                    bold_kws = bold_runs_to_keywords(block)
                    if bold_kws:
                        bold_kws_buf.extend(bold_kws)
        else:
            md = word_table_to_markdown(block, assume_first_row_header=True)
            if md:
                table_buf.append(f"\n[[TABLE_MD]]\n{md}\n[[/TABLE_MD]]\n")
            table_kws.extend(table_first_col_keywords(block))

    flush()
    return docs
