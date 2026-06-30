# -*- coding: utf-8 -*-
"""PDF 文档解析：按书签层级分割 + 无结构回退解析，保留表格"""

import os
import re
from typing import List, Dict, Any, Tuple
from uuid import uuid4
from difflib import SequenceMatcher

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

from config import MAX_CHUNK_CHARS, MIN_CHUNK_CHARS, CHUNK_OVERLAP_CHARS
from utils.text_utils import cheap_keywords
from document_processing.word_parser_helpers import sanitize_meta


# ==================== 共享 Chunk Size Guard ====================

def _apply_chunk_size_guard(
    full_text: str,
    label: str,
    uploaded_name: str,
    keywords: list,
    chunk_size: int = MAX_CHUNK_CHARS,
    chunk_overlap: int = CHUNK_OVERLAP_CHARS,
    min_chunk_chars: int = MIN_CHUNK_CHARS,
) -> list:
    """
    对超大文本块应用 RecursiveCharacterTextSplitter 二次切分，保证 embedding 质量

    策略：
    1. 若 full_text <= chunk_size → 原样返回单元素列表
    2. 若 full_text > chunk_size → 用 RecursiveCharacterTextSplitter 切分为子块，
       每个子块继承父级 label 并追加 "（1/N）" 序号后缀
    3. 过短的尾部子块（< min_chunk_chars）合并到前一个子块

    :param full_text: 待切分的完整文本
    :param label: 父级层级标签（如 "H1 > H2"）
    :param uploaded_name: 源文件名
    :param keywords: 父级关键词列表（子块继承并补充局部关键词）
    :param chunk_size: 最大字符数阈值
    :param chunk_overlap: 二次切分重叠字符数
    :param min_chunk_chars: 最小字符数阈值（尾部小块合并）
    :return: [(doc, index_entry), ...] 列表，每个元素为 (Document, dict)
    """
    from langchain_core.documents import Document
    from uuid import uuid4

    out: list = []

    if len(full_text) <= chunk_size:
        # 大小正常，直接产出
        uid = str(uuid4())
        meta = {
            "label": label,
            "hierarchy": label,
            "source": uploaded_name,
            "chunk_uid": uid,
            "keywords": keywords,
            "keywords_text": ", ".join(keywords),
        }
        injected = f"[LABEL] {label}\n[KEYWORDS] {', '.join(keywords)}\n{full_text}"
        doc = Document(page_content=injected, metadata=sanitize_meta(meta))
        entry = {
            "chunk_uid": uid,
            "label": label,
            "source": uploaded_name,
            "keywords": keywords,
            "keywords_text": ", ".join(keywords),
            "char_count": len(full_text),
        }
        out.append((doc, entry))
        return out

    # 超大块 → 二次切分
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " "],
    )
    sub_texts = splitter.split_text(full_text)

    # 合并过短的尾部小块
    if len(sub_texts) >= 2 and len(sub_texts[-1]) < min_chunk_chars:
        sub_texts[-2] = sub_texts[-2] + "\n" + sub_texts[-1]
        sub_texts.pop()

    total = len(sub_texts)
    for i, sub_text in enumerate(sub_texts):
        sub_label = f"{label}（{i + 1}/{total}）" if total > 1 else label
        sub_kw = cheap_keywords(sub_text, topn=12)
        # 合并父级关键词 + 子块局部关键词，去重
        merged_kw = list(dict.fromkeys(keywords + sub_kw))

        uid = str(uuid4())
        meta = {
            "label": sub_label,
            "hierarchy": label,  # 保留原始层级
            "source": uploaded_name,
            "chunk_uid": uid,
            "keywords": merged_kw,
            "keywords_text": ", ".join(merged_kw),
        }
        injected = f"[LABEL] {sub_label}\n[KEYWORDS] {', '.join(merged_kw)}\n{sub_text}"
        doc = Document(page_content=injected, metadata=sanitize_meta(meta))
        entry = {
            "chunk_uid": uid,
            "label": sub_label,
            "source": uploaded_name,
            "keywords": merged_kw,
            "keywords_text": ", ".join(merged_kw),
            "char_count": len(sub_text),
        }
        out.append((doc, entry))

    return out


# ==================== 辅助函数 ====================

def pdf_table_to_markdown(tab, assume_first_row_header: bool = True) -> str:
    """将 PyMuPDF 提取的表格转换为 Markdown 格式"""
    if not tab.extract():
        return ""
    rows = tab.extract()
    if not rows:
        return ""

    max_cols = max(len(r) for r in rows)
    rows = [r + [""] * (max_cols - len(r)) for r in rows]

    header = rows[0] if assume_first_row_header else [f"col{i + 1}" for i in range(max_cols)]
    body = rows[1:] if assume_first_row_header else rows

    md = []
    md.append("| " + " | ".join(str(h) if h is not None else "" for h in header) + " |")
    md.append("| " + " | ".join("---" for _ in range(max_cols)) + " |")
    for r in body:
        md.append("| " + " | ".join(str(c) if c is not None else "" for c in r) + " |")
    return "\n".join(md)


def pdf_table_first_col_keywords(tab) -> List[str]:
    """提取 PDF 表格第一列内容作为关键词"""
    out = []
    rows = tab.extract()
    if not rows:
        return out
    for r in rows[1:]:
        if r and len(r) >= 1:
            cell_text = str(r[0]).strip()
            if cell_text:
                out.append(cell_text)
    merged = " ".join(out)
    return cheap_keywords(merged, topn=10)


# ==================== 主解析函数（带书签层级） ====================

def parse_pdf_with_toc_to_chunks(pdf_path: str, uploaded_name: str) -> Tuple[List[Document], List[Dict[str, Any]]]:
    """按 PDF 书签层级解析为结构化文本块"""
    doc = fitz.open(pdf_path)
    toc = doc.get_toc(simple=False)
    if not toc:
        doc.close()
        return [], []

    out_docs: List[Document] = []
    index_entries: List[Dict[str, Any]] = []

    current_headings: Dict[int, str] = {}
    content_buf: List[str] = []
    prev_table = None
    table_kws: List[str] = []
    title_kws_buf: List[str] = []

    def current_label() -> str:
        return " > ".join(current_headings[i] for i in sorted(current_headings.keys()))

    def flush():
        nonlocal content_buf, table_kws, title_kws_buf, prev_table
        if not current_headings:
            content_buf, table_kws, title_kws_buf, prev_table = [], [], [], None
            return

        label = current_label()
        full_text = "\n".join(content_buf).strip()

        kws = cheap_keywords(full_text)
        kws.extend(table_kws)
        kws.extend(title_kws_buf)
        kws = list(dict.fromkeys(kws))

        if full_text.strip():
            level = max(current_headings.keys()) if current_headings else 0
            # 应用 size guard（超大块二次切分，保证 embedding 质量）
            sub_results = _apply_chunk_size_guard(
                full_text=full_text,
                label=label,
                uploaded_name=uploaded_name,
                keywords=kws,
            )
            for doc, entry in sub_results:
                entry["level"] = level  # 保留 PDF 层级信息
                out_docs.append(doc)
                index_entries.append(entry)

        content_buf, table_kws, title_kws_buf, prev_table = [], [], [], None

    def is_continuous_table(prev_tab, curr_tab, prev_page, curr_page) -> bool:
        """判断两个表格是否为跨页连续表格"""
        if len(prev_tab.extract()[0]) != len(curr_tab.extract()[0]):
            return False
        prev_page_h = prev_page.rect.height
        prev_tab_bottom = prev_tab.bbox[3]
        curr_tab_top = curr_tab.bbox[1]
        if not (prev_page_h - prev_tab_bottom < 60 and curr_tab_top < 60):
            return False
        prev_last_row = " ".join([str(c) for c in prev_tab.extract()[-1] if c])
        curr_first_row = " ".join([str(c) for c in curr_tab.extract()[0] if c])
        return SequenceMatcher(None, prev_last_row, curr_first_row).ratio() > 0.3

    def table_to_markdown_internal(tab) -> str:
        if isinstance(tab, list):
            if not tab:
                return ""
            header = "| " + " | ".join(map(str, tab[0])) + " |"
            separator = "| " + " | ".join(["---"] * len(tab[0])) + " |"
            rows = [header, separator] + ["| " + " | ".join(map(str, row)) + " |" for row in tab[1:]]
            return "\n".join(rows)
        else:
            return pdf_table_to_markdown(tab)

    for i in range(len(toc)):
        level, title, page_num, _ = toc[i]
        start_page = page_num - 1 if page_num else 0
        if i + 1 < len(toc):
            next_page_num = toc[i + 1][2]
            end_page = next_page_num - 1 - 1 if next_page_num else len(doc) - 1
        else:
            end_page = len(doc) - 1
        end_page = max(start_page, end_page)

        flush()

        title_text = title.strip()
        current_headings[level] = title_text
        for k in list(current_headings.keys()):
            if k > level:
                del current_headings[k]
        title_kws_buf = cheap_keywords(title_text, topn=5)

        for p in range(start_page, end_page + 1):
            page = doc.load_page(p)
            full_rect = page.rect
            clip_rect = fitz.Rect(full_rect.x0, full_rect.y0 + 60, full_rect.x1, full_rect.y1 - 75)

            tabs = page.find_tables(clip=clip_rect, strategy="lines").tables
            current_tables = []
            temp_elements = []

            if prev_table and tabs:
                if is_continuous_table(prev_table, tabs[0], doc.load_page(p - 1), page):
                    merged_data = prev_table.extract() + tabs[0].extract()[1:]
                    current_tables.append(merged_data)
                    current_tables.extend(tabs[1:])
                    prev_table = None
                else:
                    table_md = table_to_markdown_internal(prev_table)
                    if table_md:
                        prev_table_y0 = prev_table.bbox[1]
                        temp_elements.append((prev_table_y0, f"\n[[TABLE_MD]]\n{table_md}\n[[/TABLE_MD]]\n"))
                    current_tables = tabs
            else:
                current_tables = tabs

            if current_tables:
                prev_table = current_tables[-1] if p < end_page else None

            for tab in current_tables:
                if isinstance(tab, list):
                    table_y0 = tabs[0].bbox[1] if tabs else 0
                else:
                    table_y0 = tab.bbox[1]
                table_md = table_to_markdown_internal(tab)
                if table_md:
                    table_content = f"\n[[TABLE_MD]]\n{table_md}\n[[/TABLE_MD]]\n"
                    temp_elements.append((table_y0, table_content))
                    table_data = tab if isinstance(tab, list) else tab.extract()
                    first_col = [row[0] for row in table_data if row]
                    first_col_text = " ".join([str(c) for c in first_col if c])
                    table_kws.extend(cheap_keywords(first_col_text, topn=5))

            blocks = page.get_text("blocks", clip=clip_rect)
            table_rects = [
                fitz.Rect(t.bbox) if not isinstance(t, list) else fitz.Rect(0, 0, 0, 0)
                for t in current_tables
            ]
            for b in blocks:
                block_rect = fitz.Rect(b[:4])
                if not any(block_rect.intersects(rect) for rect in table_rects):
                    text = b[4].strip()
                    if text:
                        text_y0 = block_rect.y0
                        temp_elements.append((text_y0, text))

            temp_elements.sort(key=lambda x: -x[0])
            for elem in temp_elements:
                content_buf.append(elem[1])

        if prev_table:
            table_md = table_to_markdown_internal(prev_table)
            if table_md:
                table_content = f"\n[[TABLE_MD]]\n{table_md}\n[[/TABLE_MD]]\n"
                content_buf.append(table_content)
            prev_table = None

    flush()
    doc.close()
    return out_docs, index_entries


# ==================== 无结构回退解析 ====================

def _chunk_text_to_docs(
    full_text: str,
    uploaded_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[List[Document], List[Dict[str, Any]]]:
    """将纯文本按 chunk_size/chunk_overlap 切分为 Document 列表（Markdown 通用路径）"""
    out_docs: List[Document] = []
    index_entries: List[Dict[str, Any]] = []

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(full_text)

    for chunk in chunks:
        label = chunk.splitlines()[0][:100] if chunk.splitlines() else uploaded_name
        kw = cheap_keywords(chunk, topn=12)
        uid = str(uuid4())
        meta = {
            "label": label,
            "hierarchy": label,
            "source": uploaded_name,
            "chunk_uid": uid,
            "keywords": kw,
            "keywords_text": ", ".join(kw),
        }
        injected = f"[LABEL] {label}\n[KEYWORDS] {', '.join(kw)}\n{chunk}"
        out_docs.append(Document(page_content=injected, metadata=sanitize_meta(meta)))
        index_entries.append({
            "chunk_uid": uid,
            "label": label,
            "source": uploaded_name,
            "keywords": kw,
            "keywords_text": ", ".join(kw),
            "char_count": len(injected),
        })

    return out_docs, index_entries


# ==================== Markdown 结构化解析（按标题层级）====================

def parse_markdown_to_chunks(md_path: str, uploaded_name: str) -> Tuple[List[Document], List[Dict[str, Any]]]:
    """
    使用 MarkdownHeaderTextSplitter 按标题层级（H1~H6）结构化切分 Markdown 文档

    与 Word（按 Heading 层级）和 PDF（按书签层级）不同，此函数使用 LangChain 的
    MarkdownHeaderTextSplitter，能识别 # ~ ###### 标题并自动填充层级元数据。

    :param md_path: Markdown 文件路径
    :param uploaded_name: 上传文件名（用于 source 字段）
    :return: (docs, index_entries)
    """
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        return [], []

    headers_to_split_on = [
        ("#", "H1"),
        ("##", "H2"),
        ("###", "H3"),
        ("####", "H4"),
        ("#####", "H5"),
        ("######", "H6"),
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,  # 保留标题在正文中
    )

    md_docs = splitter.split_text(text)

    out_docs: List[Document] = []
    index_entries: List[Dict[str, Any]] = []

    for doc in md_docs:
        meta_headers = doc.metadata or {}
        # 按 H1 > H2 > H3 > ... 顺序拼接层级标签
        hierarchy_parts = []
        for level in ["H1", "H2", "H3", "H4", "H5", "H6"]:
            val = meta_headers.get(level, "").strip()
            if val:
                hierarchy_parts.append(val)
        label = " > ".join(hierarchy_parts) if hierarchy_parts else uploaded_name

        chunk_text = (doc.page_content or "").strip()
        if not chunk_text:
            continue

        kw = cheap_keywords(chunk_text, topn=12)

        # 应用 size guard（超大块二次切分，保证 embedding 质量）
        sub_results = _apply_chunk_size_guard(
            full_text=chunk_text,
            label=label,
            uploaded_name=uploaded_name,
            keywords=kw,
        )
        for doc, entry in sub_results:
            # 合并标题层级元数据到每个子块
            for level in ["H1", "H2", "H3", "H4", "H5", "H6"]:
                if level in meta_headers:
                    doc.metadata[f"md_{level.lower()}"] = meta_headers[level]
            out_docs.append(doc)
            index_entries.append(entry)

    return out_docs, index_entries


def _process_pdf_unstructured(
    pdf_path: str,
    uploaded_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[List[Document], List[Dict[str, Any]]]:
    """无结构 PDF 回退解析：逐页提取文本+表格，机械分割"""
    import fitz
    doc = fitz.open(pdf_path)

    out_docs: List[Document] = []
    index_entries: List[Dict[str, Any]] = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        full_rect = page.rect
        clip_rect = fitz.Rect(full_rect.x0, full_rect.y0 + 43, full_rect.x1, full_rect.y1 - 50)

        tabs = page.find_tables(clip=clip_rect, strategy="lines")
        table_buf = []
        for tab in tabs.tables:
            md = pdf_table_to_markdown(tab, assume_first_row_header=True)
            if md:
                table_buf.append(f"\n[[TABLE_MD]]\n{md}\n[[/TABLE_MD]]\n")

        table_rects = [tab.bbox for tab in tabs.tables]
        blocks = page.get_text("blocks", clip=clip_rect)
        text = ""
        for b in blocks:
            block_rect = fitz.Rect(b[:4])
            if not any(block_rect.intersects(tr) for tr in table_rects):
                text += b[4] + "\n"
        text = text.strip()

        full_text = text + "\n".join(table_buf) if table_buf else text

        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(full_text)

        for chunk in chunks:
            label = chunk.splitlines()[0][:100] if chunk.splitlines() else uploaded_name
            kw = cheap_keywords(chunk, topn=12)
            uid = str(uuid4())
            meta = {
                "label": label,
                "hierarchy": label,
                "source": uploaded_name,
                "chunk_uid": uid,
                "keywords": kw,
                "keywords_text": ", ".join(kw),
            }
            injected = f"[LABEL] {label}\n[KEYWORDS] {', '.join(kw)}\n{chunk}"
            out_docs.append(Document(page_content=injected, metadata=sanitize_meta(meta)))
            index_entries.append({
                "chunk_uid": uid,
                "label": label,
                "source": uploaded_name,
                "keywords": kw,
                "keywords_text": ", ".join(kw),
                "char_count": len(injected),
            })

    doc.close()
    return out_docs, index_entries


def process_unstructured(loader, uploaded_name: str, chunk_size: int, chunk_overlap: int) -> Tuple[List[Document], List[Dict[str, Any]]]:
    """
    无结构 PDF/Markdown 回退解析：机械分割

    根据文件类型分流：
    - PDF（无书签/目录结构）：逐页提取文本+表格，按 chunk_size 分割
    - Markdown / 纯文本：通过 loader 加载后按 chunk_size 分割
    """
    file_path = getattr(loader, "file_path", "")
    ext = os.path.splitext(file_path)[1].lower() if file_path else ""

    if ext in (".md", ".mdx"):
        # Markdown 路径：优先使用结构化解析（按 H1~H6 标题层级切分）
        return parse_markdown_to_chunks(file_path, uploaded_name)
    elif ext in (".txt", ".rst"):
        # 纯文本路径：通过 loader 加载全文后机械分割
        docs = loader.load()
        full_text = "\n\n".join(d.page_content for d in docs if d.page_content)
        return _chunk_text_to_docs(full_text, uploaded_name, chunk_size, chunk_overlap)
    else:
        # PDF 路径（无书签回退）：逐页提取 + 表格识别
        return _process_pdf_unstructured(file_path, uploaded_name, chunk_size, chunk_overlap)
