# -*- coding: utf-8 -*-
"""知识库管理器：构建、列表、删除 —— 支持多文件夹导入"""

import os
import json
import tempfile
from typing import List, Dict, Any, Tuple

import streamlit as st
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredMarkdownLoader, PyMuPDFLoader

from config import OLLAMA_BASE_URL
from embeddings.manager import create_embeddings
from utils.file_utils import ensure_dirs_for_kb, kb_dir, kb_summary_json
from knowledge_base.vector_store import (
    build_vectorstore_for_kb, add_documents_to_vs, filter_complex_metadata,
)
from knowledge_base.index_manager import (
    write_index_and_prompt, generate_kb_summary,
)


def build_knowledge_base_from_files(
    uploaded_files,
    kb_name: str,
    chunk_size: int = 600,
    chunk_overlap: int = 80,
    llm_provider: str = "ollama",
    llm_model: str = "",
    openai_api_key: str = "",
    openai_base_url: str = "",
) -> bool:
    """
    从上传的文件列表构建知识库（Streamlit 文件上传器接口）
    """
    try:
        if not kb_name or not kb_name.strip():
            st.error("请先填写知识库名称。")
            return False

        ensure_dirs_for_kb(kb_name)

        content_docs: List[Document] = []
        index_entries: List[Dict[str, Any]] = []

        progress_bar = st.progress(0)
        total = len(uploaded_files)
        for i, f in enumerate(uploaded_files):
            st.write(f"📄 处理中: _{f.name}_")
            ext = os.path.splitext(f.name)[1].lower()
            docs, idx = _process_single_file(f, ext, kb_name, chunk_size, chunk_overlap)
            content_docs.extend(docs)
            index_entries.extend(idx)
            progress_bar.progress(
                (i + 1) / total,
                text=f"处理中 ({i + 1}/{total}): {f.name}",
            )

        if not content_docs:
            st.warning("没有加载任何有效文档。")
            return False

        # 过滤复杂元数据
        if filter_complex_metadata:
            content_docs = filter_complex_metadata(content_docs)

        # 存储到向量库
        embeddings = create_embeddings()
        vs = build_vectorstore_for_kb(embeddings, kb_name)
        add_documents_to_vs(vs, content_docs)

        # 写入索引和提示词
        write_index_and_prompt(kb_name, index_entries, content_docs)

        # 生成总结
        with st.spinner("正在生成知识库主要内容总结..."):
            summary = generate_kb_summary(
                content_docs,
                provider=llm_provider,
                model_name=llm_model,
                openai_api_key=openai_api_key,
                openai_base_url=openai_base_url,
            )
            with open(kb_summary_json(kb_name), "w", encoding="utf-8") as f:
                json.dump({"summary": summary}, f, ensure_ascii=False, indent=2)

        st.success(f"知识库『{kb_name}』构建完成：内容块 {len(content_docs)} 条，索引 {len(index_entries)} 条。")
        st.cache_resource.clear()
        st.cache_data.clear()
        return True

    except Exception as e:
        st.error(f"知识库构建过程中发生错误: {e}")
        import traceback
        st.code(traceback.format_exc())
        return False


def build_knowledge_base_from_folders(
    folder_paths: List[str],
    kb_name: str,
    chunk_size: int = 600,
    chunk_overlap: int = 80,
    file_extensions: Tuple[str, ...] = (".docx", ".pdf", ".md"),
    llm_provider: str = "ollama",
    llm_model: str = "",
    openai_api_key: str = "",
    openai_base_url: str = "",
) -> bool:
    """
    从多个本地文件夹批量导入文档构建知识库（新增：多文件夹支持）
    :param folder_paths: 源文件夹路径列表
    :param kb_name: 知识库名称
    :param chunk_size: PDF/MD 文本块大小
    :param chunk_overlap: PDF/MD 文本块重叠大小
    :param file_extensions: 支持的文件扩展名
    :return: 是否成功
    """
    try:
        if not kb_name or not kb_name.strip():
            st.error("请先填写知识库名称。")
            return False
        if not folder_paths:
            st.error("请提供至少一个源文件夹路径。")
            return False

        ensure_dirs_for_kb(kb_name)

        content_docs: List[Document] = []
        index_entries: List[Dict[str, Any]] = []

        # 收集所有文件
        all_files: List[str] = []
        for folder in folder_paths:
            if not os.path.isdir(folder):
                st.warning(f"跳过无效目录: {folder}")
                continue
            for root, dirs, files in os.walk(folder):
                for fname in files:
                    if any(fname.lower().endswith(ext) for ext in file_extensions):
                        all_files.append(os.path.join(root, fname))

        if not all_files:
            st.warning("未在指定文件夹中找到支持的文档（.docx/.pdf/.md）。")
            return False

        with st.spinner(f"正在从 {len(folder_paths)} 个文件夹构建知识库『{kb_name}』（共 {len(all_files)} 个文件）..."):
            progress_bar = st.progress(0)
            for i, file_path in enumerate(all_files):
                ext = os.path.splitext(file_path)[1].lower()
                # 创建模拟的上传文件对象
                with open(file_path, "rb") as fh:
                    file_data = fh.read()
                # 使用 NamedTemporaryFile 包装
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(file_data)
                    tmp_path = tmp.name

                class MockUploadedFile:
                    def __init__(self, name, path):
                        self.name = name
                        self._path = path
                    def getvalue(self):
                        with open(self._path, "rb") as f:
                            return f.read()

                mock_file = MockUploadedFile(os.path.basename(file_path), tmp_path)
                docs, idx = _process_single_file(mock_file, ext, kb_name, chunk_size, chunk_overlap)
                content_docs.extend(docs)
                index_entries.extend(idx)
                os.unlink(tmp_path)

                progress_bar.progress((i + 1) / len(all_files), text=f"处理中: {os.path.basename(file_path)}")

        if not content_docs:
            st.warning("没有加载任何有效文档。")
            return False

        if filter_complex_metadata:
            content_docs = filter_complex_metadata(content_docs)

        embeddings = create_embeddings()
        vs = build_vectorstore_for_kb(embeddings, kb_name)
        add_documents_to_vs(vs, content_docs)

        write_index_and_prompt(kb_name, index_entries, content_docs)

        with st.spinner("正在生成知识库主要内容总结..."):
            summary = generate_kb_summary(
                content_docs,
                provider=llm_provider,
                model_name=llm_model,
                openai_api_key=openai_api_key,
                openai_base_url=openai_base_url,
            )
            with open(kb_summary_json(kb_name), "w", encoding="utf-8") as f:
                json.dump({"summary": summary}, f, ensure_ascii=False, indent=2)

        st.success(f"知识库『{kb_name}』构建完成：从 {len(folder_paths)} 个文件夹导入 {len(all_files)} 个文件，"
                   f"生成 {len(content_docs)} 个内容块，{len(index_entries)} 条索引。")
        st.cache_resource.clear()
        st.cache_data.clear()
        return True

    except Exception as e:
        st.error(f"知识库构建过程中发生错误: {e}")
        import traceback
        st.code(traceback.format_exc())
        return False


def _process_single_file(f, ext: str, kb_name: str, chunk_size: int, chunk_overlap: int) -> Tuple[List[Document], List[Dict[str, Any]]]:
    """处理单个文件，返回 (docs, index_entries)"""
    from document_processing.image_processor import (
        process_single_docx_to_placeholders, process_pdf_with_images_for_kb,
        process_markdown_with_images_for_kb,
    )
    from document_processing.word_parser import parse_docx_to_chunks
    from document_processing.pdf_parser import (
        parse_pdf_with_toc_to_chunks, process_unstructured, parse_markdown_to_chunks,
    )

    if ext == ".docx":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(f.getvalue())
            tmp_path = tmp.name
        processed_path = process_single_docx_to_placeholders(tmp_path, kb_name)
        os.unlink(tmp_path)
        docs = parse_docx_to_chunks(processed_path)
        idx = []
        for d in docs:
            meta = d.metadata or {}
            idx.append({
                "chunk_uid": meta.get("chunk_uid", ""),
                "label": meta.get("label", ""),
                "source": meta.get("source", os.path.basename(processed_path)),
                "keywords": meta.get("keywords", []),
                "keywords_text": meta.get("keywords_text", ""),
                "char_count": len(d.page_content or ""),
            })
        return docs, idx

    elif ext == ".pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.getvalue())
            tmp_path = tmp.name
        processed_pdf = process_pdf_with_images_for_kb(tmp_path, kb_name)
        os.unlink(tmp_path)
        docs, idx = parse_pdf_with_toc_to_chunks(processed_pdf, os.path.basename(processed_pdf))
        if not docs:
            loader = PyMuPDFLoader(processed_pdf)
            docs, idx = process_unstructured(loader, os.path.basename(processed_pdf), chunk_size, chunk_overlap)
        return docs, idx

    elif ext in (".md",):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmp:
            tmp.write(f.getvalue())
            tmp_path = tmp.name
        # 1. 提取本地图片并用占位符替换
        processed_md = process_markdown_with_images_for_kb(tmp_path, kb_name)
        os.unlink(tmp_path)
        # 2. 按 H1~H6 标题层级结构化切分
        docs, idx = parse_markdown_to_chunks(processed_md, os.path.basename(processed_md))
        return docs, idx

    else:
        return [], []
