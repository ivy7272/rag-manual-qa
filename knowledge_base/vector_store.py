# -*- coding: utf-8 -*-
"""Chroma 向量库操作：构建、查询、文档获取"""

from typing import List, Dict
from langchain_core.documents import Document

from config import COLLECTION_NAME

# Chroma 导入兼容
try:
    from langchain_chroma import Chroma
    from chromadb import PersistentClient
    NEW_CHROMA = True
except Exception:
    from langchain_community.vectorstores import Chroma  # type: ignore
    PersistentClient = None
    NEW_CHROMA = False


def build_vectorstore_for_kb(embeddings, kb_name: str) -> Chroma:
    """
    为指定知识库构建 Chroma 向量库（兼容新旧版 Chroma）
    """
    from utils.file_utils import kb_dir
    path = kb_dir(kb_name)
    if NEW_CHROMA and PersistentClient:
        client = PersistentClient(path=path)
        return Chroma(client=client, collection_name=COLLECTION_NAME, embedding_function=embeddings)
    else:
        return Chroma(persist_directory=path, collection_name=COLLECTION_NAME, embedding_function=embeddings)


def get_collection_from_vs(vs: Chroma):
    """从 Chroma 向量库实例中获取底层集合对象"""
    return getattr(vs, "_collection", None) or getattr(vs, "collection", None)


def add_documents_to_vs(vs: Chroma, docs: List[Document]) -> None:
    """添加文档到向量库并持久化"""
    ids = [(d.metadata or {}).get("chunk_uid", "") for d in docs]
    vs.add_documents(docs, ids=ids)
    if hasattr(vs, "persist"):
        vs.persist()


def fetch_docs_by_chunk_uids(vs: Chroma, uids: List[str]) -> List[Document]:
    """
    通过 chunk_uid 从向量库中精确获取文档（回表操作）
    使用批量查询（$in 操作符），一次请求获取所有文档
    """
    col = get_collection_from_vs(vs)
    if not col or not uids:
        return []

    uid_to_doc: Dict[str, Document] = {}

    try:
        res = col.get(
            where={"chunk_uid": {"$in": list(uids)}},
            include=["documents", "metadatas"],
        )
        docs_raw = res.get("documents", []) or []
        metas_raw = res.get("metadatas", []) or []
        for d, m in zip(docs_raw, metas_raw):
            uid = (m or {}).get("chunk_uid", "")
            if uid:
                uid_to_doc[uid] = Document(page_content=d or "", metadata=m or {})
    except Exception:
        # 批量查询失败时回退到逐个查询
        for uid in uids:
            try:
                res = col.get(
                    where={"chunk_uid": {"$eq": uid}},
                    limit=1,
                    include=["documents", "metadatas"],
                )
                docs = [
                    Document(page_content=d or "", metadata=m or {})
                    for d, m in zip(res.get("documents", []) or [], res.get("metadatas", []) or [])
                ]
                if docs:
                    uid_to_doc[uid] = docs[0]
            except Exception:
                continue

    # 保持输入顺序
    return [uid_to_doc[uid] for uid in uids if uid in uid_to_doc]


# 复杂元数据过滤（可选）
try:
    from langchain_community.vectorstores.utils import filter_complex_metadata
except Exception:
    filter_complex_metadata = None  # type: ignore
