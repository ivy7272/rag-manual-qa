# -*- coding: utf-8 -*-
"""文件/目录工具：路径生成、目录管理、知识库删除"""

import os
import shutil
from typing import List, Tuple

from config import BASE_KB_DIR, BASE_DATA_DIR


# ==================== 路径生成工具 ====================

def kb_dir(kb_name: str) -> str:
    """获取指定知识库的根目录路径（存储向量库和索引文件）"""
    return os.path.join(BASE_KB_DIR, kb_name)


def kb_index_json(kb_name: str) -> str:
    """获取指定知识库的索引文件（index.json）路径"""
    return os.path.join(kb_dir(kb_name), "index.json")


def kb_prompt_json(kb_name: str) -> str:
    """获取指定知识库的提示词配置文件（prompt_hints.json）路径"""
    return os.path.join(kb_dir(kb_name), "prompt_hints.json")


def kb_summary_json(kb_name: str) -> str:
    """获取指定知识库的总结文件（summary.json）路径"""
    return os.path.join(kb_dir(kb_name), "summary.json")


def kb_processed_docx_dir(kb_name: str) -> str:
    """获取指定知识库的处理后Word文档目录路径"""
    return os.path.join(BASE_DATA_DIR, kb_name, "processed_RAG_doc")


def kb_images_root(kb_name: str) -> str:
    """获取指定知识库的图片存储根目录路径"""
    return os.path.join(BASE_DATA_DIR, kb_name, "images")


def kb_image_meta_dir(kb_name: str) -> str:
    """获取指定知识库的图片元数据目录路径"""
    return os.path.join(BASE_DATA_DIR, kb_name, "image_metadata")


def kb_source_folders_json(kb_name: str) -> str:
    """获取指定知识库的源文件夹配置路径"""
    return os.path.join(kb_dir(kb_name), "source_folders.json")


# ==================== 目录管理 ====================

def ensure_dirs_for_kb(kb_name: str):
    """确保指定知识库的所有必要目录已创建"""
    os.makedirs(kb_dir(kb_name), exist_ok=True)
    os.makedirs(kb_processed_docx_dir(kb_name), exist_ok=True)
    os.makedirs(kb_images_root(kb_name), exist_ok=True)
    os.makedirs(kb_image_meta_dir(kb_name), exist_ok=True)


def list_kbs() -> List[str]:
    """列出所有已创建的知识库名称（按字母排序）"""
    if not os.path.exists(BASE_KB_DIR):
        return []
    return sorted([
        d for d in os.listdir(BASE_KB_DIR)
        if os.path.isdir(os.path.join(BASE_KB_DIR, d))
    ])


# ==================== KB 删除 ====================

def delete_dir_safe(path: str) -> Tuple[bool, str]:
    """安全删除目录（处理路径为空、目录不存在、删除异常等边界情况）"""
    if not path:
        return False, "路径为空"
    if not os.path.exists(path):
        return True, f"路径不存在，跳过：{path}"
    try:
        shutil.rmtree(path)
        return True, f"已删除：{path}"
    except Exception as e:
        return False, f"删除失败：{path} -> {e}"


def delete_kb(kb_name: str) -> Tuple[bool, List[str]]:
    """完整删除一个知识库（包括索引/向量库目录 + 处理后数据目录）"""
    if not kb_name or not kb_name.strip():
        return False, ["知识库名称为空"]

    logs: List[str] = []
    kb_root = kb_dir(kb_name)
    data_root = os.path.join(BASE_DATA_DIR, kb_name)

    ok1, msg1 = delete_dir_safe(kb_root)
    logs.append(msg1)
    ok2, msg2 = delete_dir_safe(data_root)
    logs.append(msg2)

    return (ok1 and ok2), logs
