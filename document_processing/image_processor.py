# -*- coding: utf-8 -*-
"""图片处理：从 Word/PDF/Markdown 文档中提取图片并用占位符替换"""

import os
import re
import json
import shutil
import zipfile
from typing import Dict, List

import fitz  # PyMuPDF
from docx import Document as DocxDocument
from docx.oxml.ns import qn

from utils.file_utils import kb_images_root, kb_image_meta_dir, kb_processed_docx_dir


# ==================== Word 图片抽取与替换 ====================

def process_single_docx_to_placeholders(docx_path: str, kb_name: str) -> str:
    """
    处理单个 Word 文档：提取图片并替换为占位符，生成处理后的文档
    :param docx_path: 原始 Word 文档路径
    :param kb_name: 知识库名称
    :return: 处理后 Word 文档的路径
    """
    doc_basename = os.path.splitext(os.path.basename(docx_path))[0]

    image_output_folder = os.path.join(kb_images_root(kb_name), doc_basename)
    metadata_output_folder = kb_image_meta_dir(kb_name)
    processed_docs_folder = kb_processed_docx_dir(kb_name)

    os.makedirs(image_output_folder, exist_ok=True)
    os.makedirs(metadata_output_folder, exist_ok=True)
    os.makedirs(processed_docs_folder, exist_ok=True)

    new_docx_path = os.path.join(processed_docs_folder, f"{doc_basename}_with_placeholders.docx")
    json_output = os.path.join(metadata_output_folder, f"{doc_basename}_image_map.json")

    doc = DocxDocument(docx_path)
    image_map: Dict[str, str] = {}
    sequence: List[str] = []
    image_index = 1

    # 提取段落中的图片
    for para in doc.paragraphs:
        for run in para.runs:
            if 'graphic' in run._element.xml:
                inline_shapes = run._element.xpath('.//pic:pic')
                for shape in inline_shapes:
                    blip_list = shape.xpath('.//a:blip')
                    if not blip_list:
                        continue
                    blip = blip_list[0]
                    rId = blip.get(qn('r:embed'))
                    image_part = doc.part.related_parts.get(rId)
                    if not image_part:
                        continue

                    image_bytes = image_part.blob
                    content_type = image_part.content_type
                    ext = content_type.split("/")[-1] if "/" in content_type else "bin"
                    image_filename = f"img_{image_index}.{ext}"
                    image_save_path = os.path.join(image_output_folder, image_filename)
                    with open(image_save_path, "wb") as f:
                        f.write(image_bytes)

                    image_map[f"img_{image_index}"] = image_save_path
                    sequence.append(f"img_{image_index}")

                    run.text = f"<img>img_{image_index}</img>"
                    image_index += 1

    # 提取 EMF/WMF 格式图片（可能不被 paragraph.runs 捕获）
    with zipfile.ZipFile(docx_path, 'r') as docx_zip:
        for file in docx_zip.namelist():
            if file.startswith("word/media/") and (file.endswith(".emf") or file.endswith(".wmf")):
                ext = os.path.splitext(file)[1][1:]
                image_filename = f"img_{image_index}.{ext}"
                image_save_path = os.path.join(image_output_folder, image_filename)
                with docx_zip.open(file) as src, open(image_save_path, "wb") as dst:
                    dst.write(src.read())
                image_map[f"img_{image_index}"] = image_save_path
                sequence.append(f"img_{image_index}")
                image_index += 1

    doc.save(new_docx_path)

    orig_base = os.path.splitext(os.path.basename(docx_path))[0]
    proc_base = os.path.splitext(os.path.basename(new_docx_path))[0]
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump({
            "document_original": orig_base,
            "document_processed": proc_base,
            "mapping": image_map,
            "sequence": sequence,
        }, f, ensure_ascii=False, indent=4)

    return new_docx_path


# ==================== PDF 图片抽取与替换 ====================

def process_pdf_with_images_for_kb(pdf_path: str, kb_name: str) -> str:
    """
    处理 PDF 文件：提取图片并用占位符替代（<img>img_N</img>），
    输出处理后的 PDF，并保存图片映射信息
    :param pdf_path: 输入 PDF 路径
    :param kb_name: 知识库名称
    :return: 处理后 PDF 文件路径
    """
    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
    image_output_folder = os.path.join(kb_images_root(kb_name), pdf_basename)
    metadata_output_folder = kb_image_meta_dir(kb_name)
    processed_docs_folder = kb_processed_docx_dir(kb_name)

    os.makedirs(image_output_folder, exist_ok=True)
    os.makedirs(metadata_output_folder, exist_ok=True)
    os.makedirs(processed_docs_folder, exist_ok=True)

    output_pdf = os.path.join(processed_docs_folder, f"{pdf_basename}_with_placeholders.pdf")
    json_output = os.path.join(metadata_output_folder, f"{pdf_basename}_image_map.json")

    doc = fitz.open(pdf_path)
    image_map: Dict[str, str] = {}
    sequence: List[str] = []
    img_count = 0

    for page in doc:
        page_images = page.get_images(full=True)
        for img in page_images:
            xref = img[0]
            name = img[7]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            img_ext = base_image["ext"] or "png"
            img_count += 1
            img_name = f"img_{img_count}.{img_ext}"

            img_path = os.path.join(image_output_folder, img_name)
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            image_map[f"img_{img_count}"] = img_path
            sequence.append(f"img_{img_count}")

            rect = page.get_image_bbox(name)
            if rect:
                label_point = fitz.Point(rect.x0, rect.y1 + 5)
                page.insert_text(
                    label_point,
                    f"<img>img_{img_count}</img>",
                    fontsize=8,
                    fontname="helv",
                    fill=(0, 0, 0),
                )

    for page in doc:
        page.clean_contents(sanitize=True)

    doc.save(output_pdf)
    doc.close()

    with open(json_output, "w", encoding="utf-8") as f:
        json.dump({
            "document_original": pdf_basename,
            "document_processed": os.path.splitext(os.path.basename(output_pdf))[0],
            "mapping": image_map,
            "sequence": sequence,
        }, f, ensure_ascii=False, indent=4)

    print(f"✅ PDF处理完成：共提取 {img_count} 张图片 → {output_pdf}")
    return output_pdf


# ==================== Markdown 图片抽取与替换 ====================

# Markdown 图片语法: ![alt](path)
_MD_IMG_PATTERN = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')


def process_markdown_with_images_for_kb(md_path: str, kb_name: str) -> str:
    """
    处理 Markdown 文件：提取引用的本地图片并用占位符替换（<img>img_N</img>）
    :param md_path: 原始 Markdown 文件路径
    :param kb_name: 知识库名称
    :return: 处理后 Markdown 文件的路径
    """
    md_basename = os.path.splitext(os.path.basename(md_path))[0]
    md_dir = os.path.dirname(os.path.abspath(md_path))

    image_output_folder = os.path.join(kb_images_root(kb_name), md_basename)
    metadata_output_folder = kb_image_meta_dir(kb_name)
    processed_docs_folder = kb_processed_docx_dir(kb_name)

    os.makedirs(image_output_folder, exist_ok=True)
    os.makedirs(metadata_output_folder, exist_ok=True)
    os.makedirs(processed_docs_folder, exist_ok=True)

    output_md = os.path.join(processed_docs_folder, f"{md_basename}_with_placeholders.md")
    json_output = os.path.join(metadata_output_folder, f"{md_basename}_image_map.json")

    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    image_map: Dict[str, str] = {}
    sequence: List[str] = []
    img_count = 0

    def _replace_img(match: re.Match) -> str:
        nonlocal img_count
        img_path = match.group(2).strip()

        img_count += 1
        img_name = f"img_{img_count}"

        # 跳过网络图片和 data URI
        if img_path.startswith(("http://", "https://", "data:")):
            return f"<img>{img_name}</img>"

        # 尝试在多个位置定位本地图片文件
        candidates = [
            os.path.join(md_dir, img_path),              # 相对于 md 文件目录
            os.path.join(md_dir, "images", os.path.basename(img_path)),
            os.path.join(md_dir, "img", os.path.basename(img_path)),
            os.path.join(md_dir, "assets", os.path.basename(img_path)),
            os.path.join(md_dir, "media", os.path.basename(img_path)),
            os.path.join(md_dir, "imgs", os.path.basename(img_path)),
        ]

        for candidate in candidates:
            if os.path.isfile(candidate):
                ext = os.path.splitext(candidate)[1][1:] or "png"
                dest_path = os.path.join(image_output_folder, f"{img_name}.{ext}")
                try:
                    shutil.copy2(candidate, dest_path)
                    image_map[img_name] = dest_path
                    sequence.append(img_name)
                    return f"<img>{img_name}</img>"
                except OSError:
                    pass

        # 本地图片未找到 — 保留占位符但不存储映射
        return f"<img>{img_name}</img>"

    processed_content = _MD_IMG_PATTERN.sub(_replace_img, content)

    with open(output_md, "w", encoding="utf-8") as f:
        f.write(processed_content)

    with open(json_output, "w", encoding="utf-8") as f:
        json.dump({
            "document_original": md_basename,
            "document_processed": os.path.splitext(os.path.basename(output_md))[0],
            "mapping": image_map,
            "sequence": sequence,
        }, f, ensure_ascii=False, indent=4)

    print(f"✅ Markdown处理完成：共提取 {len(image_map)}/{img_count} 张本地图片 → {output_md}")
    return output_md
