import os
import re
from typing import List

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import config


def get_pdf_files(directory):
    """获取指定目录下的所有PDF文件路径
    - param directory: 目录路径
    """
    pdf_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files


def save_text_to_file(text, output_path):
    """将提取的文本保存到指定文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 确保目录存在
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(text)


def load_pdf2document(directory: str) -> List[Document]:
    """从指定目录加载 PDF 文件并提取文本内容
    :param directory: PDF 文件所在目录
    :return: 返回包含提取文本、元数据的 Document (langchain_core.documents) 列表
    """

    # 从目录读取 pdf 文件
    pdf_file_list = get_pdf_files(directory)
    texts, metadatas = [], []
    # 提取文本
    for pdf_file in pdf_file_list:
        document = fitz.open(pdf_file)
        text = ""
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
            # 去除页眉页脚等不必要的内容
            fixed_strings_to_remove = [
                "国家金融监督管理总局规章",
                "国家金融监督管理总局发布"
            ]
            # 去除固定字符串
            for item in fixed_strings_to_remove:
                text = text.replace(item, "")
            # 使用正则表达式去除所有 "- 数字 -" 格式的页码（如 "- 1 -", "- 2 -", "- 10 -" 等）
            text = re.sub(r"-\s*\d+\s*-", "", text)

            # 去除可能产生的多余空行
            text = re.sub(r'\n\s*\n', '\n', text).strip()

        # 把文本保存为 txt 文件，便于优化
        output_path = os.path.join(config.FILE_OUTPUT_PATH, os.path.basename(pdf_file).replace('.pdf', '.txt'))
        save_text_to_file(text, output_path)

        texts.append(text)
        metadatas.append({"source": "《" + os.path.basename(pdf_file).replace('.pdf', '') + "》"})

    # 创建 document
    documents = RecursiveCharacterTextSplitter().create_documents(texts, metadatas=metadatas)

    return documents
