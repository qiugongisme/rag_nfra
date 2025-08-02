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

        # 去除无用的字符
        text = rm_useless_content(text)

        # 把文本保存为 txt 文件，便于优化
        output_path = os.path.join(config.FILE_OUTPUT_PATH, os.path.basename(pdf_file).replace('.pdf', '.txt'))
        save_text_to_file(text, output_path)

        texts.append(text)
        metadatas.append({"source": "《" + os.path.basename(pdf_file).replace('.pdf', '') + "》"})

    # 创建 document
    documents = RecursiveCharacterTextSplitter().create_documents(texts, metadatas=metadatas)

    return documents


def rm_useless_content(text):
    # 去除页眉页脚等不必要的内容
    fixed_strings_to_remove = [
        "国家金融监督管理总局规章",
        "国家金融监督管理总局发布",
        "中国银行保险监督管理委员会规章",
        "中国银行保险监督管理委员会发布"
    ]
    # 去除固定字符串
    for item in fixed_strings_to_remove:
        text = text.replace(item, "")
    # 使用正则表达式去除所有 "- 数字 -" 格式的页码（如 "- 1 -", "- 2 -", "- 10 -" 等）
    text = re.sub(r"-\s*\d+\s*-", "", text)

    # 正则表达式模式，用于匹配网址 http 或 https
    text = re.sub(r'(https?://[^\s]+)', '', text)

    # 使用正则表达式匹配并删除形如 docId=xxx&itemId=yyy 的字符串
    text = re.sub(r'docId=\d+(&itemId=\d+)?', '', text)

    # 去除可能产生的多余空行
    text = re.sub(r'\n\s*\n', '\n', text).strip()

    return text


def split_by_pattern(content: str, pattern: str = r"第\S*条") -> List[str]:
    """根据正则表达式切分内容
    :param content: 文本内容
    :param pattern: 正则表达式，默认是：r"第\S*条"
    """
    # 匹配所有以“第X条”开头的位置
    matches = list(re.finditer(rf"^{pattern}", content, re.MULTILINE))
    if not matches:
        return [content.strip()]

    result = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        part = content[start:end].strip()
        if part:
            result.append(part)
    return result


class CustomDocument:
    def __init__(self, content, metadata):
        self.content = content
        self.metadata = metadata


def load_and_split(directory: str) -> List[CustomDocument]:
    """从指定文件目录加载 PDF 文件并提取、切分文本内容
    :param directory: 文件目录
    :return: 返回包含提取、切分后的文本、元数据的 CustomDocument 列表
    """
    result = []

    # 从目录读取 pdf 文件
    pdf_file_list = get_pdf_files(directory)
    # 提取文本
    for pdf_file in pdf_file_list:
        document = fitz.open(pdf_file)
        text_content = ""
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text_content += page.get_text()

        # 去除无用的字符
        text_content = rm_useless_content(text_content)

        # 把文本保存为 txt 文件，便于优化
        output_path = os.path.join(config.FILE_OUTPUT_PATH, os.path.basename(pdf_file).replace('.pdf', '.txt'))
        save_text_to_file(text_content, output_path)

        # 切分文本内容
        split_list = split_by_pattern(text_content)
        # 元数据
        metadata = {"source": "《" + os.path.basename(pdf_file).replace('.pdf', '') + "》"}
        for split_content in split_list:
            result.append(CustomDocument(split_content, metadata))

    return result
