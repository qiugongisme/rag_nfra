import re

import markdown
from bs4 import BeautifulSoup
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_core.callbacks import Callbacks
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import connections, Collection, CollectionSchema, utility

from config import config


def get_cached_embedder() -> CacheBackedEmbeddings:
    """获取缓存的嵌入模型实例"""

    # 创建本地文件存储实例
    fs = LocalFileStore(config.EMBEDDINGS_CACHE_PATH)
    # 创建 HuggingFace 嵌入模型实例
    underlying_embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL_NAME,  # 嵌入模型名称
        model_kwargs={'device': config.EMBED_MODEL_KWARGS_PROCESSOR},  # 嵌入模型参数
        encode_kwargs={'normalize_embeddings': config.IS_NORMALIZE_EMBEDDINGS},  # 是否归一化嵌入，BGE推荐使用
    )
    # 创建缓存嵌入模型实例
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings, fs, namespace=underlying_embeddings.model_name
    )
    return cached_embedder


# Milvus 数据库工具类
class MilvusUtils:
    def __init__(self, host=config.MILVUS_HOST, port=config.MILVUS_PORT):
        """
        初始化 MilvusUtils 实例
        :param host:
        :param port:
        """
        # 连接到 Milvus 服务
        self.collection = connections.connect(host=host, port=port)

    def create_collection(self, collection_name: str, schema: CollectionSchema) -> Collection:
        """
        创建集合
        :param collection_name: 集合名称
        :param schema: 集合的 schema
        :return: Collection 对象
        """
        if utility.has_collection(collection_name):
            raise ValueError(f"Collection {collection_name} already exists.")
            # utility.drop_collection(collection_name)  # 测试阶段，暂时是删除 todo
        collection = Collection(name=collection_name, schema=schema)
        return collection

    def get_collection(self, collection_name: str) -> Collection:
        """
        获取已存在的集合
        :param collection_name: 集合名称
        :return: Collection 对象
        """
        if not utility.has_collection(collection_name):
            raise ValueError(f"Collection {collection_name} does not exist.")
        return Collection(name=collection_name)

    def is_collection_loaded(self, collection_name: str) -> bool:
        """ 检查集合是否已加载
        :param collection_name: 集合名称
        :return: bool，表示集合是否已加载，True 表示已加载，False 表示未加载或发生错误
        """
        try:
            return utility.load_state(collection_name=collection_name) == "Loaded"
        except Exception as e:
            print(f"Error checking collection load state: {e}")
            return False


def get_deepseek_model(
        model: str = "deepseek-chat",
        streaming: bool = True,
        callbacks: Callbacks = None) -> ChatDeepSeek:
    """获取 DeepSeek 模型实例
    - model: 模型名称，默认为 "deepseek-chat"
    - streaming: 是否启用流式输出，默认为 True
    - callbacks: 回调函数列表，默认为 None
    """
    model = ChatDeepSeek(model=model, streaming=streaming, callbacks=callbacks)

    return model


def markdown_to_text(md_text):
    """ 将 Markdown 文本转换为纯文本，并去除列表前缀和空行"""

    # 1. Markdown 转 HTML
    html = markdown.markdown(md_text)

    # 2. 使用 BeautifulSoup 提取纯文本
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text()
    # 3. 后处理：去除列表前缀（-、*、数字等）
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        # 去除列表前缀：如 "- 内容"、"* **内容"
        line = re.sub(r'^[\-\*\**\.]+\s*', '', line.strip())
        cleaned_lines.append(line)

    # 4. 去除空行和合并段落（可选）
    cleaned_text = '\n'.join([line for line in cleaned_lines if line.strip()])

    return cleaned_text
