import re
import threading
from typing import Dict, Optional

import markdown
from bs4 import BeautifulSoup
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.chat_models import ChatTongyi
from langchain_core.callbacks import Callbacks
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import connections, Collection, CollectionSchema, utility

from config import config

# 全局缓存和锁
_embedder_cache: Dict[str, CacheBackedEmbeddings] = {}
_cache_lock = threading.RLock()


def get_cached_embedder(
        local_cache_path: Optional[str] = None,
        force_reload: bool = False
) -> CacheBackedEmbeddings:
    """获取缓存的嵌入模型实例（带实例级缓存）
    Args:
        local_cache_path: 本地缓存路径，默认使用配置
        force_reload: 是否强制重新加载模型
    Returns:
        CacheBackedEmbeddings: 缓存嵌入模型实例
    """
    # 处理默认参数
    if local_cache_path is None:
        local_cache_path = config.EMBEDDINGS_CACHE_PATH

    # 缓存键
    cache_key = local_cache_path

    # 检查缓存（双重检查锁定模式）
    if not force_reload:
        with _cache_lock:
            if cache_key in _embedder_cache:
                return _embedder_cache[cache_key]

    try:
        # 创建本地文件存储实例
        fs = LocalFileStore(local_cache_path)

        # 创建 HuggingFace 嵌入模型实例
        underlying_embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBED_MODEL_NAME,
            model_kwargs={'device': config.EMBED_MODEL_KWARGS_PROCESSOR},
            encode_kwargs={'normalize_embeddings': config.IS_NORMALIZE_EMBEDDINGS},
        )

        # 创建缓存嵌入模型实例
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings,
            fs,
            namespace=underlying_embeddings.model_name
        )

        # 缓存实例
        with _cache_lock:
            _embedder_cache[cache_key] = cached_embedder

        return cached_embedder

    except Exception as e:
        raise RuntimeError(f"Failed to create cached embedder: {str(e)}") from e


def clear_embedder_cache():
    """清理嵌入模型缓存"""
    global _embedder_cache
    with _cache_lock:
        _embedder_cache.clear()


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


def get_qwen_model(
        model: str = "qwen-turbo-2025-07-15",
        api_key: str = None,
        streaming: bool = True,
        model_kwargs: dict = None,
        callbacks: Callbacks = None):
    """获取通义千问模型实例
    - model: 模型名称，默认为 "qwen-turbo-2025-07-15"
    - api_key: 调用模型的 api_key
    - streaming: 是否启用流式输出，默认为 True
    - callbacks: 回调函数列表，默认为 None
    """
    model = ChatTongyi(model=model, api_key=api_key, streaming=streaming, model_kwargs=model_kwargs, callbacks=callbacks)

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
        line = re.sub(r'^[\-\*\.]+\s*', '', line.strip())
        cleaned_lines.append(line)

    # 4. 去除空行和合并段落（可选）
    cleaned_text = '\n'.join([line for line in cleaned_lines if line.strip()])

    return cleaned_text


def unique_objects_by_id(objects):
    seen_ids = set()
    unique_objects = []

    for obj in objects:
        if obj.id not in seen_ids:
            seen_ids.add(obj.id)
            unique_objects.append(obj)

    return unique_objects
