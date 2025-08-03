import itertools
import logging
from typing import List, Optional

import numpy as np
from langchain.embeddings import CacheBackedEmbeddings
from langchain.retrievers import RePhraseQueryRetriever, MultiQueryRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.retrievers import BaseRetriever
from pydantic.v1 import BaseModel
from pymilvus import Hits, Collection

from config import config
from src.prompt import RE_QUERY_PROMPT_TEMPLATE, MULTI_QUERY_PROMPT_TEMPLATE
from src.utils import get_cached_embedder, MilvusUtils, unique_objects_by_id

logger = logging.getLogger(__name__)


class MilvusRetriever(BaseRetriever):
    # 声明字段并添加类型注解以避免 Pydantic 报错
    milvus_client: Optional[MilvusUtils] = None
    embedder: Optional[CacheBackedEmbeddings] = None
    collection: Optional[Collection] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.milvus_client = MilvusUtils()
        self.embedder = get_cached_embedder("." + config.EMBEDDINGS_CACHE_PATH)
        self.collection = self.milvus_client.get_collection(config.COLLECTION_NAME)
        if not self.milvus_client.is_collection_loaded(collection_name=config.COLLECTION_NAME):
            self.collection.load()

    def _search_single_query(self, qry: str) -> List[Hits]:
        try:
            logger.info("embedding and query vector execute. query is -> %s", qry)
            query_vector = self.embedder.embed_query(qry)
            query_vector = np.array([query_vector], dtype=np.float32).tolist()

            search_params = {
                "metric_type": config.METRIC_TYPE,
                "params": {"nprobe": config.NPROBE}
            }

            results = self.collection.search(
                data=query_vector,
                anns_field=config.EMBEDDING_FIELD_NAME,
                param=search_params,
                limit=config.TOP_K,
                output_fields=["id", "text", "metadata"],
            )
            return results[0]
        except Exception as e:
            logger.error("Error occurred during search for query: %s, error: %s", qry, str(e))
            return []

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Hits]:
        """从 Milvus 中检索与查询相关的文档。
        :param query: 查询字符串
        :param run_manager: 用于检索的回调管理器
        :return: 检索到的 Hits 集合
        """
        query_list = [q for q in query.split("\n") if q.strip()]
        result_list = [self._search_single_query(qry) for qry in query_list]

        flat_list = list(itertools.chain.from_iterable(result_list))

        # 根据 hits 的 id 过滤重复的
        return unique_objects_by_id(flat_list)


def retrieved_deal(hits: list[Hits]) -> str:
    """
    处理检索到的 Hits 集合，更好地用作 LLM 的上下文输入
    :param hits: Collection.search() 返回的 Hits 集合
    :return: 处理后的 str
    """
    retrieved_context = ""
    for hit in hits:
        # 这里为何不使用 hit.entity.get('text', '') ? 是因为 pymilvus 的 Hits 对象的 entity 属性不是一个字典
        text = hit.entity.get('text') if hit.entity.get('text') is not None else ''
        retrieved_context = retrieved_context + text + "\n"

        metadata = hit.entity.get('metadata')['source']
        if metadata:
            retrieved_context = retrieved_context + "\n" + "—— 来源于国家金融监督管理总局发布文件 " + metadata + "\n"
        else:
            retrieved_context = retrieved_context + "\n" + "—— 未知来源 " + "\n"

    retrieved_context = retrieved_context.strip()
    # 测试阶段 todo
    # print(f"\n 依据相似度排名在：{config.TOP_K}，检索到的知识，按块分行组合结果如下: \n", retrieved_context)
    # print("\n")

    return retrieved_context


def retrieved_deal_eval(hits: list[Hits]) -> tuple:
    """
    处理检索到的 Hits 集合，更好地用作 LLM 的上下文输入
    :param hits: Collection.search() 返回的 Hits 集合
    :return: 处理后的 str
    """
    retrieved_context = ""
    similarity_str = ""
    for i, hit in enumerate(hits):
        text = hit.entity.get('text') if hit.entity.get('text') is not None else ''
        retrieved_context = retrieved_context + text + "\n"

        metadata = hit.entity.get('metadata')['source']
        if metadata:
            retrieved_context = retrieved_context + "\n" + metadata + "\n"
        else:
            retrieved_context = retrieved_context + "\n" + "—— 未知来源 " + "\n"

        distance_float = "%.4f" % hit.distance
        if distance_float:
            similarity_str += f"文本块 {i + 1}: {distance_float}\n"
        else:
            similarity_str += f"文本块 {i + 1}: Nan\n"

    retrieved_context = retrieved_context.strip()
    # 测试阶段
    # print(f"\n 依据相似度排名在：{config.TOP_K}，检索到的知识，按块分行组合结果如下: \n", retrieved_context)
    return retrieved_context, similarity_str


def query_rewrite_retriever(retriever: BaseRetriever, model: BaseModel) -> BaseRetriever:
    retriever_from_llm = RePhraseQueryRetriever.from_llm(
        retriever=retriever,
        llm=model,
        prompt=RE_QUERY_PROMPT_TEMPLATE
    )
    return retriever_from_llm


class LineListOutputParser(BaseOutputParser[List[str]]):
    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # 过滤空行


def query_multi_retiever(retriever: BaseRetriever, model: BaseModel) -> BaseRetriever:
    # 定义输出格式
    output_parser = LineListOutputParser()
    # 构建执行链
    llm_chain = MULTI_QUERY_PROMPT_TEMPLATE | model | output_parser

    retriever = MultiQueryRetriever(
        retriever=retriever, llm_chain=llm_chain, parser_key="lines"
    )

    return retriever
