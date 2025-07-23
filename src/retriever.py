import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from pymilvus import Hits

from config import config
from src.utils import get_cached_embedder, MilvusUtils


class MilvusRetriever(BaseRetriever):

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Hits]:
        """从 Milvus 中检索与查询相关的文档。
        :param query: 查询字符串
        :param run_manager: 用于检索的回调管理器
        :return: 检索到的 Hits 集合
        """

        # 获取 Milvus 集合
        collection_name = config.COLLECTION_NAME
        milvus_client = MilvusUtils()
        collection = milvus_client.get_collection(collection_name)
        if not milvus_client.is_collection_loaded(collection_name=collection_name):
            collection.load()

        # 生成查询嵌入向量
        print("\n query:", query)
        embeddings = get_cached_embedder()
        query_vector = embeddings.embed_query(query)
        query_vector = np.array([query_vector]).astype(np.float32).tolist()

        # 设置检索参数
        search_params = {"metric_type": config.METRIC_TYPE, "params": {"nprobe": config.NPROBE}}
        # 执行检索
        results = collection.search(
            data=query_vector,
            anns_field=config.EMBEDDING_FIELD_NAME,  # 嵌入向量字段名
            param=search_params,
            limit=config.TOP_K,
            output_fields=["id", "text", "metadata"],  # 输出字段
        )

        return results[0] if results else []


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
    # print(f"\n 依据相似度排名在：{config.TOP_K}，检索到的知识，按块分行组合结果如下: \n", retrieved_context)

    return retrieved_context, similarity_str
