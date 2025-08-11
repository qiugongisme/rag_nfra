import asyncio
import logging
import sys

import numpy as np
from dotenv import load_dotenv
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import FieldSchema, DataType, CollectionSchema

from config import config
from src.callback import OutCallbackHandler
from src.chain import get_check_chain, get_qa_chain
from src.loader import load_pdf2document
from src.splitter import FileSplitter
from src.utils import get_cached_embedder, MilvusUtils


# 初始化 Milvus 数据库并加载数据
def load_data_milvus():
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # 加载目录文件
    logger.info("Loading data...")
    documents = load_pdf2document(config.FILE_PATH)

    # 分块
    logger.info("Chucking data...")
    texts = FileSplitter(chunk_size=config.FILE_CHUNK_SIZE, chunk_overlap=config.FILE_CHUNK_OVERLAP).split_documents(
        documents)

    # 嵌入
    logger.info("Embeddings...")
    embeddings = get_cached_embedder()
    raw_text = [text.page_content for text in texts]
    vectors = embeddings.embed_documents(raw_text)

    # 存储
    logger.info("Vectors and Store Milvus...")
    # 嵌入向量归一化处理，将向量转换为 float32 类型，以节省内存和提升 Milvus 的性能
    vectors = np.array(vectors, dtype=np.float32)
    logger.info("vectors num:", len(vectors))
    # 定义字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name=config.EMBEDDING_FIELD_NAME, dtype=DataType.FLOAT_VECTOR, dim=vectors.shape[1]),
        # vectors.shape[1] 动态获取向量的维度
        FieldSchema(name="metadata", dtype=DataType.JSON)  # metadata 字段
    ]
    # 创建集合模式
    schema = CollectionSchema(fields=fields, description="RAG_NFRA Collection with BAAI/bge-large-zh-v1.5 embeddings")

    milvus_util = MilvusUtils()
    # 创建集合
    collection = milvus_util.create_collection(collection_name=config.COLLECTION_NAME, schema=schema)

    # 插入数据
    data = [
        raw_text,  # 文本内容
        vectors.tolist(),  # 向量
        [text.metadata for text in texts]  # 元数据
    ]
    collection.insert(data)

    # 构建索引
    index_params = {
        "index_type": config.INDEX_TYPE,  # 索引类型
        "metric_type": config.METRIC_TYPE,  # 距离度量类型
        "params": {"nlist": config.NLIST}  # nlist 参数，适用于 IVF_FLAT 和 IVF_PQ 索引类型
    }
    collection.create_index(config.EMBEDDING_FIELD_NAME, index_params)

    # 加载集合到内存（异步加载）
    collection.load()

    logger.info(f"成功将 {len(vectors)} 条向量存储到 Milvus 集合 {config.COLLECTION_NAME} 且集合已启动内存加载。")


def load_data_milvus_hybrid():
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # 加载目录文件
    logger.info("Loading data...")
    documents = load_pdf2document(config.FILE_PATH)

    # 分块
    logger.info("Chucking data...")
    docs = FileSplitter(chunk_size=config.FILE_CHUNK_SIZE, chunk_overlap=config.FILE_CHUNK_OVERLAP).split_documents(
        documents)

    # 嵌入
    logger.info("Embeddings...")
    bgem3_ef = BGEM3EmbeddingFunction(use_fp16=config.BGEM3_USE_FP16, device=config.BGEM3_DEVICE)

    texts = [text.page_content for text in docs]
    texts_embeddings = bgem3_ef(texts)
    logger.info(f"向量生成完成，密集向量维度：{bgem3_ef.dim['dense']}")

    # 存储
    logger.info("Vectors and Store Milvus...")

    dense_vector_field = config.DENSE_VECTOR_FIELD
    sparse_vector_field = config.SPARSE_VECTOR_FIELD

    # 定义字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="metadata", dtype=DataType.JSON),
        FieldSchema(name=dense_vector_field, dtype=DataType.FLOAT_VECTOR, dim=bgem3_ef.dim["dense"]),
        FieldSchema(name=sparse_vector_field, dtype=DataType.SPARSE_FLOAT_VECTOR)
    ]

    # 创建集合模式
    schema = CollectionSchema(fields=fields, description="RAG_NFRA Collection hybrid search support")

    # 创建集合
    milvus_util = MilvusUtils()
    collection = milvus_util.create_collection(collection_name=config.COLLECTION_NAME,
                                               schema=schema,
                                               consistency_level=config.CONSISTENCY_LEVEL)

    # 构建索引
    # 密集向量索引
    dense_index_params = {
        "index_type": config.INDEX_TYPE,
        "metric_type": config.METRIC_TYPE,
        "params": {"nlist": config.NLIST}
    }
    collection.create_index(dense_vector_field, dense_index_params)
    # 稀疏向量索引
    sparse_index_params = {
        "index_type": config.SPARSE_INDEX_TYPE,
        "metric_type": config.SPARSE_METRIC_TYPE
    }
    collection.create_index(sparse_vector_field, sparse_index_params)

    # 插入数据
    data = [
        texts,  # text 字段
        [text.metadata for text in docs],  # metadata 字段
        texts_embeddings["dense"],  # dense_vector 字段
        texts_embeddings["sparse"]  # sparse_vector 字段，用于稀疏检索
    ]
    collection.insert(data)

    # 加载集合到内存
    collection.load()

    logger.info(
        f"成功将 {len(texts_embeddings["dense"])} 条向量存储到 Milvus 集合 {config.COLLECTION_NAME} 并支持混合检索。")


# 运行问答
async def question_answering():
    # 加载环境变量（建议将API密钥放在.env文件中）
    load_dotenv()

    check_chain = get_check_chain()

    out_callback = OutCallbackHandler()
    chain = get_qa_chain(out_callback=out_callback)

    while True:
        question = input("\n用户:")
        if question.strip() == "stop":
            break
        print("\n国家金融监督管理总局政策法规的小助手:", end="")
        is_nfra = check_chain.invoke({"question": question})
        if not is_nfra:
            print("不好意思，我是国家金融监督管理总局政策法规AI助手，请提问相关的问题。")
            continue

        task = asyncio.create_task(chain.ainvoke({"question": question}))

        async for new_token in out_callback.aiter():
            print(new_token, end="", flush=True)

        res = await task
        print("用于回答的上下内容是：\n", res["retrieve_context"])
        # print("\n", markdown_to_text(res["answer"]))

        out_callback.done.clear()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Please perform only one operation at a time.")
    parser.add_argument(
        "-i",
        "--init",
        action="store_true",
        help=('''
                init vectorstore Milvus
            ''')
    )
    parser.add_argument(
        "-b",
        "--init-ds",
        action="store_true",
        help=('''
                init vectorstore Milvus with hybrid support
            ''')
    )
    parser.add_argument(
        "-s",
        "--shell",
        action="store_true",
        help=('''
            run question answering in shell mode
        ''')
    )

    if len(sys.argv) <= 1:
        parser.print_help()
        exit()

    args = parser.parse_args()
    if args.init:
        load_data_milvus()
    if args.init_ds:
        load_data_milvus_hybrid()
    if args.shell:
        asyncio.run(question_answering())
