import asyncio
import sys

import numpy as np
from dotenv import load_dotenv
from pymilvus import FieldSchema, DataType, CollectionSchema

from config import config
from src.callback import OutCallbackHandler
from src.chain import get_check_chain, get_qa_chain
from src.loader import load_pdf2document
from src.splitter import FileSplitter
from src.utils import get_cached_embedder, MilvusUtils, markdown_to_text

# 初始化 Milvus 数据库并加载数据
def load_data_milvus():
    # 加载目录文件
    print("Loading data...")
    documents = load_pdf2document(config.FILE_PATH)

    # 分块
    print("Chucking data...")
    texts = FileSplitter(chunk_size=config.FILE_CHUNK_SIZE, chunk_overlap=config.FILE_CHUNK_OVERLAP).split_documents(
        documents)

    # 嵌入
    print("Embeddings...")
    embeddings = get_cached_embedder()
    raw_text = [text.page_content for text in texts]
    vectors = embeddings.embed_documents(raw_text)

    # 存储
    print("Vectors and Store Milvus...")
    # 将向量转换为 float32 类型，以节省内存和提升 Milvus 的性能
    vectors = np.array(vectors, dtype=np.float32)
    print("vectors num:", len(vectors))
    # 定义字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name=config.EMBEDDING_FIELD_NAME, dtype=DataType.FLOAT_VECTOR, dim=vectors.shape[1]),
        # vectors.shape[1] 动态获取向量的维度
        FieldSchema(name="metadata", dtype=DataType.JSON)  # metadata 字段
    ]
    # 创建集合模式
    schema = CollectionSchema(fields=fields, description="RAG_NFRA Collection with BGE embeddings")

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

    print(f"成功将 {len(vectors)} 条向量存储到 Milvus 集合 {config.COLLECTION_NAME} 且集合已启动内存加载。")

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
    if args.shell:
        asyncio.run(question_answering())
