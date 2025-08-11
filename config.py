class Config:
    # 文档目录路径
    FILE_PATH = "./data"
    # 读取 PDF 提取文本的输出路径
    FILE_OUTPUT_PATH = "./data/output"

    # 分块
    FILE_CHUNK_SIZE = 500
    # FILE_CHUNK_SIZE = 300
    FILE_CHUNK_OVERLAP = 100  # 通常设为 chunk_size 的10-25%
    # FILE_CHUNK_OVERLAP = 75  # 通常设为 chunk_size 的10-25%

    # 嵌入模型
    EMBEDDINGS_CACHE_PATH = "./.cache/embeddings"
    EMBED_MODEL_NAME = "BAAI/bge-base-zh-v1.5"
    # EMBED_MODEL_NAME = "BAAI/bge-large-zh-v1.5"
    EMBED_MODEL_KWARGS_PROCESSOR = "cpu"
    IS_NORMALIZE_EMBEDDINGS = True  # 是否使用归一化，BGE 推荐归一化

    EMBEDDING_FIELD_NAME = "embedding"  # 向量字段名称

    # 向量存储
    MILVUS_HOST = "localhost"
    MILVUS_PORT = 19530

    # COLLECTION_NAME = "RAG_PART_NFRA"
    # COLLECTION_NAME = "RAG_PART_NFRA_LARGE"
    COLLECTION_NAME = "RAG_PART_NFRA_HYBRID"

    """
        分块大小为 300，重叠大小为 75
        
        依据评估数据集：evaluation/data/retrieveInputData_V1_2.xlsx，评估结果来看召回率比分块大小：500，重叠大小：100 差点
    """
    # COLLECTION_NAME = "RAG_PART_NFRA_375"

    # 索引配置
    INDEX_TYPE = "IVF_FLAT"  # 使用 IVF_FLAT（倒排文件索引 + 精确搜索）
    METRIC_TYPE = "IP"  # 使用内积作为相似度计算方式

    """
    nlist 是聚类数目，参考如下：
        向量数据规模（个）    推荐 nlist 值
            < 10 万           100 ~ 1000
        10 万 ~ 100 万        500 ~ 5000
            > 100 万         1000 ~ 10000
    """
    NLIST = 100

    # 检索参数
    NPROBE = 10  # nprobe 是查询时的聚类数目，通常设置为 nlist 的 1/10 到 1/5
    # TOP_K = 2 # 检索时返回的最相似向量数量
    TOP_K = 3  # 检索时返回的最相似向量数量

    """ ================================== hybrid param start ================================== """
    BGEM3_USE_FP16 = False
    BGEM3_DEVICE = "cpu"

    DENSE_VECTOR_FIELD = "dense_vector"
    SPARSE_VECTOR_FIELD = "sparse_vector"

    CONSISTENCY_LEVEL = "Strong"

    SPARSE_INDEX_TYPE = "SPARSE_INVERTED_INDEX"
    SPARSE_METRIC_TYPE = "IP"

    """ rerank_method 可选：'weighted' 或 'rrf'，默认 rrf """
    RERANK_METHOD_RRF = "rrf"
    RRF_K = 60  # RRF 参数

    RERANK_METHOD_WEIGHTED = "weighted"
    DENSE_WEIGHTED = 0.7
    SPARSE_WEIGHTED = 1.0

    """ ================================== hybrid param end ================================== """

    """ MilvusRetriever 支持的搜索类型 "dense" : 单密集检索, "hybrid"：混合检索（稀疏和密集，默认） """
    MILVUS_RETRIEVER_SEARCH_TYPE = "hybrid"

config = Config()
