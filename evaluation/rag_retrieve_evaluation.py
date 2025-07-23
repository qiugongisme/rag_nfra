import csv
import os
import re

import numpy as np
import pandas as pd
from langchain.embeddings import CacheBackedEmbeddings
from pymilvus import MilvusClient

from config import config
from src.utils import get_cached_embedder, MilvusUtils


def load_collection(collect_name: str) -> None:
    # 获取 Milvus 集合
    milvus_client = MilvusUtils()
    collection = milvus_client.get_collection(collect_name)
    if not milvus_client.is_collection_loaded(collection_name=collect_name):
        collection.load()


def read_xlsx_and_process(file_path, sheets: list[int]) -> list:
    """
    读取 .xlsx 文件，提取第一列作为 key，
    其余列组成字典列表返回。

    :param file_path: Excel 文件路径
    :param sheets: 表单名称或索引集合，支持读取多个表单
    :return: 列表，分别对应sheet1,2,3...，格式为 [{key: [{col2: val, col3: val, ...}, ...]}]
    """
    result = []
    for sheet_name in sheets:
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        dict_var = {}
        # 获取第一列作为 key 列
        key_col = df.columns[0]
        other_cols = df.columns[1:]

        for _, row in df.iterrows():
            key = row[key_col]
            values = row[other_cols].to_dict()

            if key in dict_var:
                dict_var[key].append(values)
            else:
                dict_var[key] = [values]
        result.append(dict_var)

    return result


def retrieve_query(question: str, embeddings: CacheBackedEmbeddings, client: MilvusClient, collect_name: str) -> list:
    search_params = {"metric_type": config.METRIC_TYPE, "params": {"nprobe": config.NPROBE}}

    all_results = []
    query_vector = embeddings.embed_query(question)
    query_vector = np.array([query_vector]).astype(np.float32).tolist()

    retrieve_results = client.search(
        collection_name=collect_name,
        data=query_vector,
        output_fields=["text", "metadata"],
        search_params=search_params,
        limit=config.TOP_K
    )
    for hit in retrieve_results[0]:
        all_results.append({
            "query": question,
            "参考答案": "",  # 这里可以根据需要填充答案 dict.update(dict2)，把字典dict2的键/值对更新到dict里
            "参考来源": "",  # 这里可以根据需要填充来源
            "text": re.sub(r'[\r\n]+', '；', hit['entity']['text']),
            "distance": "%.4f" % hit['distance'] if hit['distance'] else "0.0000",
            "metadata": hit['entity']['metadata']['source'] if 'source' in hit['entity']['metadata'] else ""
        })

    return all_results


def write_results_to_csv(retrieve_result_list, csv_f):
    # 删除旧的 CSV 文件
    if os.path.exists(csv_f):
        os.remove(csv_f)
    write_header = not os.path.exists(csv_f)
    with open(csv_f, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["查询问题", "参考答案", "参考来源", "文本块", "相似度", "元数据"])
        for item in retrieve_result_list:
            writer.writerow(
                [item["query"], item["参考答案"], item["参考来源"], item["text"], item["distance"], item["metadata"]])


def execute_retrieval():
    print("获取 Excel 表格中表单1,2,3 的问题列表...")
    input_data_path = "../evaluation/data/evaluationData.xlsx"
    sheet_list = [0, 1, 2]  # 需要读取的表单

    # 读取 Excel 文件并处理数据
    dt_list = read_xlsx_and_process(input_data_path, sheet_list)

    # 输出的名称分别与上述读取的表单对应
    output_csv_path_list = [
        "./retrieve_out/retrieve_output_WX.csv",
        "./retrieve_out/retrieve_output_TY.csv",
        "./retrieve_out/retrieve_output_TXYB.csv",
    ]

    collection_name = config.COLLECTION_NAME
    load_collection(collection_name)
    # 连接到 Milvus 服务
    client = MilvusClient(host=config.MILVUS_HOST, port=config.MILVUS_PORT)
    # 初始化嵌入模型
    embedding = get_cached_embedder()

    for i, dt in enumerate(dt_list):
        results = []
        print("开始检索问题列表...")
        for key, values in dt.items():
            # print(f"正在检索问题：{key}，请稍候...")
            res = retrieve_query(question=key, embeddings=embedding, client=client, collect_name=collection_name)
            for ddt in res:
                ddt['参考答案'] = values[0]['答案'] if values else ""
                ddt['参考来源'] = values[0]['来源'] if values else ""
            results.append(res)
            print(f"问题 '{key}' 的检索完成")

        print(f"表单 {i + 1} 的检索完成，写入结果到 CSV 文件...")
        # 展开嵌套列表
        flat_results = [item for sublist in results for item in sublist]
        write_results_to_csv(flat_results, output_csv_path_list[i])

        print(f"表单 {i + 1} 的结果已写入到 {output_csv_path_list[i]}")

    # 释放集合（从内存释放）
    client.release_collection(collection_name)


"""
    这个文件是用于评估 RAG 检索效果的。
"""
if __name__ == '__main__':
    execute_retrieval()
