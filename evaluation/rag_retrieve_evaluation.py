import csv
import logging
import os
import random
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pymilvus
from dotenv import load_dotenv
from langchain.embeddings import CacheBackedEmbeddings
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import MilvusClient

from config import config
from src.retriever import MilvusRetriever, query_rewrite_retriever, milvus_hybrid_retrieve
from src.utils import get_cached_embedder, MilvusUtils, get_qwen_model

logger = logging.getLogger(__name__)


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


def read_and_group_csvs(directory: str, batch_flag: str):
    # 获取目录下所有 batch_flag 标识结尾的 .csv 文件
    csv_files = [f for f in os.listdir(directory) if f.endswith(batch_flag + '.csv')]

    # 按文件名的最后 last_n 个字符分组
    last_n = 18 + len(batch_flag) + 1
    # last_n = len(batch_flag) + 4
    grouped_files = {}
    for file_name in csv_files:
        key = file_name[-last_n:]
        if key not in grouped_files:
            grouped_files[key] = []
        grouped_files[key].append(os.path.join(directory, file_name))

    return grouped_files


def read_csv_and_count(file_path) -> int:
    set_count = set()
    try:
        df = pd.read_csv(file_path)

        for row in df.itertuples():
            if row[7] or row[7] == 'TRUE':
                set_count.add(row[1])

    except Exception as e:
        logging.info(f"Error processing file {file_path}: {e}")

    return len(set_count)


def read_directory_csv_and_count(directory: str, batch_flag: str):
    # 读取并分组CSV文件
    grouped_files = read_and_group_csvs(directory=directory, batch_flag=batch_flag)

    for key in grouped_files.keys():
        key_sum = 0
        for file_path in grouped_files[key]:
            key_sum += read_csv_and_count(file_path)
        logging.info(f"Group Key: {key} , total_tag: {key_sum}")


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
            "text": re.sub(r'[\r\n]+', '', hit['entity']['text']),
            "distance": "%.4f" % hit['distance'] if hit['distance'] else "0.0000",
            "metadata": hit['entity']['metadata']['source'] if 'source' in hit['entity']['metadata'] else ""
        })

    return all_results


def extract_txt_str(text) -> str:
    """提取文本中第一个出现的"第X条"模式
    Args:
        text (str): 输入文本
    Returns:
        list: 包含第一个匹配内容的列表，如果没有匹配则返回空列表
    """
    # 正则表达式模式，用于匹配"第X条"
    pattern = r"第\S*条"

    # 输入验证
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # 使用re.search函数查找第一个匹配的内容
    try:
        match = re.search(pattern, text)
        return match.group() if match else ""
    except re.error:
        # 如果正则表达式执行出错，返回空列表
        return ""


def extract_txt_list(text) -> list:
    # 正则表达式模式，用于匹配“第X条”
    pattern = r"第(?:[一二三四五六七八九]?[千百十]?[百十]?[一二三四五六七八九]?|十)条"

    # 使用re.findall函数查找所有匹配的内容
    matches = re.findall(pattern, text)

    return matches


def write_results_to_csv(retrieve_result_list, csv_f):
    # 删除旧的 CSV 文件
    if os.path.exists(csv_f):
        os.remove(csv_f)
    write_header = not os.path.exists(csv_f)
    with open(csv_f, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            # writer.writerow(["查询问题", "参考答案", "参考来源", "文本块", "相似度", "元数据"])
            writer.writerow(["查询问题", "参考答案", "参考来源", "文本块", "相似度", "元数据", "召回说明"])
            # writer.writerow(
            #     ["查询问题", "参考答案", "参考来源", "文本块", "相似度", "元数据", "召回说明", "HyDE后新问题"])
        for item in retrieve_result_list:
            writer.writerow(
                # [item["query"], item["参考答案"], item["参考来源"], item["text"], item["distance"], item["metadata"]]
                [item["query"], item["参考答案"], item["参考来源"], item["text"], item["distance"], item["metadata"],
                 item["召回说明"]]
                # [item["query"], item["参考答案"], item["参考来源"], item["text"], item["distance"], item["metadata"],
                #  item["召回说明"], item["HyDE后新问题"]]
            )


def execute_retrieval(embedding: CacheBackedEmbeddings, client: MilvusClient, collection_name: str,
                      output_fname_ctrl: str = None):
    logging.info("获取 Excel 表格中表单1,2,3 的问题列表...")
    # input_data_path = "../evaluation/data/retrieveInputData.xlsx"
    # input_data_path = "../evaluation/data/retrieveInputData_V1_1.xlsx"
    input_data_path = "../evaluation/data/retrieveInputData_V1_2.xlsx"
    sheet_list = [0, 1, 2]  # 需要读取的表单
    # sheet_list = [1]  # 需要读取的表单

    # 读取 Excel 文件并处理数据
    dt_list = read_xlsx_and_process(input_data_path, sheet_list)

    # 输出的名称分别与上述读取的表单对应
    output_csv_path_list = [
        # "./eval_out/retrieve_output_WX.csv",
        # "./eval_out/retrieve_output_TY.csv",
        # "./eval_out/retrieve_output_YB.csv",
        "./eval_out/batch/retrieve_output_WX.csv",
        "./eval_out/batch/retrieve_output_TY.csv",
        "./eval_out/batch/retrieve_output_YB.csv",
    ]

    # collection_name = config.COLLECTION_NAME
    # load_collection(collection_name)
    # # 连接到 Milvus 服务
    # client = MilvusClient(host=config.MILVUS_HOST, port=config.MILVUS_PORT)
    #
    # embedding = get_cached_embedder(local_cache_path="." + config.EMBEDDINGS_CACHE_PATH)

    # 加载环境变量
    load_dotenv()

    for i, dt in enumerate(dt_list):
        results = []
        logging.info("开始检索问题列表...")
        for key, values in dt.items():
            # logging.info(f"正在检索问题：{key}，请稍候...")

            # HyDE 扩展 query
            # hyde_chain = get_hyde_chain()
            # hyde_query = hyde_chain.invoke(input=key)
            # logging.info(f"query: {key}")
            # logging.info(f"HyDE query: {hyde_query}")

            # res = retrieve_query(question=hyde_query, embeddings=embedding, client=client, collect_name=collection_name)

            res = retrieve_query(question=key, embeddings=embedding, client=client, collect_name=collection_name)

            for ddt in res:
                ref_source = values[0]['来源'] if values else ""
                retri_txt = re.sub(r'[\r\n]+', '', ddt['text'])

                ref_source_num = extract_txt_str(ref_source)
                retri_txt_nums = extract_txt_list(retri_txt)

                ddt['参考答案'] = values[0]['答案'] if values else ""
                ddt['参考来源'] = ref_source
                ddt['召回说明'] = "TRUE" if len(retri_txt_nums) > 0 and ref_source_num in set(
                    retri_txt_nums) else "FALSE"

                # ddt['HyDE后新问题'] = hyde_query
                # ddt['query'] = key  # 在调用检索时，query 已被赋值为 HyDE之后的了，因此这里需要在写到CSV文件前覆盖回来

            results.append(res)
            logging.info(f"问题 '{key}' 的检索完成")

        logging.info(f"表单 {i + 1} 的检索完成，写入结果到 CSV 文件...")
        # 展开嵌套列表
        flat_results = [item for sublist in results for item in sublist]

        output_f_name = output_csv_path_list[i]
        if output_fname_ctrl:
            output_f_name = output_f_name.replace(".csv", output_fname_ctrl + ".csv")

        write_results_to_csv(flat_results, output_f_name)

        logging.info(f"表单 {i + 1} 的结果已写入到 {output_f_name}")

    # 释放集合（从内存释放）
    # client.release_collection(collection_name)


def execute_rewrite_retrieval(output_fname_ctrl: str = None):
    logging.info("execute_rewrite_retrieval, 获取 Excel 表格中表单1,2,3 的问题列表...")
    # input_data_path = "../evaluation/data/retrieveInputData_V1_1.xlsx"
    input_data_path = "../evaluation/data/retrieveInputData_V1_2.xlsx"
    sheet_list = [0, 1, 2]  # 需要读取的表单
    # sheet_list = [2]  # 需要读取的表单

    # 读取 Excel 文件并处理数据
    dt_list = read_xlsx_and_process(input_data_path, sheet_list)

    # 输出的名称分别与上述读取的表单对应
    output_csv_path_list = [
        "./eval_out/batch/retrieve_output_WX.csv",
        "./eval_out/batch/retrieve_output_TY.csv",
        "./eval_out/batch/retrieve_output_YB.csv",
    ]

    # 定义 milvus检索器
    milvus_retriever = MilvusRetriever()

    # 加载环境变量
    load_dotenv()

    # model_deepseek = get_deepseek_model(streaming=False)

    model_qwen = get_qwen_model(streaming=False, api_key=os.getenv("DASHSCOPE_API_KEY"))

    # 调用RePhraseQueryRetriever进行查询重写，并获取检索结果
    time.sleep(random.randint(30, 60))  # 随机等待30-60秒，避免请求频率被限制
    re_retriever = query_rewrite_retriever(milvus_retriever, model_qwen)

    # 调用 MultiQueryRetriever 进行查询分解
    # multi_retiever = query_multi_retiever(milvus_retriever, model_deepseek)

    for i, dt in enumerate(dt_list):
        results = []
        logging.info("开始检索问题列表...")
        for key, values in dt.items():
            res = re_retriever.invoke(key)

            # res = multi_retiever.invoke(key)

            # id_set = set(hits['id'] for hits in res)
            # logging.info(f"multi_retiever 检索结果分块 id set -> : ", id_set)

            #  RRF 对检索结果进行重排，取重排前三
            # rrf_res = get_top_n_rrf(res)

            for hit in res:
                ref_source = values[0]['来源'] if values else ""
                retri_txt = re.sub(r'[\r\n]+', '', hit['entity']['text'])

                ref_source_num = extract_txt_str(ref_source)
                retri_txt_nums = extract_txt_list(retri_txt)

                results.append({
                    "query": key,
                    "参考答案": values[0]['答案'] if values else "",
                    "参考来源": ref_source,
                    "text": retri_txt,
                    "distance": "%.4f" % hit['distance'] if hit['distance'] else "0.0000",
                    "metadata": hit['entity']['metadata']['source'] if 'source' in hit['entity']['metadata'] else "",
                    "召回说明": "TRUE" if len(retri_txt_nums) > 0 and ref_source_num in set(retri_txt_nums) else "FALSE"
                })
            logging.info(f"问题 '{key}' 的检索完成")

        logging.info(f"表单 {i + 1} 的检索完成，写入结果到 CSV 文件...")

        output_f_name = output_csv_path_list[i]
        if output_fname_ctrl:
            output_f_name = output_f_name.replace(".csv", output_fname_ctrl + ".csv")

        write_results_to_csv(results, output_f_name)

        logging.info(f"表单 {i + 1} 的结果已写入到 {output_f_name}")


def execute_rewrite_retrieval_batch(batch_num: int, eval_strategy_name: str):
    total = 0
    while total < batch_num:
        execute_rewrite_retrieval(datetime.now().strftime('%Y%m%d%H%M%S') + "_" + eval_strategy_name)

        total += 1

        wait_time = random.randint(30, 60)
        logging.info(f"第 {total} 次执行完成，等待 {wait_time} 秒...")
        # 等待指定时间
        time.sleep(wait_time)

    # 统计结果 （要注意目录下的文件是否都是本批次统计的，使用：batch_eval_strategy_name，控制）
    output_batch_path = "./eval_out/batch"
    read_directory_csv_and_count(output_batch_path, eval_strategy_name)


def execute_retrieval_batch(batch_num: int, eval_strategy_name: str):
    client = MilvusClient(host=config.MILVUS_HOST, port=config.MILVUS_PORT)

    collection_name = config.COLLECTION_NAME
    if client.get_load_state(collection_name)["state"] != 3:
        client.load_collection(collection_name)

    # 等待集合加载完成
    while True:
        try:
            if client.get_load_state(collection_name)["state"] == 3:
                break
        except Exception as e:
            logger.warning(f"检查集合加载状态时出错: {e}")
        logger.info("等待集合加载完成...")
        time.sleep(5)

    embedding = get_cached_embedder(local_cache_path="." + config.EMBEDDINGS_CACHE_PATH)

    i = 0
    while i < batch_num:

        # 在每次执行前检查集合状态
        try:
            # 执行检索
            execute_retrieval(
                embedding=embedding,
                client=client,
                collection_name=collection_name,
                output_fname_ctrl=datetime.now().strftime('%Y%m%d%H%M%S') + "_" + eval_strategy_name
            )
        except pymilvus.exceptions.MilvusException as e:
            if "collection not loaded" in str(e):
                logger.warning("检测到集合未加载，重新加载...")
                client.load_collection(collection_name)
                time.sleep(10)  # 等待重新加载
                continue  # 重试当前批次
            else:
                raise e  # 其他异常直接抛出

        i += 1

        wait_time = random.randint(30, 60)
        logging.info(f"第 {i} 次执行完成，等待 {wait_time} 秒...")
        # 等待指定时间
        time.sleep(wait_time)

    client.release_collection(collection_name)

    # 统计结果 （要注意目录下的文件是否都是本批次统计的，使用：batch_eval_strategy_name，控制）
    output_batch_path = "./eval_out/batch"
    read_directory_csv_and_count(output_batch_path, eval_strategy_name)


def hybrid_retrieve_evaluation(output_fname_ctrl: str = None):
    logging.info("hybrid_retrieve_evaluation start...")

    milvus_util = MilvusUtils()
    collection_var = milvus_util.get_collection(config.COLLECTION_NAME)

    bgem3_ef = BGEM3EmbeddingFunction(use_fp16=config.BGEM3_USE_FP16, device=config.BGEM3_DEVICE)

    input_data_path = "../evaluation/data/retrieveInputData_V1_2.xlsx"
    sheet_list = [0, 1, 2]  # 需要读取的表单
    # sheet_list = [1]  # 需要读取的表单

    # 读取 Excel 文件并处理数据
    dt_list = read_xlsx_and_process(input_data_path, sheet_list)

    # 输出的名称分别与上述读取的表单对应
    output_csv_path_list = [
        "./eval_out/batch/retrieve_output_WX.csv",
        "./eval_out/batch/retrieve_output_TY.csv",
        "./eval_out/batch/retrieve_output_YB.csv",
    ]

    for i, dt in enumerate(dt_list):
        results = []
        logging.info("开始检索问题列表...")
        for key, values in dt.items():

            res_list = milvus_hybrid_retrieve(collection=collection_var, bgem3_ef=bgem3_ef, query=key)

            for hit in res_list:
                ref_source = values[0]['来源'] if values else ""
                retrieve_txt = re.sub(r'[\r\n]+', '', hit.entity.get('text'))

                ref_source_num = extract_txt_str(ref_source)
                retrieve_txt_nums = extract_txt_list(retrieve_txt)
                results.append({
                    "query": key,
                    "参考答案": values[0]['答案'] if values else "",
                    "参考来源": ref_source,
                    "text": retrieve_txt,
                    "distance": "%.4f" % hit.distance if hit.distance else "0.0000",
                    "metadata": hit.entity.get('metadata')['source'] if 'source' in hit.entity.get('metadata') else "",
                    "召回说明": "TRUE" if len(retrieve_txt_nums) > 0 and ref_source_num in set(
                        retrieve_txt_nums) else "FALSE"
                })

            logging.info(f"问题 '{key}' 的检索完成")

        logging.info(f"表单 {i + 1} 的检索完成，写入结果到 CSV 文件...")
        # 展开嵌套列表
        # flat_results = [item for sublist in results for item in sublist]

        output_f_name = output_csv_path_list[i]
        if output_fname_ctrl:
            output_f_name = output_f_name.replace(".csv", output_fname_ctrl + ".csv")

        write_results_to_csv(results, output_f_name)

        logging.info(f"表单 {i + 1} 的结果已写入到 {output_f_name}")


"""
    这个文件是用于评估 RAG 检索效果的。
"""
if __name__ == '__main__':
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # execute_retrieval()

    # execute_rewrite_retrieval()

    # 根据实际情况赋值 eval_strategy_name
    # execute_retrieval_batch(10, "")

    hybrid_retrieve_evaluation("987654321_hybridTop3_")

    # 统计结果 （要注意目录下的文件是否都是本批次统计的，使用：batch_flag，控制）
    out_batch_path = "./eval_out/batch"
    read_directory_csv_and_count(out_batch_path, "987654321_hybridTop3_")
