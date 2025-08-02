from langchain.load import dumps
from pymilvus import Hits


def reciprocal_rank_fusion(results: list[Hits], k=60) -> list[tuple]:
    """RRF（Reciprocal Rank Fusion）算法实现
    功能：将多个检索结果列表融合成一个统一的排序列表
    算法原理：
        1. 对于每个检索结果列表中的每个文档
        2. 计算该文档的RRF分数：score = 1 / (rank + k)
        3. 如果同一文档出现在多个列表中，累加其分数
        4. 按最终分数对所有文档进行排序
    优势：
        - rank越小（排名越靠前），分数越高
        - k参数防止分母为0，并调节不同排名之间的差距
        - 多次出现的文档会获得更高的累积分数
    :param results: 多个检索结果 Hits 列表，每个列表包含按相关性排序的文档
    :param k: RRF算法的调节参数，默认值60（经验值）
    :return: list[tuple] 融合后的(Hits, 分数)元组列表，按分数降序排序
    """
    used_scores = {}

    # 遍历该列表中的每个文档
    # for rank, doc in enumerate(docs):
    for rank in range(len(results)):
        hits = results[rank]

        # 将 Hits id 作为唯一标识
        hits_id = dumps(hits['id'])

        # 如果该文档首次出现，初始化分数
        if hits_id not in used_scores:
            used_scores[hits_id] = 0

        # 计算RRF分数并累加
        rrf_score = 1 / (rank + k)
        used_scores[hits_id] += rrf_score

    # 按分数降序排序
    reranked_results = [
        (key_id, score)
        for key_id, score in sorted(used_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # 创建 id 到 hit 对象的映射字典，注意属性类型
    id_to_hit = {str(hit.id): hit for hit in results}

    # 替换元组中的 id 为对应的 hit 对象
    result_list = []
    for id_val, score in reranked_results:
        if id_val in id_to_hit:
            result_list.append((id_to_hit[id_val], score))

    print(f"RRF融合完成，共 {len(reranked_results)} 个唯一文档")

    return result_list


def get_top_n_rrf(results: list[Hits], k=60, top_n=3) -> list[tuple]:
    """将多个检索结果列表融合成一个统一的排序列表，返回 top_n 个（默认为前3个）
    :param results: 多个检索结果 Hits列表，每个列表包含按相关性排序的文档
    :param k: RRF算法的调节参数，默认值60（经验值）
    :param top_n: 融合后按分数降序排序需返回的前 n 个
    :return: 融合后的(Hits, 分数)元组列表，按分数降序排序的前 top_n 个
    """
    reranked_results = reciprocal_rank_fusion(results, k)
    if len(reranked_results) <= top_n:
        return reranked_results
    else:
        return reranked_results[:top_n]
