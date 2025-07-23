from operator import itemgetter

from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chains.base import Chain
from langchain.output_parsers import BooleanOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap

from src.prompt import CHECK_NFRA_PROMPT, QUERY_PROMPT
from src.retriever import MilvusRetriever, retrieved_deal
from src.utils import get_deepseek_model


def get_check_chain() -> Chain:
    """获取检查链，用于判断问题是否与国家金融监督管理总局（NFRA）政策法规相关的提问。"""
    model = get_deepseek_model()

    check_chain = CHECK_NFRA_PROMPT | model | BooleanOutputParser()

    return check_chain


def get_qa_chain(out_callback: AsyncIteratorCallbackHandler) -> Chain:
    """
    获取问答链，用于回答与国家金融监督管理总局（NFRA）政策法规相关的问题。
    :param out_callback:
    :return: Chain (langchain.chains.base)
    """

    callbacks = [out_callback] if out_callback else []

    # 定义 milvus检索器
    milvus_retriever = MilvusRetriever()
    chain = (
            RunnableMap({
                "retrieve_docs": itemgetter("question") | milvus_retriever, # 从问题中检索相关文档
                "question": lambda x: x["question"] # 获取问题
            }) |
            RunnableMap({
                "retrieve_context": lambda x: retrieved_deal(x["retrieve_docs"]), # 处理检索到的文档
                # "retrieve_context": lambda x: retrieved_deal_eval(x["retrieve_docs"]), # 处理检索到的文档（进行 LLM 评估时使用）
                "question": lambda x: x["question"] # 获取问题，继续传递
            }) |
            RunnableMap({
                "retrieve_context": lambda x: x["retrieve_context"], # 获取检索到的上下文
                # "retrieve_context": lambda x: x["retrieve_context"][0], # 获取检索到的上下文（进行 LLM 评估时使用）
                # "similarity": lambda x: x["retrieve_context"][1], # 获取相似度 evaluation （进行 LLM 评估时使用）
                "prompt": QUERY_PROMPT # 构建查询提示
            }) |
            RunnableMap({
                "retrieve_context": lambda x: x["retrieve_context"],  # 获取检索到的上下文
                # "similarity": lambda x: x["similarity"],  # 获取相似度  evaluation （进行 LLM 评估时使用）
                # 使用模型（默认使用 deepseek-chat）生成答案
                    "answer": itemgetter("prompt") | get_deepseek_model(callbacks=callbacks) | StrOutputParser()
            })
    )

    return chain
