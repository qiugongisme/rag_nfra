from langchain_core.prompts import PromptTemplate

check_prompt_template = """你是一位十分熟悉国家金融监督管理总局政策法规的专家，请判断下面问题是否和国家金融监督总局的政策法规相关，相关请回答：YES，不想关请回答：NO，不允许其它回答，不允许在答案中添加其他任何成分。
问题: {question}
"""

CHECK_NFRA_PROMPT = PromptTemplate(
    template=check_prompt_template, input_variables=["question"]
)

# 初始版本
prompt_template = """你是一位十分熟悉国家金融监督管理总局政策法规的专家，请你结合以下内容回答问题:
{retrieve_context}

问题: {question}
"""
# 更新版本，增加了回答要求
prompt_template_2 = """你是一位十分熟悉国家金融监督管理总局政策法规的专家，请你结合上下文，回答问题:
上下文：
{retrieve_context}

问题: {question}

#要求：
1. 回答要简洁明了，通俗易懂
2. 要符合国家金融监督管理总局政策法规
"""

QUERY_PROMPT = PromptTemplate(
    template=prompt_template_2, input_variables=["retrieve_context", "question"]
)

re_query_prompt_template = """您是 AI 语言模型助手。您的任务是生成给定问题的3个不同问法，用来从矢量数据库中检索相关文档。
通过对问题生成多个不同的问法，来克服基于内积（IP）的相似性检索的一些限制。提供这些用换行符分隔的替代问题，不要给出多余的回答。
问题：{question}"""
RE_QUERY_PROMPT_TEMPLATE = PromptTemplate(
    template=re_query_prompt_template, input_variables=["question"]
)

multi_query_prompt_template = """您是 AI 语言模型助手。您的任务是生成给定用户问题的3个不同版本，用来从矢量数据库中检索相关文档。
通过对用户问题生成多个视角，来帮助用户克服基于内积（IP）的相似性搜索的一些限制。提供这些用换行符分隔的替代问题，不要给出多余的回答。
问题：{question}"""
MULTI_QUERY_PROMPT_TEMPLATE = PromptTemplate(
    template=multi_query_prompt_template, input_variables=["question"]
)
