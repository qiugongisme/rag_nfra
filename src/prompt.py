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