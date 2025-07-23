import asyncio
import os

import pandas as pd
from dotenv import load_dotenv

from src.callback import OutCallbackHandler
from src.chain import get_check_chain, get_qa_chain
from src.utils import markdown_to_text


async def qa_llm_eval():
    # 加载环境变量（建议将API密钥放在.env文件中）
    load_dotenv()

    check_chain = get_check_chain()
    out_callback = OutCallbackHandler()
    chain = get_qa_chain(out_callback=out_callback)

    # 读取Excel文件
    input_file_path = 'data\\evaluationData.xlsx'
    sheet_name = [0, 1, 2] # 处理Excel文件中的工作表，第1、2、3个工作表
    sheet_df = pd.read_excel(input_file_path, sheet_name=[0])

    # 遍历每个工作表，每个工作表读取、处理、输出一个新的Excel文件
    for sheet_name, df in sheet_df.items():
        results = []
        for index, row in df.iterrows():
            question = row['问题']
            src_answer = row['答案']
            source = row['来源']

            print(f"\n正在处理问题: {question}")
            is_nfra = check_chain.invoke({"question": question})
            if not is_nfra:
                print("不好意思，问题是关于国家金融监督管理总局政策法规的，请重新输入相关问题。")
                results.append({
                    '问题': question,
                    '答案': src_answer,
                    '来源': source,
                    '上下文': '',
                    '相似度': '',
                    '生成回复': '问题不符合主题'
                })
                continue

            task = asyncio.create_task(chain.ainvoke({"question": question}))

            res = await task
            response_answer = markdown_to_text(res["answer"])

            similarity = res.get('similarity', '')
            retrieve_context = res.get('retrieve_context', '')

            results.append({
                '问题': question,
                '答案': src_answer,
                '来源': source,
                '上下文': retrieve_context,
                '相似度': similarity,
                '生成回复': response_answer
            })

            out_callback.done.clear()

        # 将结果写入新的Excel文件
        output_df = pd.DataFrame(results)

        file_name = "output_test_data_sheet" + str(sheet_name + 1)
        output_file_path = fr'data\{file_name}.xlsx'

        if os.path.exists(output_file_path):  # 确保输出目录存在，如果存在则删除旧文件
            os.remove(output_file_path)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        output_df.to_excel(output_file_path, index=False)
        print(f"结果已保存到 {output_file_path}")


"""
    这个文件是用于评估 RAG 生成效果的。
"""
# 运行异步主函数
if __name__ == "__main__":
    asyncio.run(qa_llm_eval())
