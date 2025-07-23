from langchain.callbacks import AsyncIteratorCallbackHandler

"""

    回调处理器，流式处理 LangChain 生成的输出，用于异步迭代输出，构建需要逐步消费 LangChain 输出的异步应用。
    目前没有额外的实现，保留以便未来扩展。
    
"""
class OutCallbackHandler(AsyncIteratorCallbackHandler):
    pass

    def on_chat_model_start(self, *args, **kwargs):
        pass