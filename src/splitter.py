from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter


class FileSplitter(RecursiveCharacterTextSplitter):
    def __init__(self, **kwargs: Any) -> None:
        """Initialize a FileSplitter."""
        separators = [r"第\S*条 "] # 使用正则表达式匹配“第X条”的开头
        is_separator_regex = True # 设置为True以启用正则表达式分隔符

        super().__init__(separators=separators, is_separator_regex=is_separator_regex, **kwargs)
