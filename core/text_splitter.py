"""文本分割模块

将长文档切分为适合向量化的小片段。
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import config


def split_documents(documents: List[Document]) -> List[Document]:
    """将文档列表切分为小片段

    使用 RecursiveCharacterTextSplitter，优先按段落分割，
    其次按句子，最后按空格，保证语义连续性。

    Args:
        documents: 原始文档列表

    Returns:
        List[Document]: 分割后的文档片段列表，保留原始 metadata
    """
    # 中文友好的分隔符列表
    separators = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", "；", ";", " "]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=separators,
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # 为每个 chunk 添加索引信息
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks
