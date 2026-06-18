"""检索器模块

支持相似度检索和 MMR（最大边际相关性）检索两种模式。
"""

from typing import List, Literal

from langchain_core.documents import Document

from config import config
from core.vector_store import get_vector_store


def retrieve(
    query: str,
    k: int = config.TOP_K,
    search_type: Literal["similarity", "mmr"] = "similarity",
    collection_name: str = config.DEFAULT_COLLECTION,
) -> List[Document]:
    """根据用户问题检索最相关的文档片段

    Args:
        query: 用户问题
        k: 返回的片段数量
        search_type: 检索类型 - "similarity"（纯相似度）或 "mmr"（最大边际相关性）
        collection_name: 集合名称

    Returns:
        List[Document]: 检索到的相关片段
    """
    vector_store = get_vector_store(collection_name)

    if search_type == "mmr":
        # MMR 检索：平衡相关性和多样性
        results = vector_store.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=k * 2,  # 候选集更大，MMR 效果更好
        )
    else:
        # 纯相似度检索
        results = vector_store.similarity_search(query=query, k=k)

    return results
