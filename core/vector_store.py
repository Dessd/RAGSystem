"""向量存储模块

使用 ChromaDB 进行向量存储和管理。
"""

from typing import Optional, List

from langchain_core.documents import Document
from langchain_chroma import Chroma

from config import config
from core.embeddings import get_embedding_model


def get_vector_store(
    collection_name: str = config.DEFAULT_COLLECTION,
) -> Chroma:
    """获取或创建 ChromaDB 向量存储实例

    Args:
        collection_name: 集合名称

    Returns:
        Chroma: 向量存储实例
    """
    embedding = get_embedding_model()

    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )


def add_documents(
    documents: List[Document],
    collection_name: str = config.DEFAULT_COLLECTION,
) -> None:
    """将文档添加到向量存储

    Args:
        documents: 分割后的文档片段列表
        collection_name: 集合名称
    """
    vector_store = get_vector_store(collection_name)
    vector_store.add_documents(documents)


def delete_collection(collection_name: str = config.DEFAULT_COLLECTION) -> None:
    """删除指定集合

    Args:
        collection_name: 集合名称
    """
    vector_store = get_vector_store(collection_name)
    vector_store.delete_collection()
