"""Embedding 模型封装

使用 HuggingFace 的 BAAI/bge-small-zh-v1.5 中文 Embedding 模型。
本地运行，无需 API Key。
"""

from langchain_community.embeddings import HuggingFaceEmbeddings

from config import config


def get_embedding_model() -> HuggingFaceEmbeddings:
    """获取 Embedding 模型实例

    使用 BAAI/bge-small-zh-v1.5，一个轻量级中文向量模型，
    首次运行会自动下载模型（约 90MB）。

    Returns:
        HuggingFaceEmbeddings: Embedding 模型
    """
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
