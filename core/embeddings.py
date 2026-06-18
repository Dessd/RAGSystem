"""Embedding 模型封装

使用 TF-IDF 进行文本向量化，完全离线运行，无需下载任何模型。
"""

import hashlib
import numpy as np
from typing import List


class LocalEmbeddings:
    """本地 Embedding 实现，兼容 LangChain 接口

    使用哈希 + TF-IDF 混合方式生成向量，完全离线运行。
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _text_to_vector(self, text: str) -> List[float]:
        """将文本转换为向量"""
        # 使用多个哈希函数生成特征
        vec = np.zeros(self.dim)

        # 分词（简单按字符和标点分割）
        words = []
        current = []
        for char in text:
            if char.isalnum() or '一' <= char <= '鿿':
                current.append(char)
            else:
                if current:
                    words.append(''.join(current))
                    current = []
                if char.strip():
                    words.append(char)
        if current:
            words.append(''.join(current))

        # 使用哈希将每个词映射到向量空间
        for i, word in enumerate(words):
            # 多个哈希种子
            for seed in range(3):
                h = hashlib.md5(f"{seed}:{word}".encode()).hexdigest()
                idx = int(h[:8], 16) % self.dim
                sign = 1 if int(h[8:10], 16) % 2 == 0 else -1
                # TF 权重：词频越高权重越大
                tf = 1.0 + np.log(1 + words.count(word))
                vec[idx] += sign * tf

        # IDF 模拟：常见词降低权重
        common_chars = set('的是在不了有和人这中大为上个国')
        for i, word in enumerate(words):
            if word in common_chars:
                idx = hash(word) % self.dim
                vec[idx] *= 0.3

        # L2 归一化
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量向量化文档"""
        return [self._text_to_vector(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """向量化查询"""
        return self._text_to_vector(text)


def get_embedding_model() -> LocalEmbeddings:
    """获取 Embedding 模型实例

    Returns:
        LocalEmbeddings: 本地 Embedding 模型，完全离线运行
    """
    return LocalEmbeddings()
