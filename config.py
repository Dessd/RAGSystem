"""DocuMind 配置管理模块

统一管理所有配置项，优先从环境变量读取，使用默认值兜底。
"""

import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


class Config:
    """应用配置"""

    # DeepSeek API
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"

    # 文本分割参数
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # 检索参数
    TOP_K: int = int(os.getenv("TOP_K", "5"))

    # ChromaDB 持久化目录
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    # 默认集合名称
    DEFAULT_COLLECTION: str = "documind"

    # 支持的文件格式
    SUPPORTED_EXTENSIONS: list = [".pdf", ".md", ".txt", ".docx", ".doc", ".html", ".htm"]

    # HuggingFace Embedding 模型
    EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"

    @classmethod
    def validate(cls) -> bool:
        """验证必要配置是否完整"""
        if not cls.DEEPSEEK_API_KEY:
            raise ValueError(
                "缺少 DEEPSEEK_API_KEY，请在 .env 文件中配置。\n"
                "获取地址：https://platform.deepseek.com/"
            )
        return True


config = Config()
