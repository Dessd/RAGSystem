"""RAG Chain 端到端测试

注意：这些测试需要有效的 ZHIPUAI_API_KEY 环境变量。
跳过条件：无 API Key 时跳过。
"""

import os
import pytest

from core.chain import RAGChain


@pytest.fixture
def rag_chain():
    """创建 RAGChain 实例"""
    if not os.getenv("DEEPSEEK_API_KEY"):
        pytest.skip("缺少 DEEPSEEK_API_KEY，跳过集成测试")
    return RAGChain()


def test_ask_returns_structure(rag_chain):
    """测试 ask() 返回正确的结构"""
    result = rag_chain.ask("你好")

    assert "answer" in result
    assert "sources" in result
    assert "question" in result
    assert isinstance(result["answer"], str)
    assert isinstance(result["sources"], list)


def test_ask_with_history(rag_chain):
    """测试多轮对话"""
    history = [
        {"role": "user", "content": "什么是RAG？"},
        {"role": "assistant", "content": "RAG是检索增强生成的缩写。"},
    ]

    result = rag_chain.ask_with_history("它有什么优势？", history=history)

    assert "answer" in result
    assert len(result["answer"]) > 0
