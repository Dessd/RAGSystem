"""文本分割模块测试"""

from langchain_core.documents import Document

from core.text_splitter import split_documents


def test_split_documents():
    """测试文档分割"""
    # 创建一个长文档
    long_text = "这是一段测试文本。" * 100  # 约900字符
    doc = Document(page_content=long_text, metadata={"source": "test.txt", "page": 1})

    chunks = split_documents([doc])

    # 应该被分割成多个片段
    assert len(chunks) > 1

    # 每个片段应该有 metadata
    for chunk in chunks:
        assert "source" in chunk.metadata
        assert "chunk_index" in chunk.metadata


def test_short_document_not_split():
    """测试短文档不被分割"""
    short_text = "这是一段短文本。"
    doc = Document(page_content=short_text, metadata={"source": "short.txt", "page": 1})

    chunks = split_documents([doc])

    assert len(chunks) == 1
    assert chunks[0].page_content == short_text


def test_metadata_preserved():
    """测试 metadata 在分割后保留"""
    text = "测试内容。" * 50
    doc = Document(
        page_content=text,
        metadata={"source": "api.pdf", "page": 12}
    )

    chunks = split_documents([doc])

    for chunk in chunks:
        assert chunk.metadata["source"] == "api.pdf"
        assert chunk.metadata["page"] == 12
