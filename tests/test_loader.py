"""文档加载模块测试"""

import os
import tempfile
import pytest

from core.document_loader import load_document, UnsupportedFormatError


def test_load_txt_file():
    """测试加载 TXT 文件"""
    # 创建临时 TXT 文件
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("这是一个测试文档。\n包含多行内容。\n")
        tmp_path = f.name

    try:
        docs = load_document(tmp_path)
        assert len(docs) > 0
        assert "测试文档" in docs[0].page_content
        assert "source" in docs[0].metadata
    finally:
        os.unlink(tmp_path)


def test_unsupported_format():
    """测试不支持的文件格式"""
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        tmp_path = f.name

    try:
        with pytest.raises(UnsupportedFormatError):
            load_document(tmp_path)
    finally:
        os.unlink(tmp_path)


def test_file_not_found():
    """测试文件不存在"""
    with pytest.raises(FileNotFoundError):
        load_document("/nonexistent/file.txt")
