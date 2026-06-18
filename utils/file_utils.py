"""文件处理工具模块"""

import os
import tempfile
from typing import Optional


def save_uploaded_file(uploaded_file, target_dir: str) -> str:
    """保存上传的文件到指定目录

    Args:
        uploaded_file: Streamlit UploadedFile 对象
        target_dir: 目标目录

    Returns:
        str: 保存后的文件路径
    """
    os.makedirs(target_dir, exist_ok=True)
    file_path = os.path.join(target_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def get_file_extension(file_name: str) -> str:
    """获取文件扩展名（小写）

    Args:
        file_name: 文件名

    Returns:
        str: 文件扩展名，如 ".pdf"
    """
    return os.path.splitext(file_name)[1].lower()
