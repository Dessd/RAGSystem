# DocuMind 架构说明

## 系统架构

```
用户上传文档
     |
     v
[文档加载器] ---> 读取 PDF/MD/TXT
     |
     v
[文本分割器] ---> 按 chunk_size=500, overlap=50 切分
     |
     v
[Embedding模型] ---> BAAI/bge-small-zh-v1.5 向量化
     |
     v
[ChromaDB] ---> 存储向量 + 原文元数据
     |
     v
用户提问
     |
     v
[问题向量化] ---> 同一个 Embedding 模型
     |
     v
[相似度检索] ---> 从 ChromaDB 取 Top-K 相关片段
     |
     v
[Prompt模板] ---> 组装：系统提示 + 检索结果 + 用户问题
     |
     v
[DeepSeek] ---> 生成答案
     |
     v
[Streamlit UI] ---> 展示答案 + 出处引用
```

## 核心组件

### 1. 文档加载器 (`core/document_loader.py`)
- 支持 PDF、Markdown、TXT 格式
- 自动提取元数据（文件名、页码）

### 2. 文本分割器 (`core/text_splitter.py`)
- 使用 RecursiveCharacterTextSplitter
- 中文友好的分隔符策略

### 3. Embedding 模型 (`core/embeddings.py`)
- BAAI/bge-small-zh-v1.5（本地运行，无需 API Key）
- 512 维向量

### 4. 向量存储 (`core/vector_store.py`)
- ChromaDB 本地持久化
- 支持增量添加

### 5. 检索器 (`core/retriever.py`)
- 相似度检索：精确匹配
- MMR 检索：平衡相关性和多样性

### 6. RAG Chain (`core/chain.py`)
- 串联检索和生成
- 支持多轮对话
