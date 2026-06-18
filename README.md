# DocuMind

基于 LangChain + DeepSeek 的技术文档知识库问答系统

## 功能特性

- 📄 支持 PDF/Markdown/TXT 文档上传
- ✂️ 智能文档切分和向量化
- 🔍 基于相似度/MMR 的智能检索
- 💬 答案附带原文出处引用
- 🗣️ 支持多轮对话

## 技术架构

```
用户上传文档 → 文档加载器 → 文本分割器 → Embedding模型 → ChromaDB
                                                            ↓
用户提问 → 问题向量化 → 相似度检索 → Prompt模板 → DeepSeek → 答案+出处
```

| 层级 | 技术 | 说明 |
|------|------|------|
| LLM | DeepSeek | 高性价比、中文能力强 |
| 框架 | LangChain | RAG 流程编排 |
| 向量数据库 | ChromaDB | 轻量级、本地运行 |
| Embedding | BAAI/bge-small-zh-v1.5 | 本地中文向量模型，无需 API |
| 前端 | Streamlit | 快速原型展示 |

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/yourname/documind.git
cd documind
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key
# 获取地址：https://platform.deepseek.com/
```

### 4. 运行

```bash
streamlit run app.py
```

浏览器会自动打开 http://localhost:8501

## 项目结构

```
documind/
├── README.md                    # 项目说明
├── requirements.txt             # Python 依赖
├── .env.example                 # 环境变量模板
├── .gitignore
├── app.py                       # Streamlit 主入口
├── config.py                    # 配置管理
├── core/
│   ├── document_loader.py       # 文档加载（PDF/MD/TXT）
│   ├── text_splitter.py         # 文本分割策略
│   ├── embeddings.py            # Embedding 模型封装
│   ├── vector_store.py          # ChromaDB 向量存储
│   ├── retriever.py             # 检索器（相似度 + MMR）
│   └── chain.py                 # LangChain RAG 链
├── prompts/
│   └── templates.py             # Prompt 模板
├── utils/
│   ├── file_utils.py            # 文件处理工具
│   └── logger.py                # 日志工具
├── tests/                       # 测试用例
├── data/sample_docs/            # 示例文档
├── screenshots/                 # 项目截图
└── docs/architecture.md         # 架构说明
```

## 技术细节

### 文本分割策略

使用 `RecursiveCharacterTextSplitter`，参数：
- `chunk_size=500`：约250个中文字，平衡上下文和关键信息
- `chunk_overlap=50`：重叠区域保证语义连续
- `separators`：优先按段落 `\n\n`，其次按中文句号 `。`，最后按空格

### 检索模式

| 模式 | 特点 | 适用场景 |
|------|------|----------|
| 相似度检索 | 速度快，精确匹配 | 精确问答 |
| MMR 检索 | 平衡相关性和多样性 | 探索性问答 |

## TODO

- [ ] 支持更多文档格式（Word、HTML）
- [ ] 对话历史持久化
- [ ] 支持多知识库切换
- [ ] 添加文档预览功能
- [ ] 部署为 Docker 容器

## License

MIT
