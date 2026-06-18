"""Prompt 模板模块

定义 RAG 问答的 Prompt 模板。
"""

from langchain_core.prompts import ChatPromptTemplate

# RAG 问答系统提示
RAG_SYSTEM_PROMPT = """你是一个技术文档问答助手。请根据以下检索到的文档片段回答用户的问题。

规则：
1. 只根据提供的文档内容回答，不要编造信息
2. 如果文档中没有相关信息，请明确说"根据现有文档，我无法找到相关信息"
3. 回答时请引用出处，格式为：（来源：{{source}}，第{{page}}页）
4. 用中文回答，保持专业但易懂"""

# RAG 问答 Prompt 模板
RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human", """检索到的文档片段：

{context}

用户问题：{question}

请回答："""),
])


def format_context(documents: list) -> str:
    """将检索到的文档片段格式化为上下文文本

    Args:
        documents: 检索到的 Document 列表

    Returns:
        str: 格式化后的上下文文本
    """
    context_parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page", "未知页码")
        context_parts.append(
            f"--- 片段 {i}（来源：{source}，第{page}页）---\n{doc.page_content}"
        )

    return "\n\n".join(context_parts)
