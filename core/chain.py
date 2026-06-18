"""RAG Chain 核心链

串联检索 + 生成，完成完整的 RAG 流程。
使用 DeepSeek 作为大语言模型。
"""

from typing import List, Dict, Any

from langchain_openai import ChatOpenAI

from config import config
from core.retriever import retrieve
from prompts.templates import RAG_PROMPT_TEMPLATE, format_context


class RAGChain:
    """RAG 问答链"""

    def __init__(self):
        """初始化 RAG 链"""
        config.validate()

        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
            temperature=0.7,
        )

    def ask(
        self,
        question: str,
        k: int = config.TOP_K,
        search_type: str = "similarity",
    ) -> Dict[str, Any]:
        """单轮问答

        Args:
            question: 用户问题
            k: 检索的片段数量
            search_type: 检索类型 - "similarity" 或 "mmr"

        Returns:
            dict: {
                "answer": "生成的答案",
                "sources": [{"source": "文件名", "page": 页码, "content": "片段内容"}],
                "question": "原始问题"
            }
        """
        # 1. 检索相关片段
        docs = retrieve(question, k=k, search_type=search_type)

        # 2. 格式化上下文
        context = format_context(docs)

        # 3. 组装 Prompt 并调用 LLM
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        response = self.llm.invoke(prompt)

        # 4. 提取来源信息
        sources = []
        for doc in docs:
            sources.append({
                "source": doc.metadata.get("source", "未知来源"),
                "page": doc.metadata.get("page", "未知页码"),
                "content": doc.page_content[:200],
            })

        return {
            "answer": response.content,
            "sources": sources,
            "question": question,
        }

    def ask_with_history(
        self,
        question: str,
        history: List[Dict[str, str]],
        k: int = config.TOP_K,
        search_type: str = "similarity",
    ) -> Dict[str, Any]:
        """多轮对话

        Args:
            question: 用户问题
            history: 对话历史 [{"role": "user/assistant", "content": "..."}]
            k: 检索的片段数量
            search_type: 检索类型

        Returns:
            dict: 同 ask() 返回格式
        """
        # 构建包含历史的问题
        history_text = ""
        if history:
            history_parts = []
            for msg in history[-6:]:  # 只取最近3轮对话
                role = "用户" if msg["role"] == "user" else "助手"
                history_parts.append(f"{role}: {msg['content']}")
            history_text = "\n".join(history_parts)

        # 将历史信息加入问题进行检索
        enhanced_query = question
        if history_text:
            enhanced_query = f"对话历史：\n{history_text}\n\n当前问题：{question}"

        return self.ask(enhanced_query, k=k, search_type=search_type)
