"""DocuMind - 技术文档智能问答系统

基于 LangChain + GLM-4 的 RAG 知识库问答系统。
"""

import os
import tempfile

import streamlit as st

from config import config
from core.document_loader import load_document
from core.text_splitter import split_documents
from core.vector_store import add_documents, delete_collection, get_vector_store
from core.chain import RAGChain
from utils.file_utils import save_uploaded_file

# ========== 页面配置 ==========
st.set_page_config(
    page_title="DocuMind - 技术文档智能问答",
    page_icon="📚",
    layout="wide",
)

# ========== Session State 初始化 ==========
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "rag_chain" not in st.session_state:
    try:
        st.session_state.rag_chain = RAGChain()
    except Exception as e:
        st.session_state.rag_chain = None


# ========== 侧边栏 ==========
with st.sidebar:
    st.title("📚 DocuMind")
    st.caption("技术文档智能问答系统")

    st.divider()

    # 文档上传区
    st.subheader("📄 上传文档")
    uploaded_files = st.file_uploader(
        "支持 PDF / Markdown / TXT",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.uploaded_files:
                with st.spinner(f"正在处理 {uploaded_file.name}..."):
                    try:
                        # 保存文件到临时目录
                        tmp_dir = tempfile.mkdtemp()
                        file_path = save_uploaded_file(uploaded_file, tmp_dir)

                        # 加载并分割文档
                        docs = load_document(file_path)
                        chunks = split_documents(docs)

                        # 存入向量数据库
                        add_documents(chunks)

                        st.session_state.uploaded_files.append(uploaded_file.name)
                        st.success(f"✅ {uploaded_file.name} 已加载（{len(chunks)} 个片段）")
                    except Exception as e:
                        st.error(f"❌ 处理失败: {e}")

    # 已上传文档列表
    if st.session_state.uploaded_files:
        st.divider()
        st.subheader("📋 已加载文档")
        for file_name in st.session_state.uploaded_files:
            st.text(f"• {file_name}")

    # 清空知识库
    st.divider()
    if st.button("🗑️ 清空知识库", type="secondary", use_container_width=True):
        try:
            delete_collection()
            st.session_state.uploaded_files = []
            st.session_state.messages = []
            st.success("知识库已清空")
            st.rerun()
        except Exception as e:
            st.error(f"清空失败: {e}")

    # 设置区
    st.divider()
    st.subheader("⚙️ 设置")
    top_k = st.slider("Top-K（检索片段数）", min_value=1, max_value=10, value=config.TOP_K)
    search_type = st.radio(
        "检索模式",
        options=["similarity", "mmr"],
        format_func=lambda x: "🔍 相似度检索" if x == "similarity" else "🎯 MMR 检索",
        index=0,
    )


# ========== 主区域 ==========
st.title("💬 技术文档智能问答")
st.caption("上传技术文档，用自然语言提问，获得精准答案并附带原文出处")

# 显示对话历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📎 查看出处"):
                for source in message["sources"]:
                    st.markdown(
                        f"**{source['source']}** - 第{source['page']}页\n\n"
                        f"> {source['content']}..."
                    )

# 用户输入
if prompt := st.chat_input("请输入你的问题..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成回答
    with st.chat_message("assistant"):
        rag_chain = st.session_state.rag_chain

        if rag_chain is None:
            st.error("⚠️ RAG 链初始化失败，请检查 ZHIPUAI_API_KEY 配置。")
        elif not st.session_state.uploaded_files:
            st.warning("⚠️ 请先上传文档再提问。")
        else:
            with st.spinner("正在思考..."):
                try:
                    result = rag_chain.ask(
                        question=prompt,
                        k=top_k,
                        search_type=search_type,
                    )

                    st.markdown(result["answer"])

                    # 显示出处
                    if result["sources"]:
                        with st.expander("📎 查看出处"):
                            for source in result["sources"]:
                                st.markdown(
                                    f"**{source['source']}** - 第{source['page']}页\n\n"
                                    f"> {source['content']}..."
                                )

                    # 保存到对话历史
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    })

                except Exception as e:
                    st.error(f"生成回答失败: {e}")
