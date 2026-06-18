"""DocuMind - 技术文档智能问答系统

基于 LangChain + DeepSeek 的 RAG 知识库问答系统。
支持多知识库、对话历史持久化、文档预览。
"""

import os
import json
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

from config import config
from core.document_loader import load_document
from core.text_splitter import split_documents
from core.vector_store import add_documents, delete_collection, get_vector_store
from core.chain import RAGChain
from utils.file_utils import save_uploaded_file

# ========== 常量 ==========
HISTORY_DIR = Path("chat_history")
HISTORY_DIR.mkdir(exist_ok=True)

# ========== 页面配置 ==========
st.set_page_config(
    page_title="DocuMind - 技术文档智能问答",
    page_icon="📚",
    layout="wide",
)

# ========== 工具函数 ==========
def save_chat_history(kb_name: str, messages: list):
    """保存对话历史到文件"""
    history_file = HISTORY_DIR / f"{kb_name}.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


def load_chat_history(kb_name: str) -> list:
    """加载对话历史"""
    history_file = HISTORY_DIR / f"{kb_name}.json"
    if history_file.exists():
        with open(history_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def get_knowledge_bases() -> list:
    """获取所有知识库列表"""
    kb_dir = Path(config.CHROMA_PERSIST_DIR)
    if not kb_dir.exists():
        return ["default"]
    kbs = [d.name for d in kb_dir.iterdir() if d.is_dir()]
    return kbs if kbs else ["default"]


def get_uploaded_files(kb_name: str) -> list:
    """获取知识库已上传的文件列表"""
    meta_file = HISTORY_DIR / f"{kb_name}_files.json"
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_uploaded_files(kb_name: str, files: list):
    """保存已上传文件列表"""
    meta_file = HISTORY_DIR / f"{kb_name}_files.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(files, f, ensure_ascii=False, indent=2)


# ========== Session State 初始化 ==========
if "current_kb" not in st.session_state:
    st.session_state.current_kb = "default"
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

    # ========== 知识库管理 ==========
    st.subheader("🗄️ 知识库管理")
    kb_list = get_knowledge_bases()

    # 知识库选择
    selected_kb = st.selectbox(
        "选择知识库",
        options=kb_list,
        index=kb_list.index(st.session_state.current_kb) if st.session_state.current_kb in kb_list else 0,
    )

    # 切换知识库时加载对应数据
    if selected_kb != st.session_state.current_kb:
        st.session_state.current_kb = selected_kb
        st.session_state.messages = load_chat_history(selected_kb)
        st.session_state.uploaded_files = get_uploaded_files(selected_kb)
        st.rerun()

    # 新建知识库
    new_kb = st.text_input("新建知识库", placeholder="输入名称...")
    if st.button("➕ 创建") and new_kb:
        if new_kb not in kb_list:
            st.session_state.current_kb = new_kb
            st.session_state.messages = []
            st.session_state.uploaded_files = []
            save_chat_history(new_kb, [])
            save_uploaded_files(new_kb, [])
            st.success(f"知识库 '{new_kb}' 已创建")
            st.rerun()
        else:
            st.warning("知识库已存在")

    st.divider()

    # ========== 文档上传区 ==========
    st.subheader("📄 上传文档")
    uploaded_files = st.file_uploader(
        "支持 PDF / Markdown / TXT / Word / HTML",
        type=["pdf", "md", "txt", "docx", "doc", "html", "htm"],
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

                        # 存入向量数据库（使用当前知识库名作为 collection）
                        add_documents(chunks, collection_name=st.session_state.current_kb)

                        st.session_state.uploaded_files.append(uploaded_file.name)
                        save_uploaded_files(st.session_state.current_kb, st.session_state.uploaded_files)
                        st.success(f"✅ {uploaded_file.name} 已加载（{len(chunks)} 个片段）")
                    except Exception as e:
                        st.error(f"❌ 处理失败: {e}")

    # 已上传文档列表 + 预览
    if st.session_state.uploaded_files:
        st.divider()
        st.subheader("📋 已加载文档")
        for file_name in st.session_state.uploaded_files:
            with st.expander(f"📄 {file_name}"):
                # 尝试读取并预览文档内容
                try:
                    tmp_dir = tempfile.gettempdir()
                    preview_path = None
                    # 在临时目录中查找文件
                    for root, _, files in os.walk(tmp_dir):
                        if file_name in files:
                            preview_path = os.path.join(root, file_name)
                            break

                    if preview_path and os.path.exists(preview_path):
                        with open(preview_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read(1000)  # 只预览前1000字符
                        st.text_area("内容预览", content, height=150, disabled=True, label_visibility="collapsed")
                    else:
                        st.caption("预览不可用")
                except:
                    st.caption("预览不可用")

    # 清空知识库
    st.divider()
    if st.button("🗑️ 清空当前知识库", type="secondary", use_container_width=True):
        try:
            delete_collection(st.session_state.current_kb)
            st.session_state.uploaded_files = []
            st.session_state.messages = []
            save_uploaded_files(st.session_state.current_kb, [])
            save_chat_history(st.session_state.current_kb, [])
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
st.caption(f"当前知识库：**{st.session_state.current_kb}** | 上传文档后即可提问")

# 加载当前知识库的对话历史
if not st.session_state.messages:
    st.session_state.messages = load_chat_history(st.session_state.current_kb)

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
            st.error("⚠️ RAG 链初始化失败，请检查 DEEPSEEK_API_KEY 配置。")
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

                    # 持久化保存
                    save_chat_history(st.session_state.current_kb, st.session_state.messages)

                except Exception as e:
                    st.error(f"生成回答失败: {e}")
