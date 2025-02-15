import os
import sys
import uuid
import streamlit as st
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage
from rag_utils import (
    stream_llm_response,
    load_docs_to_db,
    load_url_to_db,
    stream_rag_llm_response,
    initialize_vector_db
)
from dotenv import load_dotenv
import duckdb

# Ensure correct SQLite handling on Linux systems for Streamlit Cloud
ios_name = os.name
if ios_name == "posix":
    try:
        import pysqlite3
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    except ImportError:
        pass

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
gemini_api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(
    page_title="Ghazala Companion Bot",
    page_icon="🕵️‍♀️",
    layout="centered",
    initial_sidebar_state="expanded",
)

MODELS = [
    "deepseek-r1-distill-llama-70b",
    "llama-3.3-70b-versatile",
    "qwen-2.5-32b",
]

st.title("📚🔍 Do your LLM even RAG bro? 🤖💬")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! I am Ghazala Bot. How can I assist you today?"},
    ]

st.session_state.groq_api_key = groq_api_key
st.session_state.vector_db = initialize_vector_db()

with st.sidebar:
    st.title("Ghazala Chatbot Assistant")
    st.divider()
    st.session_state.model = st.selectbox("🤖 Select a Model", MODELS)

    is_vector_db_loaded = "vector_db" in st.session_state and st.session_state.vector_db is not None
    st.toggle("Use RAG", value=is_vector_db_loaded, key="use_rag", disabled=not is_vector_db_loaded)

    if st.button("Clear Chat 😬", type="primary"):
        st.session_state.messages = [{"role": "assistant", "content": "Hi there! I am Ghazala Assistant, how can I assist you today?"}]

    st.header("RAG Sources 😎")
    st.file_uploader("🖼 Upload A Document", type=["pdf", "txt", "docx", "md"], accept_multiple_files=True, on_change=load_docs_to_db, key="rag_docs")
    st.text_input("🌐 Introduce a URL 🔗", placeholder="https://www.google.com.pk/", on_change=load_url_to_db, key="rag_url")

llm_stream_groq = ChatGroq(
    api_key=groq_api_key,
    model=st.session_state.model,
    temperature=0.3,
    streaming=True,
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Your Message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🕵️‍♀️"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="🤖"):
        messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages]
        if not st.session_state.use_rag:
            st.write_stream(stream_llm_response(llm_stream_groq, messages))
        else:
            st.write_stream(stream_rag_llm_response(llm_stream_groq, messages))
