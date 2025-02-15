import os
import duckdb
import uuid
import streamlit as st
from time import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import (
    WebBaseLoader, PyPDFLoader, Docx2txtLoader, TextLoader
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

gemini_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
DB_DOCS_LIMIT = 10

def initialize_vector_db():
    db = duckdb.connect(database=':memory:', read_only=False)
    db.execute("CREATE TABLE IF NOT EXISTS vectors (id UUID PRIMARY KEY, content TEXT)")
    return db

def add_document_to_duckdb(db, doc_id, content):
    db.execute("INSERT INTO vectors (id, content) VALUES (?, ?)", (doc_id, content))

def retrieve_documents(db):
    return db.execute("SELECT * FROM vectors").fetchall()

def load_url_to_db():
    if "rag_url" in st.session_state and st.session_state.rag_url:
        url = st.session_state.rag_url
        docs = []
        if url not in st.session_state.rag_sources:
            if len(st.session_state.rag_sources) < 10:
                try:
                    loader = WebBaseLoader(url)
                    docs.extend(loader.load())
                    st.session_state.rag_sources.append(url)
                except Exception as e:
                    st.error(f"Error While Loading Documents From {url}: {e}")
                if docs:
                    _split_and_load_docs(docs)
                    st.toast(f"Document From URL *{url}* Loaded Successfully", icon="✅")
                else:
                    st.error("Maximum Number Of Documents Reached 😨")

def _split_and_load_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=1000,
    )
    document_chunks = text_splitter.split_documents(docs)
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = initialize_vector_db()
    for doc in document_chunks:
        add_document_to_duckdb(st.session_state.vector_db, str(uuid.uuid4()), doc.page_content)

def load_docs_to_db():
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = []
        for doc_file in st.session_state.rag_docs:
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                    try:
                        if doc_file.type == "application/pdf":
                            loader = PyPDFLoader(doc_file)
                        elif doc_file.name.endswith(".docx"):
                            loader = Docx2txtLoader(doc_file)
                        elif doc_file.type in ["text/plain", "text/markdown"]:
                            loader = TextLoader(doc_file)
                        else:
                            st.warning(f"Document type {doc_file.type} not supported.")
                            continue
                        docs.extend(loader.load())
                        st.session_state.rag_sources.append(doc_file.name)
                    except Exception as e:
                        st.toast(f"Error Loading Document {doc_file.name}: {e}", icon="🙏")
        if docs:
            _split_and_load_docs(docs)
            st.toast(f"Documents Loaded Successfully.", icon="✅")
            st.balloons()

def _get_context_retriever_chain(vector_db, llm):
    retriever = vector_db.execute("SELECT content FROM vectors").fetchall()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up relevant information.")
    ])
    return create_history_aware_retriever(llm, retriever, prompt)

def get_conversational_rag_chain(llm):
    retriever_chain = _get_context_retriever_chain(st.session_state.vector_db, llm)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Your name is Ghazala. {context}"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def stream_llm_response(llm_stream, messages):
    response_message = ""
    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk
    st.session_state.messages.append({"role": "assistant", "content": response_message})

def stream_rag_llm_response(llm_stream, messages):
    conversation_rag_chain = get_conversational_rag_chain(llm_stream)
    response_message = "*(RAG Response)*\n"
    for chunk in conversation_rag_chain.pick("answer").stream({"messages": messages[:-1], "input": messages[-1].content}):
        response_message += chunk
        yield chunk
    st.session_state.messages.append({"role": "assistant", "content": response_message})