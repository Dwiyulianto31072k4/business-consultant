import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
import requests
import os
from bs4 import BeautifulSoup

# === KONFIGURASI UTAMA ===
st.set_page_config(page_title="Chatbot AI", page_icon="💬", layout="wide")

st.markdown(
    """
    <style>
        body {
            background: linear-gradient(135deg, #1c1c1c, #2d2d2d);
            font-family: 'Arial', sans-serif;
            color: white;
        }
        .chat-container {
            width: 100%;
            max-width: 700px;
            margin: auto;
            padding: 20px;
            height: 75vh;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
        }
        .chat-bubble {
            padding: 12px;
            margin: 8px 0;
            border-radius: 20px;
            max-width: 80%;
            font-size: 14px;
            display: inline-block;
            animation: fadeIn 0.3s ease-in-out;
        }
        .chat-bubble-user {
            background: linear-gradient(135deg, #8a2be2, #6e3ff2);
            color: white;
            text-align: right;
            align-self: flex-end;
            float: right;
            margin-right: 10px;
        }
        .chat-bubble-bot {
            background-color: white;
            color: black;
            text-align: left;
            align-self: flex-start;
            float: left;
            margin-left: 10px;
        }
        .file-bubble {
            background: #3a3a3a;
            color: white;
            padding: 8px;
            border-radius: 10px;
            max-width: 80%;
            font-size: 12px;
        }
        .input-container {
            position: fixed;
            bottom: 0;
            width: 100%;
            background: #1c1c1c;
            padding: 10px 20px;
            box-shadow: 0 -2px 5px rgba(255, 255, 255, 0.1);
            display: flex;
            justify-content: center;
            align-items: center;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """,
    unsafe_allow_html=True
)

# === LOAD API KEY ===
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ API Key OpenAI tidak ditemukan!")
    st.stop()

# === INISIALISASI MODEL CHAT ===
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4-turbo", temperature=0.7)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# === SIMPAN HISTORY CHAT ===
if "history" not in st.session_state:
    st.session_state.history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# === LAYOUT CHAT HISTORY ===
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for role, text in st.session_state.history:
    align = "flex-end" if role == "user" else "flex-start"
    bubble_class = "chat-bubble-user" if role == "user" else "chat-bubble-bot"
    st.markdown(f"""
        <div class='message-container' style='align-items: {align};'>
            <div class='chat-bubble {bubble_class}'>
                {text}
            </div>
        </div>
    """, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# === LAYOUT INPUT CHAT & UPLOAD FILE ===
st.markdown("<div class='input-container'>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 5])

with col1:
    uploaded_files = st.file_uploader("📂 Upload", type=["pdf", "txt"], accept_multiple_files=True, label_visibility="collapsed")

with col2:
    user_input = st.chat_input("Ketik pesan Anda...")
st.markdown("</div>", unsafe_allow_html=True)

# === PROSES FILE JIKA DIUNGGAH KAPAN SAJA ===
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        if file_name not in st.session_state.uploaded_files:
            st.session_state.uploaded_files.append(file_name)
            st.session_state.history.append(("user", f"📂 {file_name} telah diunggah!"))

            # Proses dan simpan file
            documents = []
            file_path = f"./temp_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if uploaded_file.type == "application/pdf":
                loader = PyPDFLoader(file_path)
            elif uploaded_file.type == "text/plain":
                loader = TextLoader(file_path)

            documents.extend(loader.load())

            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            split_docs = text_splitter.split_documents(documents)
            st.session_state.retriever = FAISS.from_documents(split_docs, OpenAIEmbeddings()).as_retriever()
            st.success(f"✅ {file_name} berhasil diproses!")

    st.rerun()

# === PROSES INPUT USER ===
if user_input:
    st.session_state.history.append(("user", user_input))

    # === PROSES RESPON AI ===
    response = "⚠️ Maaf, saya tidak dapat memberikan jawaban."

    if st.session_state.retriever:
        try:
            response_data = ConversationalRetrievalChain.from_llm(
                llm, retriever=st.session_state.retriever, memory=memory
            ).invoke({"question": user_input})
            response = response_data.get("answer", "⚠️ Tidak ada jawaban dari dokumen.")
        except Exception as e:
            response = f"⚠️ Kesalahan dalam pemrosesan file: {str(e)}"

    st.session_state.history.append(("bot", response))
    st.rerun()
