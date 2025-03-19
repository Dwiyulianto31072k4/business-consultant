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
            background: linear-gradient(135deg, #6e3ff2, #c471ed);
            font-family: 'Arial', sans-serif;
            color: white;
            margin: 0;
            padding: 0;
        }
        .chat-container {
            width: 100%;
            max-width: 600px;
            margin: auto;
            padding: 10px;
        }
        .chat-bubble {
            padding: 15px;
            margin: 8px;
            border-radius: 20px;
            max-width: 80%;
            font-size: 14px;
            display: inline-block;
            animation: fadeIn 0.4s ease-in-out;
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
        .message-container {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            margin-bottom: 20px;
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

# === MENAMPILKAN HISTORY CHAT ===
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

# === TOMBOL UPLOAD FILE DI SEBELAH INPUT CHAT ===
col1, col2 = st.columns([1, 5])  # Kolom untuk Upload dan Input Chat
with col1:
    uploaded_files = st.file_uploader("📂", type=["pdf", "txt"], accept_multiple_files=True, label_visibility="collapsed")

# === PROSES FILE JIKA DIUNGGAH ===
if uploaded_files:
    st.success("📂 File berhasil diunggah! Chatbot akan menggunakan dokumen jika dibutuhkan.")
    documents = []
    
    for uploaded_file in uploaded_files:
        with st.spinner(f"📖 Memproses {uploaded_file.name}..."):
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
    st.success("✅ Semua file berhasil diproses!")

# === INPUT CHAT ===
with col2:
    user_input = st.chat_input("Ketik pesan Anda...")

# === LOGIKA CHATBOT ===
if user_input:
    user_input = user_input.strip()
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.history.append(("user", user_input))

    # === CEK PENCARIAN INTERNET ===
    if "cari di internet" in user_input.lower():
        try:
            query = user_input.replace("cari di internet", "").strip()
            search_url = f"https://www.google.com/search?q={query}"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(search_url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")
            results = soup.find_all("h3")
            response = "\n".join([res.get_text() for res in results[:5]]) or "⚠️ Tidak ada hasil pencarian yang ditemukan."
        except Exception as e:
            response = f"⚠️ Gagal mengambil data internet: {str(e)}"

    # === CEK ANALISIS DOKUMEN ===
    elif st.session_state.retriever:
        try:
            response_data = ConversationalRetrievalChain.from_llm(
                llm, retriever=st.session_state.retriever, memory=memory
            ).invoke({"question": user_input})
            response = response_data.get("answer", "⚠️ Tidak ada jawaban yang tersedia dari dokumen.")
        except Exception as e:
            response = f"⚠️ Terjadi kesalahan dalam pemrosesan file: {str(e)}"

    # === JIKA CHAT BIASA ===
    else:
        try:
            response_data = llm.invoke(user_input)
            response = response_data if isinstance(response_data, str) else response_data.content
        except Exception as e:
            response = f"⚠️ Terjadi kesalahan dalam memproses pertanyaan: {str(e)}"

    response = response if response.strip() else "⚠️ Tidak ada jawaban."
    st.session_state.history.append(("bot", response))
    with st.chat_message("assistant"):
        st.write(response)
