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

# === 🔹 KONFIGURASI UTAMA ===
st.set_page_config(page_title="Chatbot AI", page_icon="💬", layout="wide")

# === 🔥 TAMPILAN UI CUSTOM ===
st.markdown(
    """
    <style>
        body {
            background-color: #ffffff;
            font-family: 'Arial', sans-serif;
        }
        .css-18e3th9 {
            padding-top: 20px;
        }
        .chat-bubble {
            padding: 12px;
            margin: 5px;
            border-radius: 10px;
            max-width: 80%;
        }
        .chat-bubble-user {
            background-color: #dcf8c6;
            align-self: flex-end;
        }
        .chat-bubble-bot {
            background-color: #f1f0f0;
        }
        .message-container {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# === 🚀 LOAD API KEY ===
if "OPENAI_API_KEY" not in st.secrets:
    st.error("❌ API Key OpenAI tidak ditemukan di Secrets! Tambahkan di Streamlit Cloud.")
    st.stop()

openai_api_key = st.secrets["OPENAI_API_KEY"]

# === 🚀 INISIALISASI MODEL CHAT ===
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# === 💾 SIMPAN HISTORY CHAT ===
if "history" not in st.session_state:
    st.session_state.history = []

# === 🔹 PILIHAN MODE INTERAKSI ===
mode = st.radio("Pilih Mode Interaksi:", ["Chat", "Upload File"], horizontal=True)

# === 📂 FITUR UPLOAD FILE ===
retriever = None
if mode == "Upload File":
    uploaded_file = st.file_uploader("Unggah file (PDF, TXT)", type=["pdf", "txt"])
    if uploaded_file:
        with st.spinner("📖 Memproses file..."):
            file_path = f"./temp_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if uploaded_file.type == "application/pdf":
                loader = PyPDFLoader(file_path)
            elif uploaded_file.type == "text/plain":
                loader = TextLoader(file_path)

            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            split_docs = text_splitter.split_documents(documents)
            retriever = FAISS.from_documents(split_docs, OpenAIEmbeddings()).as_retriever()
            st.success("✅ File berhasil diunggah dan diproses!")

# === 💬 MENAMPILKAN HISTORY CHAT ===
st.markdown("<h3 style='text-align: center;'>💬 Chatbot AI</h3>", unsafe_allow_html=True)
for role, text in st.session_state.history:
    align = "flex-end" if role == "user" else "flex-start"
    bubble_class = "chat-bubble-user" if role == "user" else "chat-bubble-bot"
    st.markdown(
        f"""
        <div class='message-container' style='align-items: {align};'>
            <div class='chat-bubble {bubble_class}'>
                {text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# === 📝 INPUT CHAT ===
user_input = st.chat_input("Ketik pesan Anda...")

# === 🤖 LOGIKA CHATBOT ===

if user_input:
    user_input = user_input.strip()

    # 📝 Langsung tampilkan pesan pengguna di UI sebelum chatbot merespons
    with st.chat_message("user"):
        st.write(user_input)

    # Simpan ke chat history
    st.session_state.history.append(("user", user_input))

    # === 🔍 Jika pengguna ingin cari di internet ===
    if "cari di internet" in user_input.lower():
        response = "🔍 [Pencarian internet belum tersedia di versi ini]"

    # === 🗂 Jika ada file diunggah, gunakan retriever ===
    elif retriever:
        try:
            response_data = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory).invoke(
                {"question": user_input}
            )
            response = response_data.get("answer", "⚠️ Tidak ada jawaban yang tersedia.")
        except Exception as e:
            response = f"⚠️ Terjadi kesalahan dalam pemrosesan file: {str(e)}"

    # === 💡 Jika hanya chat biasa ===
    else:
        try:
            response_data = llm.invoke(user_input)

            if isinstance(response_data, str):
                response = response_data  # Jika langsung string
            elif hasattr(response_data, "content"):
                response = response_data.content  # Jika objek AIMessage
            else:
                response = "⚠️ Tidak ada jawaban yang tersedia."

        except Exception as e:
            response = f"⚠️ Terjadi kesalahan dalam memproses pertanyaan: {str(e)}"

    # 💡 Pastikan tidak ada respons kosong
    if not response or not response.strip():
        response = "⚠️ Terjadi kesalahan dalam mendapatkan jawaban."

    # Simpan respons chatbot ke chat history
    st.session_state.history.append(("bot", response))

    # ✅ Tampilkan respons chatbot di UI
    with st.chat_message("assistant"):
        st.write(response)
