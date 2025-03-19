import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

from langchain_community.document_loaders import PyPDFLoader, TextLoader

from langchain.text_splitter import CharacterTextSplitter
import os

# Load API Key dari Streamlit Secrets
if "OPENAI_API_KEY" not in st.secrets:
    st.error("❌ API Key OpenAI tidak ditemukan di Secrets! Tambahkan di Streamlit Cloud.")
    st.stop()

openai_api_key = st.secrets["OPENAI_API_KEY"]

# Inisialisasi Chatbot
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

st.title("📂 Chatbot dengan File Upload")

import streamlit as st

st.title("📂 Chatbot dengan File Upload")

# **🚀 Tambahkan CSS untuk Sticky Header**
st.markdown("""
    <style>
        /* Membuat header tetap di atas */
        .appview-container {
            padding-top: 80px; /* Tambahkan jarak biar header tidak ketutupan */
        }
        
        .fixed-header {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            background-color: #0e1117; /* Warna latar belakang header */
            padding: 10px 0;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: white;
            z-index: 999;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
        }

        /* Menyesuaikan tampilan chatbot */
        .stChatMessage {
            border-radius: 15px;
            padding: 10px;
        }
        .stChatMessage.user {
            background-color: #DCF8C6; /* Warna hijau seperti WhatsApp */
        }
        .stChatMessage.assistant {
            background-color: #EDEDED;
        }
    </style>
""", unsafe_allow_html=True)

# **🔹 Tambahkan Header Static**
st.markdown('<div class="fixed-header">🤖 Chatbot dengan File Upload</div>', unsafe_allow_html=True)


st.write("Upload file dan ajukan pertanyaan tentang isi file!")

# **🔹 Fitur Upload File**
uploaded_file = st.file_uploader("Upload file (PDF, TXT)", type=["pdf", "txt"])

retriever = None  # Placeholder untuk retriever

if uploaded_file:
    with st.spinner("📖 Memproses file..."):
        file_path = f"./temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # **🔹 Load File Sesuai Format**
        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(file_path)
        elif uploaded_file.type == "text/plain":
            loader = TextLoader(file_path)

        # **🔹 Split Text dan Simpan ke VectorStore**
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = text_splitter.split_documents(documents)

        retriever = FAISS.from_documents(split_docs, OpenAIEmbeddings()).as_retriever()
        st.success("✅ File berhasil diunggah dan diproses!")

# **🔹 Chatbot dengan Memory & Knowledge dari File**
conversation = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory) if retriever else None

# Simpan history chat di session
if "history" not in st.session_state:
    st.session_state.history = []

# **🔹 Tampilkan Chat History**
for role, text in st.session_state.history:
    with st.chat_message(role):
        st.write(text)

# **🔹 Input Chat di Bawah**
user_input = st.chat_input("Ketik pertanyaan Anda...")

if user_input:
    # Tampilkan pertanyaan user
    with st.chat_message("user"):
        st.write(user_input)

    # **Jika tidak ada file, gunakan LLM biasa**
    if not conversation:
        response = "Silakan upload file dulu sebelum bertanya!"
    else:
        response = conversation.invoke({"question": user_input})["answer"]

    # Tampilkan jawaban chatbot
    with st.chat_message("assistant"):
        st.write(response)

    # Simpan history chat
    st.session_state.history.append(("user", user_input))
    st.session_state.history.append(("assistant", response))
