import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
import os

# **🚀 Load API Key dari Streamlit Secrets**
if "OPENAI_API_KEY" not in st.secrets:
    st.error("❌ API Key OpenAI tidak ditemukan di Secrets! Tambahkan di Streamlit Cloud.")
    st.stop()

openai_api_key = st.secrets["OPENAI_API_KEY"]

# **🚀 Inisialisasi Chatbot**
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# **🔥 UI Styling - Full Background Putih & Clean Layout**
st.markdown("""
    <style>
        body {
            background-color: #ffffff !important;
        }
        .main-title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #d32f2f;  /* WARNA MERAH */
            margin-top: 20px;
        }
        .sub-title {
            text-align: center;
            font-size: 16px;
            color: #333;
            margin-bottom: 20px;
        }
        .chat-container {
            display: flex;
            flex-direction: column-reverse;
            max-height: 65vh;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #ddd;
            background: #f8f9fa;
            border-radius: 10px;
        }
        .chat-input-container {
            display: flex;
            align-items: center;
            gap: 10px;
            width: 100%;
            max-width: 800px;
            background: #ffffff;
            border-radius: 8px;
            padding: 8px;
            margin: auto;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        }
        .chat-input {
            flex-grow: 1;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 6px;
            background: #fafafa;
            color: #333;
            width: 100%;
        }
        .upload-file-btn {
            padding: 10px;
            background: #d32f2f;
            color: white;
            border-radius: 6px;
            cursor: pointer;
            border: none;
        }
        .upload-file-btn:hover {
            background: #b71c1c;
        }
        .stChatMessage {
            border-radius: 12px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .stChatMessage.user {
            background-color: #d32f2f;
            color: white;
        }
        .stChatMessage.assistant {
            background-color: #eeeeee;
            color: black;
        }
    </style>
""", unsafe_allow_html=True)

# **📌 Tampilkan Header Mirip ChatGPT**
st.markdown('<div class="main-title">💬 Chatbot AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">How can I help you today?</div>', unsafe_allow_html=True)

# **💾 Simpan history chat di session**
if "history" not in st.session_state:
    st.session_state.history = []

# **📌 Chatbox History di Atas**
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for role, text in reversed(st.session_state.history):
    with st.chat_message(role):
        st.write(text)

st.markdown('</div>', unsafe_allow_html=True)

# **📎 Upload File di Sebelah Input Chat**
col1, col2 = st.columns([8, 2])

with col2:
    uploaded_file = st.file_uploader("Upload File", type=["pdf", "txt"], label_visibility="collapsed")

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

        # **🔹 Split Text & Simpan ke VectorStore**
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = text_splitter.split_documents(documents)
        retriever = FAISS.from_documents(split_docs, OpenAIEmbeddings()).as_retriever()

        st.success("✅ File berhasil diunggah dan diproses!")

# **🔹 Chatbot dengan Memory & Knowledge dari File (Jika Ada)**
conversation = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory) if retriever else None

# **📩 Input Chat dengan Tombol Upload**
st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
with col1:
    user_input = st.text_input("Ketik pesan Anda...", key="chat_input", label_visibility="collapsed")

st.markdown('</div>', unsafe_allow_html=True)

if user_input:
    user_input = user_input.strip()

    # Tampilkan pertanyaan user
    with st.chat_message("user"):
        st.write(user_input)

    # **Jika ada file, gunakan retriever**
    if retriever:
        response_data = conversation.invoke({"question": user_input})
        response = getattr(response_data, "answer", "⚠️ Tidak ada jawaban yang tersedia.")

    # **Jika tidak ada file, gunakan LLM biasa**
    else:
        response_data = llm.invoke(user_input)

        # **✅ Ambil hanya bagian content tanpa metadata**
        response = getattr(response_data, "content", "⚠️ Terjadi kesalahan dalam mendapatkan jawaban.")

    # **Tampilkan jawaban chatbot**
    with st.chat_message("assistant"):
        st.write(response)

    # **Simpan history chat**
    st.session_state.history.insert(0, ("user", user_input))
    st.session_state.history.insert(0, ("assistant", response))
