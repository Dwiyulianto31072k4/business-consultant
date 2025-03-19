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

# **🚀 Load API Key dari Streamlit Secrets**
if "OPENAI_API_KEY" not in st.secrets:
    st.error("❌ API Key OpenAI tidak ditemukan di Secrets! Tambahkan di Streamlit Cloud.")
    st.stop()

openai_api_key = st.secrets["OPENAI_API_KEY"]

# **🚀 Inisialisasi Chatbot**
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# **🔍 Fungsi Web Search dengan SerpAPI**
def search_web(query, num_results=5):
    """Melakukan pencarian di Google menggunakan SerpAPI"""
    if "SERP_API_KEY" not in st.secrets:
        return ["❌ API Key SerpAPI tidak ditemukan di Secrets!"]

    api_key = st.secrets["SERP_API_KEY"]
    
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": api_key,
        "num": num_results,
        "engine": "google"
    }

    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        results = response.json().get("organic_results", [])
        return [f"🔍 [{res['title']}]({res['link']}) - {res.get('snippet', '')}" for res in results]
    else:
        return ["❌ Tidak ada hasil pencarian."]

# **💾 Simpan history chat di session**
if "history" not in st.session_state:
    st.session_state.history = []

# **🔹 Pilihan Mode**
mode = st.radio("Pilih mode interaksi:", ["Tanya Langsung", "Upload File"])

retriever = None  # Placeholder untuk retriever

if mode == "Upload File":
    # **🔹 Fitur Upload File**
    uploaded_file = st.file_uploader("Upload file (PDF, TXT)", type=["pdf", "txt"])
    
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

# **🔹 Tampilkan Chat History**
for role, text in st.session_state.history:
    with st.chat_message(role):
        st.write(text)

# **🔹 Input Chat di Bawah**
user_input = st.chat_input("Ketik pertanyaan Anda...")

if user_input:
    user_input = user_input.strip()  # Hapus spasi ekstra

    # Tampilkan pertanyaan user
    with st.chat_message("user"):
        st.write(user_input)

    # **🔍 Deteksi apakah butuh Web Search**
    if "cari di internet" in user_input.lower():
        query = user_input.replace("cari di internet", "").strip()
        search_results = search_web(query)

        response = "\n\n".join(search_results)

    # **Jika ada file yang diunggah, gunakan retriever**
    elif retriever:
        response_data = conversation.invoke({"question": user_input})

        # **🔹 FIX: Ambil hanya teks jawaban dari "answer" tanpa metadata**
        response = response_data.get("answer", "⚠️ Tidak ada jawaban yang tersedia.")

    # **Jika tidak ada file & bukan Web Search, gunakan LLM biasa**
    else:
        response_data = llm.invoke(user_input)

        # **🔹 FIX: Ambil hanya teks jawaban dari "content" tanpa metadata**
        response = response_data.get("content", "⚠️ Tidak ada jawaban yang tersedia.")

# **Cegah output `None` atau kosong**
if not response.strip():
    response = "⚠️ Terjadi kesalahan dalam mendapatkan jawaban."

# **Tampilkan jawaban chatbot**
with st.chat_message("assistant"):
    st.write(response)

# **Simpan history chat**
st.session_state.history.append(("user", user_input))
st.session_state.history.append(("assistant", response))
