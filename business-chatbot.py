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
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False)

# **🔥 UI Streamlit - Sticky Header & CSS Custom**
st.markdown("""
    <style>
        .fixed-header {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            background-color: #0e1117;
            padding: 15px 0;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: white;
            z-index: 999;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
        }
        .appview-container {
            padding-top: 80px !important;
        }
    </style>
""", unsafe_allow_html=True)

# **🚀 Tambahkan Header Static**
st.markdown('<div class="fixed-header">🤖 Chatbot - Tanya Langsung atau Upload File</div>', unsafe_allow_html=True)

# **🔹 Pilihan Mode**
mode = st.radio("Pilih mode interaksi:", ["Tanya Langsung", "Upload File"])

retriever = None  # Placeholder untuk retriever

if mode == "Upload File":
    uploaded_file = st.file_uploader("Upload file (PDF, TXT)", type=["pdf", "txt"])
    
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

# **🔹 Chatbot dengan Memory & Knowledge dari File (Jika Ada)**
conversation = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory) if retriever else None

# **🔍 Fungsi Web Search dengan SerpAPI**
def search_web(query, num_results=5):
    """Melakukan pencarian di Google menggunakan SerpAPI"""
    if "SERP_API_KEY" not in st.secrets:
        return "❌ API Key SerpAPI tidak ditemukan di Secrets!"

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
        return "\n".join([f"🔍 {res['title']} - {res['link']}\n{res.get('snippet', '')}" for res in results[:5]])
    else:
        return "❌ Tidak ada hasil pencarian untuk kata kunci ini."

# **💾 Simpan history chat di session**
if "history" not in st.session_state:
    st.session_state.history = []

# **🔹 Tampilkan Chat History**
for role, text in st.session_state.history:
    with st.chat_message(role):
        st.write(text)

# **🔹 Input Chat di Bawah**
user_input = st.chat_input("Ketik pertanyaan Anda...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    # **🔍 Deteksi apakah butuh Web Search**
    if "cari di internet" in user_input.lower():
        query = user_input.replace("cari di internet", "").strip()
        response = search_web(query)

    # **Jika ada file yang diunggah, gunakan retriever**
    elif retriever:
        response_data = conversation.invoke({"question": user_input})
        response = response_data.get("answer", "⚠️ Terjadi kesalahan dalam mendapatkan jawaban.")

    # **Jika tidak ada file & bukan Web Search, gunakan LLM biasa**
# **Jika tidak ada file & bukan Web Search, gunakan LLM biasa**
else:
    response_data = llm.invoke(user_input)

    # **🔹 FIX: Ambil hanya teks jawaban dari "content" tanpa metadata**
    if isinstance(response_data, dict) and "content" in response_data:
        response = response_data["content"]  # Hanya menampilkan teks jawaban
    else:
        response = str(response_data)  # Jika tidak dalam format dict, ubah ke string

# **Tampilkan jawaban chatbot**
with st.chat_message("assistant"):
    st.write(response)

# **Simpan history chat**
st.session_state.history.append(("user", user_input))
st.session_state.history.append(("assistant", response))

