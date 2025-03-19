import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

# Load API Key dari Streamlit Secrets
if "OPENAI_API_KEY" not in st.secrets:
    st.error("❌ API Key OpenAI tidak ditemukan di Secrets! Tambahkan di Streamlit Cloud.")
    st.stop()

openai_api_key = st.secrets["OPENAI_API_KEY"]

# Inisialisasi Chatbot
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4")
retriever = FAISS.from_texts(["Halo! Ada yang bisa saya bantu?"], OpenAIEmbeddings()).as_retriever()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)

# UI Streamlit - Desain Chatbot
st.title("🤖 Chatbot LLM dengan Memory")
st.write("Chatbot ini bisa mengingat percakapan sebelumnya.")

# Simpan history chat di session
if "history" not in st.session_state:
    st.session_state.history = []

# **🔹 Tampilkan Chat History dalam Format Chat**
for role, text in st.session_state.history:
    with st.chat_message(role):
        st.write(text)

# **🔹 Input User Pindah ke Bawah (ChatGPT-style)**
user_input = st.chat_input("Ketik pesan Anda...")

if user_input:
    # Tampilkan pesan user
    with st.chat_message("user"):
        st.write(user_input)

    # Proses chatbot
    response = conversation.invoke({"question": user_input})["answer"]

    # Tampilkan jawaban chatbot
    with st.chat_message("assistant"):
        st.write(response)

    # Simpan history ke session
    st.session_state.history.append(("user", user_input))
    st.session_state.history.append(("assistant", response))
