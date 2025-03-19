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

# **🚀 Load API Key from Streamlit Secrets**
if "OPENAI_API_KEY" not in st.secrets:
    st.error("❌ API Key OpenAI not found in Secrets! Add it in Streamlit Cloud.")
    st.stop()

openai_api_key = st.secrets["OPENAI_API_KEY"]

# **🚀 Initialize Chatbot**
llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# **🔹 UI Styling**
st.markdown(
    """
    <style>
        body {
            background-color: white;
        }
        .chat-container {
            max-width: 800px;
            margin: auto;
        }
        .chat-header {
            text-align: center;
            font-size: 26px;
            font-weight: bold;
            color: #D90429;
            margin-bottom: 10px;
        }
        .chat-box {
            border-radius: 10px;
            padding: 10px;
            background-color: #f8f9fa;
            margin-bottom: 10px;
        }
        .chat-input-container {
            display: flex;
            justify-content: space-between;
            padding: 10px;
        }
        .chat-input {
            width: 100%;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #ccc;
        }
        .upload-container {
            display: flex;
            justify-content: flex-end;
            margin-top: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# **🔹 Chatbot UI Layout**
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
st.markdown('<div class="chat-header">\U0001F4AC Chatbot AI</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>How can I help you today?</p>", unsafe_allow_html=True)

# **🔹 Display Chat History**
if "history" not in st.session_state:
    st.session_state.history = []

for role, text in st.session_state.history:
    if role == "user":
        st.markdown(f'<div class="chat-box" style="background:#ffe5e5;">\U0001F464 {text}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-box" style="background:#ffffcc;">\U0001F4DA {text}</div>', unsafe_allow_html=True)

# **🔹 User Input Section**
user_input = st.text_input("", placeholder="Type your message here...", key="chat_input", label_visibility="collapsed")

# **🔹 File Upload Section (Right-Aligned)**
col1, col2 = st.columns([4, 1])
with col2:
    uploaded_file = st.file_uploader("Upload", type=["pdf", "txt"], label_visibility="collapsed")

# **🔹 Process Uploaded File**
retriever = None
if uploaded_file:
    file_path = f"./temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    loader = PyPDFLoader(file_path) if uploaded_file.type == "application/pdf" else TextLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    retriever = FAISS.from_documents(split_docs, OpenAIEmbeddings()).as_retriever()
    st.success("✅ File uploaded successfully!")

# **🔹 Process User Input**
if user_input:
    st.session_state.history.append(("user", user_input))
    
    if retriever:
        response_data = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory).invoke({"question": user_input})
        response = response_data.get("answer", "⚠️ No answer available.")
    else:
        response_data = llm.invoke(user_input)
        response = response_data.get("content", "⚠️ No answer available.")
    
    st.session_state.history.append(("assistant", response))
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
