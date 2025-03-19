import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# API Key
openai_api_key = "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"

# Inisialisasi Chatbot
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation = ConversationalRetrievalChain.from_llm(llm, memory=memory)

# UI Streamlit
st.title("Chatbot LLM dengan Memory")
st.write("Chatbot ini bisa mengingat percakapan sebelumnya.")

# Simpan history percakapan di session
if "history" not in st.session_state:
    st.session_state.history = []

# Input user
user_input = st.text_input("Anda:", "")

if user_input:
    response = conversation.invoke({"question": user_input})
    st.session_state.history.append(("Anda", user_input))
    st.session_state.history.append(("Bot", response["answer"]))

# Tampilkan percakapan
for role, text in st.session_state.history:
    st.write(f"**{role}:** {text}")

