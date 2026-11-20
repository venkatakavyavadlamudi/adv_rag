
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.rag_pipeline import rag_chain


# Page Configuration
st.set_page_config(
    page_title="Conversational RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)


# Custom CSS
st.markdown("""
<style>
.stApp {
    background-color: #F5F7FB;
}

.main-title {
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    color: #1E3A8A;
}

.subtitle {
    text-align: center;
    color: #64748B;
    font-size: 15px;
}

</style>
""", unsafe_allow_html=True)


# Session Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Header
st.markdown(
    '<p class="main-title">🤖 Conversational RAG Chatbot</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Ask questions from your document knowledge base</p>',
    unsafe_allow_html=True
)

st.divider()


# Clear Chat Button
col1, col2, col3 = st.columns([3, 1, 1])

with col3:
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


# Welcome Message
if not st.session_state.chat_history:
    st.info("👋 Hello! How can I help you today?")


# Display Chat History
for message in st.session_state.chat_history:

    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)

    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)


# Chat Input
question = st.chat_input("Ask your question here...")


# Process Question
if question:

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                answer = rag_chain.invoke({
                    "input": question,
                    "chat_history": st.session_state.chat_history[-10:]
                })

                st.markdown(answer)

                st.session_state.chat_history.append(
                    HumanMessage(content=question)
                )

                st.session_state.chat_history.append(
                    AIMessage(content=answer)
                )

            except Exception as e:
                st.error(f"Error: {e}")
