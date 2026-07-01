import streamlit as st
from dotenv import load_dotenv

from rag import build_chain
load_dotenv()

st.set_page_config(
    page_title="YouTube RAG",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 YouTube RAG Chatbot")

st.write(
    "Paste a YouTube video link and chat with the transcript."
)

if "chain" not in st.session_state:
    st.session_state.chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

video_url = st.text_input(
    "YouTube URL"
)

col1, col2 = st.columns([1,4])

with col1:

    if st.button("Process Video"):

        if video_url == "":
            st.warning("Enter a YouTube URL")

        else:

            with st.spinner("Processing video..."):

                st.session_state.chain = build_chain(video_url)

            st.success("Video Ready!")

st.divider()

for role, message in st.session_state.messages:

    with st.chat_message(role):
        st.write(message)

if st.session_state.chain:

    question = st.chat_input("Ask anything about the video...")

    if question:

        st.session_state.messages.append(("user", question))

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                answer = st.session_state.chain.invoke(question)

                st.write(answer)

        st.session_state.messages.append(("assistant", answer))