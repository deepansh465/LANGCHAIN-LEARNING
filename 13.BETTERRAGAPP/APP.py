import streamlit as st
from dotenv import load_dotenv

from rag import build_chain

load_dotenv()

st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="wide",
)

# ---------------- Sidebar ---------------- #

with st.sidebar:

    st.title("📚 Project")

    st.markdown("""
### Tech Stack

- LangChain
- OpenAI
- GPT-4o Mini
- FAISS
- Streamlit
- YouTube Transcript API
""")

    st.divider()

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------- Session State ---------------- #

if "chain" not in st.session_state:
    st.session_state.chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- Main UI ---------------- #

st.title("🎥 YouTube RAG Chatbot")

st.write(
    "Paste any YouTube video URL and chat with its transcript."
)

video_url = st.text_input(
    "YouTube URL",
    placeholder="https://www.youtube.com/watch?v=..."
)

if video_url:
    st.video(video_url)

if st.button("🚀 Process Video"):

    if video_url == "":
        st.warning("Please enter a YouTube URL.")

    elif ("youtube.com" not in video_url) and ("youtu.be" not in video_url):
        st.error("Please enter a valid YouTube URL.")

    else:

        try:

            with st.spinner("Fetching transcript and creating embeddings..."):

                st.session_state.chain = build_chain(video_url)

            st.success("✅ Video indexed successfully!")

        except Exception as e:

            st.error(str(e))

st.divider()

for role, message in st.session_state.messages:

    with st.chat_message(role):

        st.markdown(message)

if st.session_state.chain:

    question = st.chat_input(
        "Ask anything about this video..."
    )

    if question:

        st.session_state.messages.append(
            ("user", question)
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                answer = st.session_state.chain.invoke(question)

                st.markdown(answer)

        st.session_state.messages.append(
            ("assistant", answer)
        )