from dotenv import load_dotenv

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser

import os

# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------

load_dotenv()

# --------------------------------------------------
# Configuration
# --------------------------------------------------

VIDEO_ID = "Gfr50f6ZBvo"
INDEX_PATH = "faiss_index"

# --------------------------------------------------
# Get Transcript
# --------------------------------------------------

def get_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(
            video_id,
            languages=["en"],
        )

        transcript = " ".join(
            chunk["text"] for chunk in transcript_list
        )

        return transcript

    except TranscriptsDisabled:
        print("This video has no transcript.")
        exit()

    except Exception as e:
        print(e)
        exit()


# --------------------------------------------------
# Build / Load Vector Store
# --------------------------------------------------

def build_vector_store(transcript):

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    if os.path.exists(INDEX_PATH):
        print("Loading existing FAISS index...")

        return FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    print("Creating new FAISS index...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    docs = splitter.create_documents([transcript])

    print(f"Created {len(docs)} chunks")

    vector_store = FAISS.from_documents(
        docs,
        embeddings,
    )

    vector_store.save_local(INDEX_PATH)

    print("FAISS Index Saved")

    return vector_store


# --------------------------------------------------
# Prompt
# --------------------------------------------------

prompt = PromptTemplate(
    template="""
You are a helpful AI assistant.

Answer ONLY from the transcript provided.

If the answer cannot be found in the transcript,
reply only:

"I don't know."

Transcript:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"],
)

# --------------------------------------------------
# Helper
# --------------------------------------------------

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("Fetching transcript...")

    transcript = get_transcript(VIDEO_ID)

    print("Transcript Loaded\n")

    vector_store = build_vector_store(transcript)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k":4},
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
    )

    chain = (
        RunnableParallel(
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n===============================")
    print(" YouTube RAG Ready!")
    print("===============================")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("You : ")

        if question.lower() in ["exit", "quit"]:
            break

        try:
            answer = chain.invoke(question)

            print("\nAssistant:\n")
            print(answer)

        except Exception as e:
            print(e)

        print("\n" + "-" * 70 + "\n")


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":
    main()