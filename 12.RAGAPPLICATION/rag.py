from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


def extract_video_id(url):
    if "watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]

    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]

    return url

from youtube_transcript_api import YouTubeTranscriptApi


def get_transcript(video_id):
    api = YouTubeTranscriptApi()

    transcript = api.fetch(video_id)

    text = " ".join(snippet.text for snippet in transcript)

    return text


def build_chain(video_url):

    video_id = extract_video_id(video_url)

    transcript = get_transcript(video_id)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.create_documents([transcript])

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k":4}
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2
    )

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.

Answer ONLY using the transcript.

If answer is unavailable say
"I don't know."

Context:
{context}

Question:
{question}
""",
        input_variables=["context","question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnableParallel(
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain