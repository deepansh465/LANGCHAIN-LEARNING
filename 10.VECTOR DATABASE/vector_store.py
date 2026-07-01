from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

# Documents
docs = [
    Document(
        page_content="LangChain ek framework hai LLM apps banane ke liye.",
        metadata={"id": 1},
    ),
    Document(
        page_content="Chroma ek vector database hai.",
        metadata={"id": 2},
    ),
]

# Create/Open Chroma DB
vector_store = Chroma(
    collection_name="sample",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="my_chroma_db",
)

# Add documents
vector_store.add_documents(docs)

# Retrieve everything
data = vector_store.get(
    include=["documents", "metadatas", "embeddings"]
)

print(data)