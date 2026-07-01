from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

docs = [
    "Delhi is capital of India",
    "Mumbai is financial capital of India",
    "Chandigarh is capital of Punjab and Haryana"
]

result = embedding.embed_documents(docs)

print(str(result)) 