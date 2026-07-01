from langchain_huggingface import HuggingFaceEmbeddings

print("START")

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("MODEL LOADED")

docs = [
    "Delhi is capital of India",
    "Mumbai is financial capital of India",
    "Chandigarh is capital of Punjab and Haryana"
]

result = embedding.embed_documents(docs)



print("EMBEDDING CREATED")
print(len(result))
print(result[:5])