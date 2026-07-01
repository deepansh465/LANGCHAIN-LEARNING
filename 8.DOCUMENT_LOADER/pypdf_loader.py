from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("8.DOCUMENT_LOADER/data.pdf")
docs = loader.load()

print(docs[0].page_content)
print(len(docs))
print(docs[1].metadata)
print(docs[1])