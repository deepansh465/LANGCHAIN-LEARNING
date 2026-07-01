from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='8.DOCUMENT_LOADER/socialnetwork.csv')

docs = loader.load()

print(len(docs))
print(docs[1])