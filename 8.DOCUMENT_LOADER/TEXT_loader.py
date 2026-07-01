from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()

model=ChatOpenAI()

prompt = PromptTemplate(
    template="Write a summary for the following - \n{poem}",
    input_variables=["poem"]
)

parse=StrOutputParser()

loader=TextLoader(
  'chat_history.txt'
)

docs=loader.load()

print(docs[0])

chain = prompt|model|parse

print(chain.invoke({'poem':docs[0].page_content}))
