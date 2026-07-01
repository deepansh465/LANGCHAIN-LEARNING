from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()

model=ChatOpenAI()

prompt1=PromptTemplate(
    template='tell me joke about {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='explain the joke  -{text}',
    input_variables=['text']
)



parse=StrOutputParser()

chain=RunnableSequence(prompt1,model,parse,prompt2,model,parse)

result=chain.invoke({'jee-Joint Entrenece Test'})
print(result)