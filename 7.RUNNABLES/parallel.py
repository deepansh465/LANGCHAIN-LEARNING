from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel

load_dotenv()

model=ChatOpenAI()

prompt1=PromptTemplate(
  template='tell good things about {topic}',
  input_variables=['topic']
)

prompt2=PromptTemplate(
  template='tell bad things about {topic}',
  input_variables=['topic']
)

parse=StrOutputParser()

chain=RunnableParallel({
  'good':RunnableSequence(prompt1,model,parse),
  'bad':RunnableSequence(prompt2,model,parse)
})

result=chain.invoke({'topic':'langchain'})
print(result)
