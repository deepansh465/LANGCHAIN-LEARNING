from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model=ChatAnthropic(model='claude-haiku-4-5-20251001')
result=model.invoke("what is capital of delhi")

print(result)
print(result.content)