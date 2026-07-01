from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from dotenv import load_dotenv

load_dotenv()
chat_history=[SystemMessage(content='you are a helpfull assistant')] #these are js for easy reading of chat history

model = ChatOpenAI()

while True:
    user_input = input("you : ")
    chat_history.append(HumanMessage(content=user_input))

    if user_input == "exit":
         break

    result = model.invoke(chat_history)
    chat_history.append(result.content)

    print("AI :", result.content)
   