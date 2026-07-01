from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
chat_history=[]

model = ChatOpenAI()

while True:
    user_input = input("you : ")
    chat_history.append(user_input)

    if user_input == "exit":
        break

    result = model.invoke(chat_history)
    chat_history.append(result)

    print("AI :", result.content)