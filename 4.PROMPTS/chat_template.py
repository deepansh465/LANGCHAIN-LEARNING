from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([           #this is for multiline dynamic prompt
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}') #another way in langchain for SystemMessage or humanmesage
])

prompt = chat_template.invoke({
    'domain': 'cricket',
    'topic': 'Dusra'
})

print(prompt)