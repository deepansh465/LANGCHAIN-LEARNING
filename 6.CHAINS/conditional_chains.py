from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
load_dotenv()
# 1. Initialize Model and String Parser
model = ChatOpenAI()
str_parser = StrOutputParser()

# 2. Define Pydantic Schema for Consistent Classification
# This ensures the LLM returns only 'positive' or 'negative' [4, 5]         
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the feedback")

# 3. Setup Classification Prompt and Chain
# The format instructions are injected so the LLM follows the Pydantic schema [5]
pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negative.\n{format_instructions}\nFeedback: {feedback}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": pydantic_parser.get_format_instructions()}
)

classification_chain = prompt1 | model | pydantic_parser

# 4. Define Response Prompts for the Branches [6]
prompt2 = PromptTemplate(template="Write an appropriate response to this positive feedback: {feedback}", input_variables=["feedback"])
prompt3 = PromptTemplate(template="Write an appropriate response to this negative feedback: {feedback}", input_variables=["feedback"])

# 5. Create the Branch Logic using RunnableBranch
# It acts like an if-else statement: (condition, chain_to_run) [6, 7]
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | str_parser),
    (lambda x: x.sentiment == "negative", prompt3 | model | str_parser),
    # Default case: uses RunnableLambda to convert a standard function into a chain [8, 9]
    RunnableLambda(lambda x: "Could not find sentiment.")
)

# 6. Combine into the Final Chain
# The output of classification flows directly into the branching logic [3]
final_chain = classification_chain | branch_chain

# 7. Execute the Application
feedback_text = "This is a wonderful smartphone!"
result = final_chain.invoke({"feedback": feedback_text})
print(result)                                      