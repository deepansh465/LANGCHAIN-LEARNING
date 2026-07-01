from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated
load_dotenv()


model=ChatOpenAI()

class review(TypedDict) :
  summary: Annotated[str,"breif summary of review"]                 #isse bs btaata hai ki yeh str me hina chahiye alag int wagera me bhi h
  sentiments: str                                                     #hoga toh chalega bascially struture gv

                                                      #annotated isliye taki jb bhi jayega toh usse pta chal jaye what is asking js making sure although it is already doing that bs confuse na ho isliye
structured_model=model.with_structured_output(review)  #yeh function hai typedict ka


result = structured_model.invoke("""
The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.
""")

print(result)
print(result['summary']) # showing dict hai toh yeh alag se bhi print kr skte hum