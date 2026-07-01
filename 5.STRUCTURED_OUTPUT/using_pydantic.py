from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI()


class Review(BaseModel):
    key_themes: list[str] = Field(
        description="Write down all the key themes discussed in the review in a list"
    )

    summary: str = Field(
        description="A brief summary of the review"
    )

    sentiment: Literal["pos", "neg"] = Field(
        description="Return sentiment of the review either positive or negative"
    )

    pros: Optional[list[str]] = Field(
        default=None,
        description="Write down all the pros inside a list"
    )

    cons: Optional[list[str]] = Field(
        default=None,
        description="Write down all the cons inside a list"
    )

    name: Optional[str] = Field(
        default=None,
        description="Write the name of the reviewer"
    )


# Create a structured-output version of the model
structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""
The hardware is great, but the software feels bloated.
There are too many pre-installed apps that I can't remove.
Also, the UI looks outdated compared to other brands.
Hoping for a software update to fix this.
""")

print(result)

# Since result is a Pydantic object, use dot notation
print(result.summary)

# Other examples
print(result.key_themes)
print(result.sentiment)
print(result.pros)
print(result.cons)