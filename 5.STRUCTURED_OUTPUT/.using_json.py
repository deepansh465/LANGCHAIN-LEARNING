from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize model
model = ChatOpenAI()

# JSON Schema
review_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "Write down all the key themes discussed in the review"
        },
        "summary": {
            "type": "string",
            "description": "A brief summary of the review"
        },
        "sentiment": {
            "type": "string",
            "enum": ["pos", "neg"],
            "description": "Return sentiment of the review either positive or negative"
        },
        "pros": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "Write down all the pros inside a list"
        },
        "cons": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "Write down all the cons inside a list"
        },
        "name": {
            "type": "string",
            "description": "Write the name of the reviewer"
        }
    },
    "required": [
        "key_themes",
        "summary",
        "sentiment"
    ]
}

# Create structured-output model
structured_model = model.with_structured_output(review_schema)

# Invoke model
result = structured_model.invoke("""
The hardware is great, but the software feels bloated.
There are too many pre-installed apps that I can't remove.
Also, the UI looks outdated compared to other brands.
Hoping for a software update to fix this. my name is deepansh 
""")

# Result is a dictionary
print(result)

print("\nSummary:")
print(result["summary"])

print("\nSentiment:")
print(result["sentiment"])

print("\nKey Themes:")
print(result["key_themes"])

print("\nPros:")
print(result.get("pros"))

print("\nCons:")
print(result.get("cons"))

print("\nReviewer Name:")
print(result.get("name"))