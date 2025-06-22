import os
import requests
import json
from dotenv import load_dotenv
load_dotenv()

from transformers import AutoTokenizer, AutoModel

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-8b-8192" 


def analyze_sentiment_intent(text):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a medical assistant. Analyze the doctor-patient conversation "
                    "and classify the patient's sentiment and intent."
                )
            },
            {"role": "user", "content": text}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "classify_sentiment_intent",
                    "description": "Classify patient sentiment and intent.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "Sentiment": {
                                "type": "string",
                                "enum": ["Anxious", "Neutral", "Reassured"]
                            },
                            "Intent": {
                                "type": "string",
                                "enum": [
                                    "Seeking reassurance",
                                    "Reporting symptoms",
                                    "Expressing concern",
                                    "General inquiry",
                                    "Gratitude"
                                ]
                            }
                        },
                        "required": ["Sentiment", "Intent"]
                    }
                }
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "classify_sentiment_intent"}},
        "temperature": 0.0
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers, json=payload
    )

    if response.status_code == 200:
        try:
            function_args = response.json()["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            result = json.loads(function_args)
        except Exception as e:
            print("Parsing failed:", e)
            result = empty_sentiment_structure()
    else:
        print("API Error:", response.text)
        result = empty_sentiment_structure()

    return result

def empty_sentiment_structure():
    return {
        "Sentiment": "",
        "Intent": ""
    }
