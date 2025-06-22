import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Groq API credentials
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-70b-8192"  # You can also try llama3-8b-8192

def analyze_sentiment_intent(text):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    # Few-shot system prompt to guide Groq model better:
    system_prompt = """
You are a medical assistant analyzing doctor-patient conversations.

You must classify:

1️⃣ Sentiment:  
- "Anxious" (if the patient shows worry, fear, uncertainty, or anxiety)
- "Neutral" (if the patient speaks factually without emotional cues)
- "Reassured" (if the patient feels positive, hopeful, or confident)

2️⃣ Intent:
- "Seeking reassurance"
- "Reporting symptoms"
- "Expressing concern"
- "General inquiry"
- "Gratitude"

Output strict valid JSON like:
{"Sentiment":"", "Intent":""}

### Examples:

Example 1:
Patient says: "I'm a bit worried about my back pain, but I hope it gets better soon."
Output: {"Sentiment":"Anxious", "Intent":"Seeking reassurance"}

Example 2:
Patient says: "I've been coughing for the last 3 days and have a mild fever."
Output: {"Sentiment":"Neutral", "Intent":"Reporting symptoms"}

Example 3:
Patient says: "Thank you so much, doctor."
Output: {"Sentiment":"Reassured", "Intent":"Gratitude"}

Example 4:
Patient says: "Should I take this medication before or after food?"
Output: {"Sentiment":"Neutral", "Intent":"General inquiry"}

Now analyze the following conversation.
"""

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        "temperature": 0.0
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        try:
            model_output = response.json()["choices"][0]["message"]["content"].strip()
            # Strip accidental code block formatting
            if model_output.startswith("```json"):
                model_output = model_output.replace("```json", "").replace("```", "").strip()
            result = json.loads(model_output)
        except Exception as e:
            print("JSON parsing failed:", e)
            print("Model Output:", model_output)
            result = {"Sentiment": "", "Intent": ""}
    else:
        print("API Error:", response.text)
        result = {"Sentiment": "", "Intent": ""}

    return result
