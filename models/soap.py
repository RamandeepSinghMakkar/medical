import os
import requests
import json

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-8b-8192"  # or llama3-70b-8192 if you want larger model

def generate_soap_note(text):
    system_prompt = """
You are a medical assistant. Analyze the following doctor-patient conversation and generate a SOAP note.

Output strictly in valid JSON format using exactly this structure:

{
  "Subjective": {
    "Chief_Complaint": "",
    "History_of_Present_Illness": ""
  },
  "Objective": {
    "Physical_Exam": "",
    "Observations": ""
  },
  "Assessment": {
    "Diagnosis": "",
    "Severity": ""
  },
  "Plan": {
    "Treatment": "",
    "Follow-Up": ""
  }
}

RULES:
- Output ONLY valid JSON.
- Do NOT include any explanation or extra text.
- If any field is missing, leave it as empty string "".
- Use proper JSON formatting with double quotes.
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        "temperature": 0.0,
        "max_tokens": 1024
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        try:
            raw_output = response.json()["choices"][0]["message"]["content"].strip()
            # Clean any accidental formatting like code blocks
            if raw_output.startswith("```json"):
                raw_output = raw_output.replace("```json", "").replace("```", "").strip()

            result = json.loads(raw_output)
        except Exception as e:
            print("JSON parsing failed:", e)
            print("Model output:", raw_output)
            result = empty_soap_structure()
    else:
        print("API Error:", response.text)
        result = empty_soap_structure()

    return result

def empty_soap_structure():
    return {
        "Subjective": {"Chief_Complaint": "", "History_of_Present_Illness": ""},
        "Objective": {"Physical_Exam": "", "Observations": ""},
        "Assessment": {"Diagnosis": "", "Severity": ""},
        "Plan": {"Treatment": "", "Follow-Up": ""}
    }
