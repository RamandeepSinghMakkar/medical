import os
import requests
import json

# Use Groq or OpenAI or Fireworks etc.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-70b-8192"  # Excellent for structured generation

def generate_soap_note(text):
    system_prompt = """
You are a medical assistant. Analyze the following doctor-patient conversation and generate a SOAP note.

Output strictly in this JSON format:

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

If any information is missing, leave the field empty.
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
        "temperature": 0.0
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        content = response.json()["choices"][0]["message"]["content"]
        try:
            result = json.loads(content)
        except:
            result = {
              "Subjective": {"Chief_Complaint": "", "History_of_Present_Illness": ""},
              "Objective": {"Physical_Exam": "", "Observations": ""},
              "Assessment": {"Diagnosis": "", "Severity": ""},
              "Plan": {"Treatment": "", "Follow-Up": ""}
            }
    else:
        print("Error:", response.text)
        result = {
          "Subjective": {"Chief_Complaint": "", "History_of_Present_Illness": ""},
          "Objective": {"Physical_Exam": "", "Observations": ""},
          "Assessment": {"Diagnosis": "", "Severity": ""},
          "Plan": {"Treatment": "", "Follow-Up": ""}
        }

    return result
