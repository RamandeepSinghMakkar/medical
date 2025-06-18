import os
import requests
import json

# Set your Groq API key (make sure it's properly loaded)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-70b-8192"

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

IMPORTANT RULES:
- Output ONLY valid JSON.
- Do NOT add any explanation, commentary, or extra text.
- If any field is missing, leave it as empty string "".
- Ensure valid JSON with double quotes for keys and string values.
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
        "max_tokens": 1024  # control output length
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        raw_content = response.json()["choices"][0]["message"]["content"].strip()

        # Extra safeguard: try to extract JSON portion if model adds extra text accidentally
        try:
            # Sometimes model wraps JSON inside code block ```json ... ```
            if raw_content.startswith("```json"):
                raw_content = raw_content.replace("```json", "").replace("```", "").strip()
            result = json.loads(raw_content)
        except Exception as e:
            print("JSON parsing failed:", e)
            print("Model output:", raw_content)
            result = {
              "Subjective": {"Chief_Complaint": "", "History_of_Present_Illness": ""},
              "Objective": {"Physical_Exam": "", "Observations": ""},
              "Assessment": {"Diagnosis": "", "Severity": ""},
              "Plan": {"Treatment": "", "Follow-Up": ""}
            }
    else:
        print("API Error:", response.text)
        result = {
          "Subjective": {"Chief_Complaint": "", "History_of_Present_Illness": ""},
          "Objective": {"Physical_Exam": "", "Observations": ""},
          "Assessment": {"Diagnosis": "", "Severity": ""},
          "Plan": {"Treatment": "", "Follow-Up": ""}
        }

    return result
