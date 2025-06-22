import os
import requests
import json

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-70b-8192"


def generate_soap_note(text):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a medical assistant."},
            {"role": "user", "content": text}
        ],
        "functions": [
            {
                "name": "generate_soap",
                "description": "Generate SOAP note based on conversation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "Subjective": {
                            "type": "object",
                            "properties": {
                                "Chief_Complaint": {"type": "string"},
                                "History_of_Present_Illness": {"type": "string"}
                            }
                        },
                        "Objective": {
                            "type": "object",
                            "properties": {
                                "Physical_Exam": {"type": "string"},
                                "Observations": {"type": "string"}
                            }
                        },
                        "Assessment": {
                            "type": "object",
                            "properties": {
                                "Diagnosis": {"type": "string"},
                                "Severity": {"type": "string"}
                            }
                        },
                        "Plan": {
                            "type": "object",
                            "properties": {
                                "Treatment": {"type": "string"},
                                "Follow-Up": {"type": "string"}
                            }
                        }
                    }
                }
            }
        ],
        "function_call": {"name": "generate_soap"},
        "temperature": 0.0
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        try:
            function_args = response.json()["choices"][0]["message"]["function_call"]["arguments"]
            result = json.loads(function_args)
        except Exception as e:
            print("JSON parsing failed:", e)
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
