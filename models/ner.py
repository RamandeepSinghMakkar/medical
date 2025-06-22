import os
import requests
import json
import spacy
from keybert import KeyBERT
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Load Spacy model
nlp_spacy = spacy.load("en_core_web_sm")

# Load KeyBERT model for keywords
kw_model = KeyBERT(model='sentence-transformers/all-MiniLM-L6-v2')

# Read Groq API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-8b-8192"

# --- Keyword extraction function ---
def extract_keywords(text, top_n=10):
    doc = nlp_spacy(text)
    cleaned_text = " ".join([sent.text for sent in doc.sents])
    keywords = kw_model.extract_keywords(cleaned_text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=top_n)
    keyword_list = [kw[0] for kw in keywords]
    return keyword_list

# --- NER extraction using Groq function calling ---
def extract_entities(text):
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
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "extract_medical_entities",
                    "description": "Extract relevant medical information from the conversation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "Patient_Name": {"type": "string"},
                            "Symptoms": {"type": "array", "items": {"type": "string"}},
                            "Diagnosis": {"type": "string"},
                            "Treatment": {"type": "array", "items": {"type": "string"}},
                            "Current_Status": {"type": "string"},
                            "Prognosis": {"type": "string"},
                        },
                        "required": [
                            "Patient_Name", "Symptoms", "Diagnosis",
                            "Treatment", "Current_Status", "Prognosis"
                        ]
                    }
                }
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "extract_medical_entities"}},
        "temperature": 0.0
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        try:
            function_args = response.json()["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            raw_result = json.loads(function_args)
            result = normalize_ner_structure(raw_result)
        except Exception as e:
            print("JSON parsing failed:", e)
            result = empty_ner_structure()
    else:
        print("API Error:", response.text)
        result = empty_ner_structure()

    return result

def normalize_ner_structure(raw):
    """
    Normalize the extracted entity result into correct key order
    """
    return {
        "Patient_Name": raw.get("Patient_Name", ""),
        "Symptoms": raw.get("Symptoms", []),
        "Diagnosis": raw.get("Diagnosis", ""),
        "Treatment": raw.get("Treatment", []),
        "Current_Status": raw.get("Current_Status", ""),
        "Prognosis": raw.get("Prognosis", "")
    }

def empty_ner_structure():
    return {
        "Patient_Name": "",
        "Symptoms": [],
        "Diagnosis": "",
        "Treatment": [],
        "Current_Status": "",
        "Prognosis": ""
    }
