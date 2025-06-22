import os
import requests
import json
import spacy
from keybert import KeyBERT
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load models
nlp_spacy = spacy.load("en_core_web_sm")
kw_model = KeyBERT(model='sentence-transformers/all-MiniLM-L6-v2')

# Groq API details
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-8b-8192"  # you can switch to 70b later if needed

# ----------- Fast Preprocessing Step -------------
def preprocess(text):
    doc = nlp_spacy(text)
    # Extract named entities from spaCy (weak but fast)
    entities = [ent.text for ent in doc.ents]
    # Extract keywords from KeyBERT
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=10)
    keyword_list = [kw[0] for kw in keywords]

    # Summarize to send smaller input to Groq
    summary = {
        "Entities": entities,
        "Keywords": keyword_list
    }
    return summary

# ----------- Groq LLM extraction -------------
def extract_entities_with_groq(text, summary):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    system_prompt = f"""
You are a medical assistant. Extract key medical information based on the following conversation and extracted keywords/entities:

Conversation: {text}

Entities extracted: {summary['Entities']}
Keywords extracted: {summary['Keywords']}

Return output strictly in valid JSON format like this:

{{
  "Patient_Name": "",
  "Symptoms": [],
  "Diagnosis": "",
  "Treatment": [],
  "Current_Status": "",
  "Prognosis": ""
}}

Rules:
- Output only valid JSON, no explanations.
- If any field is missing, leave empty string or empty list.
"""

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 1024
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        try:
            raw_output = response.json()["choices"][0]["message"]["content"].strip()
            if raw_output.startswith("```json"):
                raw_output = raw_output.replace("```json", "").replace("```", "").strip()
            result = json.loads(raw_output)
            return normalize_ner_structure(result)
        except Exception as e:
            print("JSON parsing failed:", e)
            return empty_ner_structure()
    else:
        print("Groq API Error:", response.text)
        return empty_ner_structure()

# ----------- Normalization -------------
def normalize_ner_structure(raw):
    def normalize_array(field):
        if isinstance(field, list):
            return field
        if isinstance(field, dict):
            try:
                items = sorted(field.items(), key=lambda x: int(x[0]))
                return [v for k, v in items]
            except:
                pass
        return []

    return {
        "Patient_Name": raw.get("Patient_Name", ""),
        "Symptoms": normalize_array(raw.get("Symptoms", [])),
        "Diagnosis": raw.get("Diagnosis", ""),
        "Treatment": normalize_array(raw.get("Treatment", [])),
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

# ----------- Full Pipeline -------------
def extract_medical_info(text):
    summary = preprocess(text)
    result = extract_entities_with_groq(text, summary)
    return result
