import os
import requests
import spacy
from keybert import KeyBERT
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Load spacy model (for preprocessing, optional)
nlp_spacy = spacy.load("en_core_web_sm")

# Load KeyBERT model for keywords
kw_model = KeyBERT(model='sentence-transformers/all-MiniLM-L6-v2')

# Read Groq API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-70b-8192"


def extract_keywords(text, top_n=10):
    doc = nlp_spacy(text)
    cleaned_text = " ".join([sent.text for sent in doc.sents])
    keywords = kw_model.extract_keywords(cleaned_text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=top_n)
    keyword_list = [kw[0] for kw in keywords]
    return keyword_list

def extract_entities(text):
    system_prompt = """
You are a medical assistant. Extract the following information from the provided transcript of a physician-patient conversation. 

Output strictly in the following JSON format:

{
  "Patient_Name": "",
  "Symptoms": [],
  "Diagnosis": "",
  "Treatment": [],
  "Current_Status": "",
  "Prognosis": ""
}

If any field is not present, keep it empty.
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

        import json
        try:
            result = json.loads(content)
        except:
            result = {
                "Patient_Name": "",
                "Symptoms": [],
                "Diagnosis": "",
                "Treatment": [],
                "Current_Status": "",
                "Prognosis": ""
            }
    else:
        print("Error:", response.text)
        result = {
            "Patient_Name": "",
            "Symptoms": [],
            "Diagnosis": "",
            "Treatment": [],
            "Current_Status": "",
            "Prognosis": ""
        }

    return result
