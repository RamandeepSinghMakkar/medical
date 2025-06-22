import spacy
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Use CPU for Streamlit
device = torch.device("cpu")

# Load spaCy for sentence splitting
nlp_spacy = spacy.load("en_core_web_sm")

# Load Sentiment Model
sentiment_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name).to(device)

# Load lightweight Intent Model (for now, reuse sentiment model as placeholder)
intent_model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
intent_tokenizer = AutoTokenizer.from_pretrained(intent_model_name)
intent_model = AutoModelForSequenceClassification.from_pretrained(intent_model_name).to(device)

# Sentiment classification
def classify_sentiment(sentence):
    inputs = sentiment_tokenizer(sentence, return_tensors="pt", truncation=True).to(device)
    outputs = sentiment_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    probs = probs.detach().cpu().numpy()[0]
    label_id = np.argmax(probs)
    if label_id + 1 <= 2:
        return "Anxious"
    elif label_id + 1 == 3:
        return "Neutral"
    else:
        return "Reassured"

# Intent classification (basic placeholder, replace with better model later)
def classify_intent(sentence):
    inputs = intent_tokenizer(sentence, return_tensors="pt", truncation=True).to(device)
    outputs = intent_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    probs = probs.detach().cpu().numpy()[0]
    label_id = np.argmax(probs)

    # map model classes to your intents manually
    label_map = {
        0: "Reporting symptoms",
        1: "Expressing concern",
        2: "Seeking reassurance",
        3: "General inquiry",
        4: "Gratitude"
    }
    return label_map.get(label_id, "General inquiry")

# Main analysis
def analyze_sentiment_intent(text):
    doc = nlp_spacy(text)
    sentences = [sent.text for sent in doc.sents]
    sentiments = []
    intents = []
    for sent_text in sentences:
        sentiment = classify_sentiment(sent_text)
        intent = classify_intent(sent_text)
        sentiments.append(sentiment)
        intents.append(intent)
    final_sentiment = max(set(sentiments), key=sentiments.count)
    final_intent = max(set(intents), key=intents.count)
    return {
        "Sentiment": final_sentiment,
        "Intent": final_intent
    }
