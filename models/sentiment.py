import spacy
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Hardcode CPU device manually
device = torch.device("cpu")

# Load intent model manually on CPU
intent_model_name = "Falconsai/intent_classification"
intent_tokenizer = AutoTokenizer.from_pretrained(intent_model_name)
intent_model = AutoModelForSequenceClassification.from_pretrained(intent_model_name).to(device)

# Load sentiment model manually on CPU
sentiment_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name).to(device)

# Load spaCy for sentence splitting
nlp_spacy = spacy.load("en_core_web_sm")

# Intent classifier function (batched version)
def classify_intent(sentences):
    encoded = intent_tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = intent_model(**encoded)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
    label_ids = np.argmax(probs, axis=1)
    id2label = intent_model.config.id2label
    return [id2label[i] for i in label_ids]

# Sentiment classifier function (batched version)
def classify_sentiment(sentences):
    encoded = sentiment_tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = sentiment_model(**encoded)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
    label_ids = np.argmax(probs, axis=1)
    
    results = []
    for label_id in label_ids:
        score = label_id + 1  # model returns 1 to 5 stars
        if score <= 2:
            results.append("Anxious")
        elif score == 3:
            results.append("Neutral")
        else:
            results.append("Reassured")
    return results

# Final combined analysis function
def analyze_sentiment_intent(text):
    doc = nlp_spacy(text)
    sentences = [sent.text for sent in doc.sents]

    if not sentences:
        return {"Sentiment": "Neutral", "Intent": "General inquiry"}

    # Batch prediction
    intents = classify_intent(sentences)
    sentiments = classify_sentiment(sentences)

    # Majority voting
    final_sentiment = max(set(sentiments), key=sentiments.count)
    final_intent = max(set(intents), key=intents.count)

    return {
        "Sentiment": final_sentiment,
        "Intent": final_intent
    }
