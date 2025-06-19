import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import numpy as np

intent_clf = pipeline("text-classification", model="Falconsai/intent_classification")

# Our defined intent labels
intent_labels = [
    "Seeking reassurance",
    "Reporting symptoms",
    "Expressing concern",
    "General inquiry",
    "Gratitude"
]


def classify_intent(text):
    return intent_clf(text)[0]['label']


# Load spaCy for sentence splitting
nlp_spacy = spacy.load("en_core_web_sm")

# Sentiment model
sentiment_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)


def classify_sentiment(sentence):
    inputs = sentiment_tokenizer(sentence, return_tensors="pt", truncation=True)
    outputs = sentiment_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    probs = probs.detach().numpy()[0]
    label_id = np.argmax(probs)
    
    # The model returns 5 labels: 1-5 stars (1 very negative, 5 very positive)
    if label_id + 1 <= 2:
        return "Anxious"
    elif label_id + 1 == 3:
        return "Neutral"
    else:
        return "Reassured"



def analyze_sentiment_intent(text):
    doc = nlp_spacy(text)
    sentiments = []
    intents = []

    for sent in doc.sents:
        sentiment = classify_sentiment(sent.text)
        intent = classify_intent(sent.text)
        sentiments.append(sentiment)
        intents.append(intent)

    # Aggregate majority result
    final_sentiment = max(set(sentiments), key=sentiments.count)
    final_intent = max(set(intents), key=intents.count)

    return {
        "Sentiment": final_sentiment,
        "Intent": final_intent
    }
