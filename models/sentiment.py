import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import numpy as np

# Hardcode CPU device manually
device = torch.device("cpu")

# Load intent model manually on CPU
intent_model_name = "Falconsai/intent_classification"
intent_tokenizer = AutoTokenizer.from_pretrained(intent_model_name)
intent_model = AutoModelForSequenceClassification.from_pretrained(intent_model_name).to(device)

intent_clf = pipeline(
    "text-classification",
    model=intent_model,
    tokenizer=intent_tokenizer,
    device_map={"": "cpu"}  # <- force CPU inference even inside pipeline
)

# Your defined intent labels (still kept for reference even though pipeline returns predefined labels)
intent_labels = [
    "Seeking reassurance",
    "Reporting symptoms",
    "Expressing concern",
    "General inquiry",
    "Gratitude"
]

def classify_intent(sentences):
    results = intent_clf(sentences)
    return [res['label'] for res in results]

# Load spaCy for sentence splitting
nlp_spacy = spacy.load("en_core_web_sm")

# Sentiment model (same process for sentiment)
sentiment_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name).to(device)

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

def analyze_sentiment_intent(text):
    doc = nlp_spacy(text)
    sentences = [sent.text for sent in doc.sents]

    intent_labels_list = classify_intent(sentences)
    sentiments = []
    intents = []

    for sent_text, intent in zip(sentences, intent_labels_list):
        sentiment = classify_sentiment(sent_text)
        sentiments.append(sentiment)
        intents.append(intent)

    final_sentiment = max(set(sentiments), key=sentiments.count)
    final_intent = max(set(intents), key=intents.count)

    return {
        "Sentiment": final_sentiment,
        "Intent": final_intent
    }
