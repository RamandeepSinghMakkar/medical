import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load spaCy for sentence splitting
nlp_spacy = spacy.load("en_core_web_sm")

# Load intent pipeline
intent_model_name = "Falconsai/intent_classification"
intent_pipeline = pipeline(
    "text-classification",
    model=intent_model_name,
    tokenizer=intent_model_name,
    device=-1  # CPU only
)

# Load sentiment pipeline
sentiment_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_pipeline = pipeline(
    "text-classification",
    model=sentiment_model_name,
    tokenizer=sentiment_model_name,
    device=-1  # CPU only
)

def classify_intent(sentences):
    results = intent_pipeline(sentences, batch_size=16, truncation=True)
    return [res['label'] for res in results]

def classify_sentiment(sentences):
    results = sentiment_pipeline(sentences, batch_size=16, truncation=True)
    output = []
    for res in results:
        label = res['label']  # e.g., "3 stars"
        score = int(label.split()[0])
        if score <= 2:
            output.append("Anxious")
        elif score == 3:
            output.append("Neutral")
        else:
            output.append("Reassured")
    return output

def analyze_sentiment_intent(text):
    doc = nlp_spacy(text)
    sentences = [sent.text for sent in doc.sents]

    if not sentences:
        return {"Sentiment": "Neutral", "Intent": "General inquiry"}

    intents = classify_intent(sentences)
    sentiments = classify_sentiment(sentences)

    final_sentiment = max(set(sentiments), key=sentiments.count)
    final_intent = max(set(intents), key=intents.count)

    return {
        "Sentiment": final_sentiment,
        "Intent": final_intent
    }
