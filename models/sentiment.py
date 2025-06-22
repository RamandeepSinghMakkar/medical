import spacy
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util

# Always run on CPU for Streamlit Cloud
device = torch.device("cpu")

# Load spaCy
nlp_spacy = spacy.load("en_core_web_sm")

# Load sentiment model
sentiment_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name).to(device)

# Load sentence transformer for intent similarity
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

# Define possible intents
intent_examples = {
    "Seeking reassurance": "I hope everything will be fine. Should I be worried?",
    "Reporting symptoms": "I have pain in my chest and feel nauseous.",
    "Expressing concern": "I'm quite worried about my health.",
    "General inquiry": "Can you tell me about my treatment?",
    "Gratitude": "Thank you so much for your help."
}

# Precompute intent embeddings
intent_embeddings = {intent: embedding_model.encode(example, convert_to_tensor=True) 
                     for intent, example in intent_examples.items()}

def classify_sentiment(sentence):
    inputs = sentiment_tokenizer(sentence, return_tensors="pt", truncation=True).to(device)
    outputs = sentiment_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    probs = probs.detach().cpu().numpy()[0]
    label_id = np.argmax(probs)
    
    # 1 to 5 stars output — map to your schema
    if label_id + 1 <= 2:
        return "Anxious"
    elif label_id + 1 == 3:
        return "Neutral"
    else:
        return "Reassured"

def classify_intent(sentence):
    sentence_embedding = embedding_model.encode(sentence, convert_to_tensor=True)
    scores = {intent: util.pytorch_cos_sim(sentence_embedding, emb).item() 
              for intent, emb in intent_embeddings.items()}
    best_intent = max(scores, key=scores.get)
    return best_intent

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
