from flask import Flask, render_template, request
import os
import requests
import json

from models.ner import extract_entities, extract_keywords
from models.summarizer import summarize_text
from models.sentiment import analyze_sentiment_intent
from models.soap import generate_soap_note

app = Flask(__name__)

# Load sample.txt correctly from project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_FILE = os.path.join(BASE_DIR, "sample.txt")

with open(SAMPLE_FILE, 'r') as file:
    sample_text = file.read()

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    text = sample_text  # default text loaded

    if request.method == 'POST':
        text = request.form['transcript']
        task = request.form['task']

        if task == 'ner':
            result = {
                'entities': extract_entities(text),
                'keywords': extract_keywords(text)
            }
        elif task == 'summarization':
            result = summarize_text(text)
        elif task == 'sentiment':
            result = analyze_sentiment_intent(text)
        elif task == 'soap':
            result = generate_soap_note(text)

    message = "Enter the transcription text:"

    return render_template('index.html', result=result, transcript=text, message=message)

if __name__ == '__main__':
    app.run(debug=True)
