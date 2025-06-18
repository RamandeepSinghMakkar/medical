from flask import Flask, render_template, request
from models.ner import extract_entities, extract_keywords
from models.summarizer import summarize_text
from models.sentiment import analyze_sentiment_intent
from models.soap import generate_soap_note

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
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

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
