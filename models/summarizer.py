from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    if len(text) > 1024:
        chunks = [text[i:i+1024] for i in range(0, len(text), 1024)]
        summaries = [summarizer(chunk)[0]['summary_text'] for chunk in chunks]
        return {'summary': ' '.join(summaries)}
    return {'summary': summarizer(text)[0]['summary_text']}
