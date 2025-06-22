import streamlit as st
import os

from models.ner import extract_entities, extract_keywords
from models.summarizer import summarize_text
from models.sentiment import analyze_sentiment_intent
from models.soap import generate_soap_note

sample_text = open('sample.txt', 'r').read()

st.title("🩺 Physician Note Generator")

st.write("### Paste or edit transcript:")

text = st.text_area("Transcript", sample_text, height=400)

task = st.selectbox("Select Task:", [
    "Task 1: Named Entity Recognition",
    "Task 2: Sentiment & Intent Analysis",
    "Task 3: SOAP Note Generation (Bonus)"
    #"Task 4: Summarization"
])

if st.button("Run Task"):
    with st.spinner("Processing..."):
        if task.startswith("Task 1"):
            ner = extract_entities(text)
            keywords = extract_keywords(text)
            st.subheader("Named Entities")
            st.json(ner)
            st.subheader("Keywords")
            st.write(keywords)
        elif task.startswith("Task 2"):
            sentiment = analyze_sentiment_intent(text)
            st.subheader("Sentiment & Intent")
            st.json(sentiment)
        elif task.startswith("Task 3"):
            soap = generate_soap_note(text)
            st.subheader("SOAP Note")
            st.json(soap)
        elif task.startswith("Task 4"):
            summary = summarize_text(text)
            st.subheader("Summary")
            st.write(summary)
