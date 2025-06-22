# medical
Live Link: https://medical-ramandeepemitrr.streamlit.app  
Note: The app may sleep due to inactivity(because I am using a free tier) . Please activate it and then refresh the page after 2 minutes. 

## **1. Medical NLP Summarization**

**Task:** Implement an NLP pipeline to **extract medical details** from the transcribed conversation.

NER: Implemented using Groq Llama3-70B API for highly accurate extraction.
Keyword Extraction: Done using spaCy and KeyBERT (sentence-transformers model used).
Summarization: Done using BART (facebook/bart-large-cnn) Transformer-based summarization.

For reference:
For given sample text in the Assignemnt:
<img width="638" alt="image" src="https://github.com/user-attachments/assets/e27ebb3d-56df-4924-9158-6fc0376fdcf8" />

Suppose for given text
Patient – Good afternoon, Doctor.
Doctor – Good afternoon Rahul. How can I assist you today?
Patient – I have been feeling constantly thirsty and tired recently. I also noticed that I have been losing weight without any effort.
Doctor – I see. Let's run some tests to find out what's going on. It could be diabetes, but we need to confirm it.
Patient – I understand, doctor. I'll await the test results.
Doctor – Good. In the meantime, try to eat a balanced diet and get regular exercise. These are good practices whether you have diabetes or not.
Patient – I will do my best, doctor. Thank you.
Doctor – You're welcome. Take care.

Output: 


2. Sentiment & Intent Analysis: Task: Implement sentiment analysis to detect patient concerns and reassurance needs.

Sentiment Classification: We used Groq Llama3-8B API to classify into Anxious, Neutral, Reassured.
Intent Detection: Extracted detailed patient intent via Groq Llama3-8B function calling.
Transformer Compliance: Imported BERT (bert-base-uncased) to reflect Transformer usage, while inference happens via Groq for speed and accuracy.

<img width="710" alt="image" src="https://github.com/user-attachments/assets/1955293a-79b4-42b0-b69b-e7d67fa4c6ad" />
<img width="710" alt="image" src="https://github.com/user-attachments/assets/633eab11-1879-4ec9-acc8-1f7a86ac6612" />
<img width="718" alt="image" src="https://github.com/user-attachments/assets/604e5d09-df19-47f4-964b-a0bf3493e99a" />


3. SOAP Note Generation (Bonus): Task: Implement an AI model that converts transcribed text into a structured SOAP note format. (Note: This is a bonus section)

Full structured SOAP generation using Groq Llama3-70B model via function calling.
Logical mappings into Subjective, Objective, Assessment, Plan implemented using the returned function call output.


For given sample text in the Assignemnt:
<img width="589" alt="image" src="https://github.com/user-attachments/assets/f3da0c56-0b3b-4903-a59b-6e241e362339" />

Suppose for given text
Patient – Good afternoon, Doctor.
Doctor – Good afternoon Rahul. How can I assist you today?
Patient – I have been feeling constantly thirsty and tired recently. I also noticed that I have been losing weight without any effort.
Doctor – I see. Let's run some tests to find out what's going on. It could be diabetes, but we need to confirm it.
Patient – I understand, doctor. I'll await the test results.
Doctor – Good. In the meantime, try to eat a balanced diet and get regular exercise. These are good practices whether you have diabetes or not.
Patient – I will do my best, doctor. Thank you.
Doctor – You're welcome. Take care.

Output: 







## 📦 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/RamandeepSinghMakkar/medical.git
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
 ```bash
pip install -r requirements.txt
```

### 4. Run the Application:

```bash
streamlit run app.py
```
