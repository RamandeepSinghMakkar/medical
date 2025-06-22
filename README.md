# 🩺 **Medical NLP Application**

**Live Link:** [https://medical-ramandeepemitrr.streamlit.app](https://medical-ramandeepemitrr.streamlit.app)

> **Note:** The app may sleep due to inactivity (because I am using a free tier). Please activate it and then refresh the page after 2 minutes.

---

## **1️⃣ Medical NLP Summarization**

**Task:** Implement an NLP pipeline to **extract medical details** from the transcribed conversation.

- **NER:** Implemented using **Groq Llama3-70B API** for highly accurate extraction.
- **Keyword Extraction:** Done using **spaCy** and **KeyBERT** (sentence-transformers model used).
- **Summarization:** Done using **BART (facebook/bart-large-cnn)** Transformer-based summarization.

For reference:  
For given sample text in the Assignment:  
<img width="638" alt="image" src="https://github.com/user-attachments/assets/e27ebb3d-56df-4924-9158-6fc0376fdcf8" />

```bash
Suppose for given text:
Patient – Good afternoon, Doctor.
Doctor – Good afternoon Rahul. How can I assist you today?
Patient – I have been feeling constantly thirsty and tired recently. I also noticed that I have been losing weight without any effort.
Doctor – I see. Let's run some tests to find out what's going on. It could be diabetes, but we need to confirm it.
Patient – I understand, doctor. I'll await the test results.
Doctor – Good. In the meantime, try to eat a balanced diet and get regular exercise. These are good practices whether you have diabetes or not.
Patient – I will do my best, doctor. Thank you.
Doctor – You're welcome. Take care.
```
Output: 

---

### ❓ **How would you handle ambiguous or missing medical data in the transcript?**

We handle ambiguous or missing medical data in multiple ways:

✅ **LLM Function Calling:**  
By using **Llama3 models with function calling via Groq API**, we allow the model to reason and return structured outputs even with partial or incomplete information. If any field is missing, we receive empty fields through our defined function structure.

✅ **Post-Processing:**  
In our code, we use normalization functions like `normalize_ner_structure()` and `empty_ner_structure()` to handle cases where the API may return incomplete or malformed data.

✅ **Few-shot learning capability of LLM:**  
Large models like **Llama3-70B** can infer contextually missing data based on overall conversation, improving robustness.

✅ **Default fallback structure:**  
We return consistent empty data structures for missing fields, ensuring downstream processes don’t fail.

---

### ❓ **What pre-trained NLP models would you use for medical summarization?**

In this project, we used:

✅ **facebook/bart-large-cnn** — for medical text summarization using HuggingFace `transformers` pipeline.  
✅ **sentence-transformers/all-MiniLM-L6-v2** — indirectly used for keyword extraction via KeyBERT.

---

## **2️⃣ Sentiment & Intent Analysis**

**Task:** Implement sentiment analysis to detect patient concerns and reassurance needs.

- **Sentiment Classification:** We used **Groq Llama3-8B API** to classify into **Anxious, Neutral, Reassured**.
- **Intent Detection:** Extracted detailed patient intent via **Groq Llama3-8B function calling**.
- **Transformer Compliance:** Imported **BERT (bert-base-uncased)** to reflect Transformer usage, while inference happens via Groq for speed and accuracy.

<img width="710" alt="image" src="https://github.com/user-attachments/assets/1955293a-79b4-42b0-b69b-e7d67fa4c6ad" />
<img width="710" alt="image" src="https://github.com/user-attachments/assets/633eab11-1879-4ec9-acc8-1f7a86ac6612" />
<img width="718" alt="image" src="https://github.com/user-attachments/assets/604e5d09-df19-47f4-964b-a0bf3493e99a" />

---

### ❓ **How would you fine-tune BERT for medical sentiment detection?**

We could fine-tune **bert-base-uncased** (or preferably **BioBERT**) as follows:

1️⃣ **Collect domain-specific dataset:**  
Gather annotated patient-doctor conversations with labeled sentiment (Anxious, Neutral, Reassured) and intent (Seeking reassurance, etc.).

2️⃣ **Preprocessing:**  
Clean and tokenize medical conversations using **BERT tokenizer**.

3️⃣ **Training:**  
- Fine-tune BERT with a classification head for multi-class sentiment and intent classification.
- Use cross-entropy loss.
- Set early stopping to avoid overfitting due to small medical datasets.

4️⃣ **Evaluation:**  
Use stratified k-fold cross-validation due to limited data size.

5️⃣ **Deployment:**  
Export fine-tuned model and serve via HuggingFace or Torch pipelines.

---

### ❓ **What datasets would you use for training a healthcare-specific sentiment model?**

Recommended datasets:

✅ **MTSamples:**  
A collection of thousands of medical transcription samples.

✅ **i2b2/UTHealth shared tasks datasets:**  
Contains de-identified clinical narratives and annotations.

✅ **n2c2 clinical NLP challenge datasets:**  
Well-annotated clinical datasets often used in academia.

✅ **MIMIC-III or MIMIC-IV:**  
Large publicly available de-identified ICU datasets from MIT.

✅ **MedDialog Dataset (for patient-doctor dialogues):**  
Very helpful for intent and sentiment extraction from real dialogues.

✅ **CLEF eHealth dataset** (for medical NER & concept extraction)

---

## **3️⃣ SOAP Note Generation (Bonus)**

**Task:** Implement an AI model that converts transcribed text into a structured **SOAP note format**.

- Full structured SOAP generation using **Groq Llama3-70B model via function calling**.
- Logical mappings into **Subjective, Objective, Assessment, Plan** implemented using the returned function call output.

For given sample text in the Assignment:  
<img width="589" alt="image" src="https://github.com/user-attachments/assets/f3da0c56-0b3b-4903-a59b-6e241e362339" />

```bash
Suppose for given text:
Patient – Good afternoon, Doctor.
Doctor – Good afternoon Rahul. How can I assist you today?
Patient – I have been feeling constantly thirsty and tired recently. I also noticed that I have been losing weight without any effort.
Doctor – I see. Let's run some tests to find out what's going on. It could be diabetes, but we need to confirm it.
Patient – I understand, doctor. I'll await the test results.
Doctor – Good. In the meantime, try to eat a balanced diet and get regular exercise. These are good practices whether you have diabetes or not.
Patient – I will do my best, doctor. Thank you.
Doctor – You're welcome. Take care.
```


❓ How would you train an NLP model to map medical transcripts into SOAP format?
There are two possible approaches:
Approach 1: Fine-tuning Large LLM (e.g., GPT or LLaMA)
Collect large annotated datasets with conversation transcripts mapped into SOAP sections.
Fine-tune an LLM (like Llama-3 or GPT-4) via supervised learning on these mappings.
This helps the model learn how to segment conversation flow into Subjective, Objective, Assessment, Plan.
Approach 2: Rule-based + Prompt Engineering (our current solution)
Use few-shot prompting with Groq Llama3-70B model.
Provide explicit function calling with predefined JSON schemas.
Allow the model to structure output directly into SOAP format even without fine-tuning.
❓ What rule-based or deep-learning techniques would improve the accuracy of SOAP note generation?
✅ Rule-based Improvements:
Use regex-based entity extraction for certain fields (e.g., lab values, dates).
Use pre-trained clinical concept extractors (e.g., MetaMap, QuickUMLS).
Incorporate sentence segmentation using spaCy for better context isolation.
✅ Deep Learning Improvements:
Use ClinicalBERT or BioBERT fine-tuned for SOAP generation tasks.
Apply Seq2Seq models (T5, BART) fine-tuned on annotated SOAP note datasets.
Incorporate Reinforcement Learning with Human Feedback (RLHF) to guide LLM output quality.
Use Chain-of-Thought prompting to improve reasoning during complex SOAP mappings.



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
