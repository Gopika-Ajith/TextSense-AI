# DocuSense AI

AI-Powered Document Classification System

## Overview

DocuSense AI is a Machine Learning and Natural Language Processing (NLP) application that classifies documents and text into five categories:

* Business
* Entertainment
* Politics
* Sport
* Technology

The system uses TF-IDF Vectorization and Logistic Regression to analyze document content and predict the most relevant category.

---

## Features

* Text Classification
* PDF Document Upload
* TXT File Upload
* Confidence Score Prediction
* Category Probability Visualization
* Low Confidence Warning System
* Interactive Streamlit Web Application
* Real-Time Document Analysis

---

## Dataset

**BBC News Dataset**

* 2225 Documents
* 5 Categories

Categories:

* Business
* Entertainment
* Politics
* Sport
* Technology

---

## Machine Learning Pipeline

### Text Preprocessing

* Data Cleaning
* Stop Word Removal
* TF-IDF Vectorization

### Models Evaluated

| Model                   | Accuracy |
| ----------------------- | -------- |
| Multinomial Naive Bayes | 96.40%   |
| Logistic Regression     | 97.08%   |

### Selected Model

**Logistic Regression**

Accuracy: **97.08%**

---

## Technologies Used

* Python
* Pandas
* Scikit-Learn
* Streamlit
* Joblib
* PyPDF2

---

## Project Structure

DocuSense-AI

├── app.py

├── train.py

├── model.pkl

├── vectorizer.pkl

├── logo.png

├── requirements.txt

├── README.md

└── bbc/

---

## How It Works

1. Enter text manually or upload a TXT/PDF document.
2. The document is transformed using TF-IDF Vectorization.
3. Logistic Regression predicts the category.
4. Confidence score and probability distribution are displayed.
5. The result is visualized through charts and tables.

---

## Run Locally

Install Dependencies:

pip install -r requirements.txt

Run Application:

streamlit run app.py

---

## Live Demo

https://docusense-ai-agop.streamlit.app/

---

## Results

* Best Accuracy: 97.08%
* Logistic Regression Selected
* PDF & TXT Document Classification Supported
* Confidence-Based Predictions
* Interactive Visualization Dashboard

---

## Future Improvements

* Batch Document Classification
* CSV Upload Support
* Prediction History
* Enhanced Dashboard Analytics
* Additional Document Categories
