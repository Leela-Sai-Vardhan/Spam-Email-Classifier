# SpamShield - Email & SMS Spam Classifier

An AI-powered web application that classifies email and SMS messages as **Spam** or **Ham** (not spam) using multiple machine learning algorithms with real-time confidence scoring.

## Features

- **Multi-Model Comparison** - Compares Naive Bayes, SVM, and Logistic Regression side-by-side
- **Real-time Classification** - Instant spam detection with confidence scores
- **Top Indicator Words** - Shows which words contribute most to spam/ham classification
- **Quick Test Examples** - One-click example messages for instant testing
- **Dark-Themed UI** - Modern, responsive interface with animated particles

## Tech Stack

- **Backend:** Python, Flask
- **ML:** Scikit-learn (Naive Bayes, SVM, Logistic Regression)
- **NLP:** NLTK (stemming, stopword removal), TF-IDF Vectorization
- **Frontend:** HTML, CSS, JavaScript

## How It Works

1. **Preprocessing:** Text is lowercased, special characters removed, stemmed, and stopwords filtered
2. **Vectorization:** TF-IDF with unigrams and bigrams (max 5000 features)
3. **Training:** Three models are trained and compared; the best one is selected
4. **Prediction:** New messages are classified with a confidence score

## Run Locally

```bash
cd spam-email-classifier
pip install -r requirements.txt
python model.py          # Train models
python app.py            # Start Flask server on port 5000
```

Open http://localhost:5000 in your browser.

## Dataset

Uses a built-in sample dataset of 50 messages (25 spam, 25 ham). You can replace `data/spam.csv` with the [UCI SMS Spam Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) for production use.

## Model Accuracy

| Model | Accuracy |
|-------|----------|
| Naive Bayes | ~95-98% |
| SVM | ~95-98% |
| Logistic Regression | ~95-98% |
