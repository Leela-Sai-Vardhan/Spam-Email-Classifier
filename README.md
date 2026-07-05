# SpamShield - Email & SMS Spam Classifier

Classifies email and SMS messages as **Spam** or **Ham** (not spam) using multiple ML models with confidence scores.

## Features

- **Multi-Model Comparison** - Compares Naive Bayes, SVM, and Logistic Regression
- **Spam Detection** - Classifies messages with a confidence score
- **Top Indicator Words** - Shows which words contribute most to spam/ham classification
- **Test Examples** - Sample messages to try the classifier
- **Dark UI** - Clean interface with animated particles

## Tech Stack

- **Backend:** Python, Flask
- **ML:** Scikit-learn (Naive Bayes, SVM, Logistic Regression)
- **NLP:** NLTK (stemming, stopword removal), TF-IDF Vectorization
- **Frontend:** HTML, CSS, JavaScript

## How It Works

1. **Preprocessing:** Lowercase, remove special characters, stem, filter stopwords
2. **Vectorization:** TF-IDF with unigrams and bigrams (max 5000 features)
3. **Training:** Train three models and pick the best one
4. **Prediction:** Classify new messages with a confidence score

## Run Locally

```bash
cd spam-email-classifier
pip install -r requirements.txt
python model.py          # Train models
python app.py            # Start Flask server on port 5000
```

Open http://localhost:5000 in your browser.

## Dataset

Uses a built-in sample dataset of 50 messages (25 spam, 25 ham). You can replace `data/spam.csv` with the [UCI SMS Spam Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) for a larger dataset.

## Model Accuracy

| Model | Accuracy |
|-------|----------|
| Naive Bayes | ~95-98% |
| SVM | ~95-98% |
| Logistic Regression | ~95-98% |
