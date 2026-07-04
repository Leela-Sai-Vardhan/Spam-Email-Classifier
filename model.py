import os
import re
import json
import joblib
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)


def generate_sample_data():
    spam_messages = [
        "FREE entry to win £1000 cash! Text WIN to 80085 now!",
        "Congratulations! You've won a $500 Walmart gift card. Click here to claim.",
        "URGENT: Your account has been compromised. Verify your identity NOW.",
        "Get rich quick! Earn $5000 per week from home. No experience needed!",
        "You have won a FREE iPhone 15! Call 1-800-CLAIM to receive.",
        "LIMITED TIME OFFER: Buy one get one FREE! Reply YES to claim.",
        "Your loan application has been approved! Get $10,000 today!",
        "Hot singles in your area want to meet you! Click the link!",
        "Make money fast! Invest $100 and get $1000 back in 24 hours!",
        "WINNER! You've been selected for a special prize. Call NOW!",
        "Free trial! No credit card required. Cancel anytime!",
        "Discount pharmacy online! Best prices on all medications!",
        "Double your income! Secret method revealed! Order now!",
        "You are a WINNER! Claim your prize at www.freecash.com",
        "Act NOW! Limited supply! 90% off designer watches!",
        "Hello Dear, I am a prince and I need your help to transfer $15 million.",
        "Your Amazon order has a problem. Click here to fix payment info.",
        "Bank alert: Your account will be closed. Verify at secure-bank.com",
        "Hey, check out this amazing deal! You won't believe the price!",
        "FREE membership! Join now and get exclusive access to premium content!",
        "Congratulations! You've been pre-approved for a credit card with 0% APR!",
        "Win a trip to Hawaii! Just text HAWAII to 55555!",
        "URGENT: Your package delivery failed. Reschedule at pkg-delivery.com",
        "Get 1000 Instagram followers FREE! No survey needed!",
        "Your PayPal account is limited. Login immediately to restore access.",
    ]
    ham_messages = [
        "Hey, are we still meeting for lunch tomorrow?",
        "Please review the attached document and let me know your thoughts.",
        "Happy birthday! Hope you have an amazing day!",
        "The meeting has been rescheduled to 3 PM on Thursday.",
        "Can you pick up some groceries on your way home?",
        "Thanks for helping me with the project yesterday.",
        "I'll be working from home today. Call me if you need anything.",
        "The movie starts at 8. Want me to save you a seat?",
        "Mom called. She wants us to visit this weekend.",
        "Don't forget about the dentist appointment on Monday.",
        "Great job on the presentation today! The client was impressed.",
        "I'm running late. Traffic is terrible right now.",
        "Can you send me the files we discussed earlier?",
        "The kids have soccer practice at 4 PM today.",
        "Let me know when you're free for a coffee catch-up.",
        "I finished reading that book you recommended. It was great!",
        "Do you want to go for a run this evening?",
        "The plumber is coming between 10 and 12 tomorrow.",
        "Thanks for the recipe! I'll try cooking it this weekend.",
        "Meeting notes from today's standup are in the shared drive.",
        "Can you help me move some furniture this Saturday?",
        "I just got home. Want me to make dinner tonight?",
        "The concert tickets go on sale Friday at noon.",
        "Your package has been delivered to your doorstep.",
        "Reminder: Team outing this Friday at 5 PM.",
    ]

    data = []
    for msg in spam_messages:
        data.append({'label': 'spam', 'message': msg})
    for msg in ham_messages:
        data.append({'label': 'ham', 'message': msg})

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(DATA_DIR, 'spam.csv'), index=False)
    return df


def load_data():
    csv_path = os.path.join(DATA_DIR, 'spam.csv')
    if not os.path.exists(csv_path):
        print("Generating sample dataset...")
        return generate_sample_data()
    df = pd.read_csv(csv_path)
    if 'label' not in df.columns or 'message' not in df.columns:
        return generate_sample_data()
    return df


def train_models():
    df = load_data()
    df['processed'] = df['message'].apply(preprocess_text)
    X = df['processed']
    y = df['label'].map({'spam': 1, 'ham': 0})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models = {
        'naive_bayes': MultinomialNB(alpha=0.1),
        'svm': LinearSVC(max_iter=10000, C=1.0),
        'logistic_regression': LogisticRegression(max_iter=1000, C=1.0),
    }

    comparison = {}
    best_model = None
    best_accuracy = 0

    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)
        comparison[name] = {
            'accuracy': round(acc * 100, 2),
            'report': classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])
        }
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'spam_model.pkl'))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, 'vectorizer.pkl'))

    with open(os.path.join(MODELS_DIR, 'comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2)

    top_features = get_top_features(best_model, vectorizer)
    with open(os.path.join(MODELS_DIR, 'top_features.json'), 'w') as f:
        json.dump(top_features, f)

    print(f"Models trained! Best: {best_accuracy*100:.2f}% accuracy")
    return comparison


def get_top_features(model, vectorizer, n=15):
    feature_names = vectorizer.get_feature_names_out()
    if hasattr(model, 'coef_'):
        coefs = model.coef_[0]
        top_spam_idx = np.argsort(coefs)[-n:][::-1]
        top_ham_idx = np.argsort(coefs)[:n]
        return {
            'spam_words': [{'word': feature_names[i], 'score': round(float(coefs[i]), 4)} for i in top_spam_idx],
            'ham_words': [{'word': feature_names[i], 'score': round(float(coefs[i]), 4)} for i in top_ham_idx],
        }
    return {'spam_words': [], 'ham_words': []}


def predict(text):
    model_path = os.path.join(MODELS_DIR, 'spam_model.pkl')
    vectorizer_path = os.path.join(MODELS_DIR, 'vectorizer.pkl')

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("Models not found. Training...")
        train_models()

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    processed = preprocess_text(text)
    tfidf = vectorizer.transform([processed])
    prediction = model.predict(tfidf)[0]

    confidence = 0.0
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(tfidf)[0]
        confidence = round(float(max(proba)) * 100, 2)
    elif hasattr(model, 'decision_function'):
        decision = model.decision_function(tfidf)[0]
        confidence = round(float(abs(decision)) * 10, 2)
        confidence = min(confidence, 99.99)

    return {
        'label': 'spam' if prediction == 1 else 'ham',
        'confidence': confidence,
        'is_spam': prediction == 1,
    }


if __name__ == '__main__':
    comparison = train_models()
    print("\nModel Comparison:")
    for name, metrics in comparison.items():
        print(f"  {name}: {metrics['accuracy']}%")
