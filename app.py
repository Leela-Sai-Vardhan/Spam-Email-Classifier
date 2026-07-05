import os
import json
from flask import Flask, render_template, request, jsonify
from model import predict, train_models

app = Flask(__name__)

EXAMPLE_MESSAGES = [
    {"text": "FREE entry to win £1000 cash! Text WIN to 80085 now!", "type": "spam"},
    {"text": "Congratulations! You've won a $500 Walmart gift card. Click here to claim.", "type": "spam"},
    {"text": "URGENT: Your account has been compromised. Verify your identity NOW.", "type": "spam"},
    {"text": "Hey, are we still meeting for lunch tomorrow?", "type": "ham"},
    {"text": "Please review the attached document and let me know your thoughts.", "type": "ham"},
    {"text": "Thanks for helping me with the project yesterday.", "type": "ham"},
    {"text": "Your PayPal account is limited. Login immediately to restore access.", "type": "spam"},
    {"text": "Don't forget about the dentist appointment on Monday.", "type": "ham"},
]

TOP_FEATURES_PATH = os.path.join(os.path.dirname(__file__), 'models', 'top_features.json')
COMPARISON_PATH = os.path.join(os.path.dirname(__file__), 'models', 'comparison.json')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/classify', methods=['POST'])
def classify():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Please provide a message.'}), 400

    message = data['message'].strip()
    if not message:
        return jsonify({'error': 'Message is empty.'}), 400

    result = predict(message)
    return jsonify(result)


@app.route('/api/examples')
def get_examples():
    return jsonify(EXAMPLE_MESSAGES)


@app.route('/api/model-comparison')
def model_comparison():
    if os.path.exists(COMPARISON_PATH):
        with open(COMPARISON_PATH, 'r') as f:
            data = json.load(f)
        simplified = {}
        for name, metrics in data.items():
            simplified[name] = {
                'accuracy': metrics['accuracy'],
                'name': name.replace('_', ' ').title()
            }
        return jsonify(simplified)
    return jsonify({})


@app.route('/api/top-features')
def top_features():
    if os.path.exists(TOP_FEATURES_PATH):
        with open(TOP_FEATURES_PATH, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({'spam_words': [], 'ham_words': []})


@app.route('/api/retrain', methods=['POST'])
def retrain():
    comparison = train_models()
    return jsonify({'status': 'success', 'comparison': comparison})


if __name__ == '__main__':
    if not os.path.exists(COMPARISON_PATH):
        print("First run: training models...")
        train_models()
    app.run(debug=True, port=5000)
