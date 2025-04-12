from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load model and tokenizer
model_path = os.path.join(os.getcwd(), "lstm_fakenews_model.h5")
tokenizer_path = os.path.join(os.getcwd(), "tokenizer.pkl")

model = load_model(model_path)
tokenizer = joblib.load(tokenizer_path)

MAXLEN = 300  # Length used in training


@app.route('/')
def home():
    """Serve the main UI page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle news prediction POST requests"""
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Text input is empty"}), 400

    # Preprocess input
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAXLEN)

    # Predict
    prediction = model.predict(padded)[0][0]
    label = "FAKE" if prediction > 0.5 else "REAL"

    return jsonify({
        "prediction": label,
        "confidence": float(prediction)
    })


if __name__ == '__main__':
    app.run(debug=True)
