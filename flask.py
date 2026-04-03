from flask import Flask, request, jsonify
import torch
import pickle
import json

from training.preprocess import encode_and_pad
from training.SentimentLSTM import SentimentLSTM

app = Flask(__name__)

def load_models():
    with open("training/models/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    with open("training/models/model_config.json", "r") as f:
        config = json.load(f)

    model = SentimentLSTM(
        vocab_size       = config["vocab_size"],
        embed_dim        = config["embed_dim"],
        hidden_dim       = config["hidden_dim"],
        embedding_matrix = torch.zeros(config["vocab_size"], config["embed_dim"])
    )

    model.load_state_dict(
        torch.load("training/models/sentiment_model.pt", map_location="cpu")
    )
    model.eval()

    return vocab, model

vocab, model = load_models()

def run_inference(text):
    encoded = encode_and_pad(text, vocab)

    tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
 
    with torch.no_grad():                      
        logit = model(tensor)                  
        prob  = torch.sigmoid(logit).item()    
 
    label      = "Positive" if prob >= 0.5 else "Negative"
    confidence = prob if prob >= 0.5 else 1 - prob
 
    return {
        "label":      label,
        "confidence": round(confidence * 100, 2),
        "probability": round(prob, 4),
    }


# ── Routes ─────────────────────────────────────────────────────
 
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status":  "running",
        "message": "Sentiment Analysis API is live",
        "usage":   "POST /predict with JSON body: {\"text\": \"your review here\"}"
    })
 
 
@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
 
    data = request.get_json()
 
    if "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body"}), 400
 
    text = data["text"].strip()
 
    if len(text) == 0:
        return jsonify({"error": "Text cannot be empty"}), 400
 
    try:
        result = run_inference(text)
        result["input_text"] = text
        return jsonify(result), 200
 
    except Exception as e:
        return jsonify({"error": str(e)}), 500
 
 
# Run server
if __name__ == "__main__":
    app.run(
        host  = "0.0.0.0",
        port  = 5000,
        debug = True
    )
