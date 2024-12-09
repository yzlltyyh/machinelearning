from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer
from model import MultiDimensionalSentimentModel
from predict import predict_sentiment

app = Flask(__name__)

def load_model(model_path):
    model = MultiDimensionalSentimentModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(torch.device('cpu'))
    model.eval()
    return model

model = load_model('WORKSPACE1/best_model.pt')
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json.get('text')
    if not text:
        return jsonify({'error': 'Text is required for prediction'}), 400

    sentiment, probabilities, confidence = predict_sentiment(text, model, tokenizer, torch.device('cpu'))

    return jsonify({
        'sentiment': sentiment,
        'probabilities': probabilities,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
