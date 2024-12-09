from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer
from model import MultiDimensionalSentimentModel
from predict import predict_sentiment

# 在全局范围初始化 Flask app
app = Flask(__name__)

# 创建一个全局变量来存储模型和tokenizer
global_model = None
global_tokenizer = None

def init_model():
    """初始化模型和tokenizer的函数"""
    global global_model, global_tokenizer
    if global_model is None:
        global_model = MultiDimensionalSentimentModel()
        global_model.load_state_dict(torch.load('WORKSPACE1/best_model.pt', map_location=torch.device('cpu')))
        global_model.to(torch.device('cpu'))
        global_model.eval()
    
    if global_tokenizer is None:
        global_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

@app.route('/predict', methods=['POST'])
def predict():
    # 确保模型已经加载
    if global_model is None or global_tokenizer is None:
        init_model()
        
    text = request.json.get('text')
    if not text:
        return jsonify({'error': 'Text is required for prediction'}), 400

    sentiment, probabilities, confidence = predict_sentiment(text, global_model, global_tokenizer, torch.device('cpu'))

    # 将 Tensor 转换为普通 Python 数据类型
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.tolist()
    if isinstance(confidence, torch.Tensor):
        confidence = float(confidence)

    return jsonify({
        'sentiment': sentiment,
        'probabilities': probabilities,
        'confidence': confidence
    })

if __name__ == '__main__':
    # 在启动服务器前初始化模型
    init_model()
    app.run(host='0.0.0.0', port=5000)
