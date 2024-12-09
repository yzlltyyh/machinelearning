from flask import Flask, request, jsonify, render_template_string
import torch
from transformers import BertTokenizer
from model import MultiDimensionalSentimentModel
from predict import predict_sentiment

# 在全局范围初始化 Flask app
app = Flask(__name__)

# 创建一个全局变量来存储模型和tokenizer
global_model = None
global_tokenizer = None

# 添加HTML模板
API_DOC = """
<!DOCTYPE html>
<html>
<head>
    <title>情感分析 API 文档</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px;
        }
        pre { 
            background-color: #f5f5f5; 
            padding: 15px; 
            border-radius: 5px;
        }
        .endpoint { 
            margin-top: 20px; 
            padding: 10px;
            border-left: 3px solid #2196F3;
        }
    </style>
</head>
<body>
    <h1>情感分析 API 文档</h1>
    
    <div class="endpoint">
        <h2>预测接口</h2>
        <p><strong>端点：</strong> /predict</p>
        <p><strong>方法：</strong> POST</p>
        <p><strong>描述：</strong> 对输入的中文文本进行情感分析</p>
        
        <h3>请求格式：</h3>
        <pre>
{
    "text": "要分析的文本"
}
        </pre>
        
        <h3>响应格式：</h3>
        <pre>
{
    "sentiment": "情感类别",
    "probabilities": [类别1概率, 类别2概率, ...],
    "confidence": 置信度
}
        </pre>
        
        <h3>示例：</h3>
        <p>使用 curl 发送请求：</p>
        <pre>
curl -X POST -H "Content-Type: application/json" \
     -d '{"text":"今天天气真好"}' \
     http://localhost:5000/predict
        </pre>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    """API文档首页"""
    return render_template_string(API_DOC)

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
