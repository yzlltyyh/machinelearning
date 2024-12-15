from flask import Flask, request, jsonify, render_template_string
import aiohttp
import asyncio
import json
import os
from functools import wraps

app = Flask(__name__)

# 修改HTML模板以反映新的多维度情感分析功能
API_DOC = """
<!DOCTYPE html>
<html>
<head>
    <title>多维度情感分析 API</title>
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
    <h1>多维度情感分析 API</h1>
    
    <div class="endpoint">
        <h2>预测接口</h2>
        <p><strong>端点：</strong> /predict</p>
        <p><strong>方法：</strong> POST</p>
        <p><strong>描述：</strong> 对输入文本进行多维度情感分析</p>
        
        <h3>请求格式：</h3>
        <pre>
{
    "text": "要分析的文本"
}
        </pre>
        
        <h3>响应格式：</h3>
        <pre>
{
    "text": "原文本",
    "scores": {
        "valence": 0.9,         // 情感效价 (-1到1)
        "arousal": 70,          // 情感唤醒度 (0-100)
        "dominance": 70,        // 情感主导性 (0-100)
        "anxiety": 1,           // 焦虑指数 (0-21)
        "depression": 1,        // 抑郁指数 (0-21)
        "stress": 1,            // 压力指数 (0-40)
        "emotional_salience": 85,    // 情感显著性 (0-100)
        "emotional_complexity": 20,   // 情感复杂度 (0-100)
        "loneliness_index": 5,       // 孤独指数 (0-100)
        "self_reflection": 10        // 自我反思 (0-100)
    }
}
        </pre>

        <h3>示例：</h3>
        <p>请求：</p>
        <pre>
curl -X POST -H "Content-Type: application/json" \\
     -d '{"text":"今天好开心"}' \\
     http://localhost:5000/predict
        </pre>
        
        <p>响应：</p>
        <pre>
{
    "text": "今天好开心",
    "scores": {
        "anxiety": 1,
        "arousal": 70,
        "depression": 1,
        "dominance": 70,
        "emotional_complexity": 20,
        "emotional_salience": 85,
        "loneliness_index": 5,
        "self_reflection": 10,
        "stress": 1,
        "valence": 0.9
    }
}
        </pre>
    </div>
</body>
</html>
"""

def async_route(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapped

class EmotionAnalyzer:
    def __init__(self, api_key: str, api_base: str = "https://api.aigclink.xyz"):
        self.api_key = api_key
        self.api_base = api_base

    async def analyze_text(self, text: str) -> dict:
        """分析文本的情感特征"""
        system_prompt = """你是一个专业的情感分析专家，精通文学表达和心理分析。请对给定文本进行深入的心理分析，返回规范的JSON格式结果，不要返回任何多余的信息和解释。分析要准确反映文本的情感特征和深层含义。"""
        
        user_prompt = f"""请分析以下文本的情感特征：
        "{text}"
        
        请按照以下JSON格式返回分析结果：
        {{
            "text": "原文本",
            "scores": {{
                "valence": 数值,  # 情感效价，范围-1-1
                "arousal": 数值,  # 情感唤醒度，范围0-100
                "dominance": 数值,  # 情感主导性，范围0-100
                "anxiety": 数值,  # 焦虑指数，范围0-21
                "depression": 数值,  # 抑郁指数，范围0-21
                "stress": 数值,  # 压力指数，范围0-40
                "emotional_salience": 数值,  # 情感显著性，范围0-100
                "emotional_complexity": 数值,  # 情感复杂度，范围0-100
                "loneliness_index": 数值,  # 孤独指数，范围0-100
                "self_reflection": 数值  # 自我反思，范围0-100
            }}
        }}"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gemini-2.0-flash-exp",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 8096
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_base}/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            ) as response:
                if response.status != 200:
                    raise Exception(f"API调用失败: {await response.text()}")
                    
                result = await response.json()
                return self._process_response(result)

    def _process_response(self, response: dict) -> dict:
        """处理API响应"""
        try:
            content = response['choices'][0]['message']['content'].strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1]
            
            return json.loads(content)
        except Exception as e:
            raise Exception(f"响应解析失败: {str(e)}")

# 初始化情感分析器
analyzer = EmotionAnalyzer(os.getenv("EMOTION_API_KEY"))

@app.route('/')
def home():
    """API文档首页"""
    return render_template_string(API_DOC)

@app.route('/predict', methods=['POST'])
@async_route
async def predict():
    """情感分析接口"""
    if not request.is_json:
        return jsonify({"error": "请求必须是JSON格式"}), 400
        
    text = request.json.get('text')
    if not text:
        return jsonify({"error": "缺少text字段"}), 400

    try:
        result = await analyzer.analyze_text(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    if not os.getenv("EMOTION_API_KEY"):
        raise ValueError("请设置环境变量 EMOTION_API_KEY")
    app.run(host='127.0.0.1', port=5000)
