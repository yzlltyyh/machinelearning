from flask import Flask, render_template, request, jsonify, make_response, Response
from flask_cors import CORS
import requests

app = Flask(__name__)
# 配置CORS，允许所有域名访问
CORS(app, resources={
    r"/predict": {
        "origins": "*",
        "methods": ["POST"],
        "allow_headers": ["Content-Type"]
    }
})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        # 这里调用实际的NLP API
        response = {
            'sentiment': '积极',  # 这里替换为实际API返回的结果
            'probabilities': [0.8, 0.1, 0.1],  # 示例数据
            'confidence': 0.8
        }
        
        # 添加CORS头
        response = make_response(jsonify(response))
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def proxy_predict():
    try:
        # 转发请求到NLP服务器
        response = requests.post(
            'https://nlp.capoo.live/predict',
            json=request.get_json(),
            headers={'Content-Type': 'application/json'}
        )
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tts', methods=['GET'])
def proxy_tts():
    try:
        text = request.args.get('text', '')
        # 构建TTS API请求
        tts_url = 'https://api2.capoo.live/tts'
        params = {
            'text': text,
            'text_lang': 'zh',
            'ref_audio_path': './referaudio.wav',
            'prompt_lang': 'zh',
            'prompt_text': '监考完啦，然后希望大家都能考个好成绩，这是我最真实的想法。'
        }
        
        response = requests.get(tts_url, params=params)
        
        if response.status_code != 200:
            return jsonify({'error': 'TTS service error'}), response.status_code
            
        # 返回音频数据
        return Response(
            response.content,
            mimetype='audio/wav',
            headers={
                'Content-Disposition': 'attachment; filename=tts.wav'
            }
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=2354) 