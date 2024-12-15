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

# TTS情感配置
TTS_CONFIGS = {
    'positive': {
        'ref_audio_path': './referaudio.wav',
        'prompt_text': '监考完啦，然后希望大家都能考个好成绩，这是我最真实的想法。'
    },
    'neutral': {
        'ref_audio_path': './referaudio.wav',
        'prompt_text': '监考完啦，然后希望大家都能考个好成绩，这是我最真实的想法。'
    },
    'negative': {
        'ref_audio_path': './referaudio.wav',
        'prompt_text': '监考完啦，然后希望大家都能考个好成绩，这是我最真实的想法。'
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        # 调用已有的proxy_predict函数来获取NLP分析结果
        nlp_response = requests.post(
            'https://nlp.capoo.live/predict',
            json={'text': text},
            headers={'Content-Type': 'application/json'}
        )
        
        if nlp_response.status_code != 200:
            return jsonify({'error': 'NLP service error'}), nlp_response.status_code
            
        result = nlp_response.json()
        
        # 添加CORS头
        response = make_response(jsonify(result))
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
        sentiment = request.args.get('sentiment', 'neutral')  # 默认使用中性
        
        # 根据情感选择配置
        if sentiment not in TTS_CONFIGS:
            sentiment = 'neutral'
        
        config = TTS_CONFIGS[sentiment]
        
        # 构建TTS API请求
        tts_url = 'https://api2.capoo.live/tts'
        params = {
            'text': text,
            'text_lang': 'zh',
            'ref_audio_path': config['ref_audio_path'],
            'prompt_lang': 'zh',
            'prompt_text': config['prompt_text']
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

@app.route('/api/speech-to-text', methods=['POST'])
def speech_to_text():
    if 'audio' not in request.files and 'file' not in request.files:
        return jsonify({'error': '没有收到音频文件'}), 400
        
    # 获取上传的文件（可能是音频录制或视频文件上传）
    upload_file = request.files.get('audio') or request.files.get('file')
    
    try:
        response = requests.post(
            'https://asr.capoo.live/api/v1/transcribe',
            files={'file': (upload_file.filename, upload_file.stream, upload_file.content_type)}
        )
        
        if response.status_code == 200:
            result = response.json()
            # 确保返回的数据包含segments字段
            if 'segments' not in result:
                # 如果没有segments，根据标点符号分割文本
                segments = [s.strip() for s in result['transcript'].replace('。', '。\n').split('\n') if s.strip()]
                result['segments'] = segments
                
            return jsonify(result)
        else:
            return jsonify({
                'error': f'转写服务返回错误: {response.status_code}'
            }), response.status_code
            
    except Exception as e:
        return jsonify({'error': f'请求转写服务时发生错误: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=2354)
