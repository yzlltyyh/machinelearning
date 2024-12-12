let recognition = null;
let isRecording = false;

// 初始化语音识别
function initSpeechRecognition() {
    if ('webkitSpeechRecognition' in window) {
        recognition = new webkitSpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'zh-CN';

        recognition.onresult = function(event) {
            let finalTranscript = '';
            for (let i = event.resultIndex; i < event.results.length; i++) {
                if (event.results[i].isFinal) {
                    finalTranscript += event.results[i][0].transcript;
                }
            }
            if (finalTranscript) {
                document.getElementById('inputText').value += finalTranscript;
            }
        };

        recognition.onerror = function(event) {
            console.error('语音识别错误:', event.error);
            stopRecording();
        };
    }
}

// 开始录音
function startRecording() {
    if (recognition) {
        recognition.start();
        isRecording = true;
        document.getElementById('startVoice').disabled = true;
        document.getElementById('stopVoice').disabled = false;
    }
}

// 停止录音
function stopRecording() {
    if (recognition) {
        recognition.stop();
        isRecording = false;
        document.getElementById('startVoice').disabled = false;
        document.getElementById('stopVoice').disabled = true;
    }
}

// 分析文本
async function analyzeText() {
    const text = document.getElementById('inputText').value;
    if (!text.trim()) {
        alert('请输入要分析的文本');
        return;
    }

    // 显示加载指示器
    document.getElementById('loadingIndicator').classList.remove('d-none');
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';

    try {
        console.log('Sending request to:', 'https://nlp.capoo.live/predict');
        console.log('Request payload:', { text: text });
        
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
        }

        const data = await response.json();
        console.log('Response data:', data);
        
        // 验证概率数据
        if (!data.probabilities || !Array.isArray(data.probabilities)) {
            throw new Error('Invalid probability data received');
        }
        
        // 确保概率值的总和接近1
        const sum = data.probabilities.reduce((a, b) => parseFloat(a) + parseFloat(b), 0);
        console.log('Probability sum:', sum);
        
        // 创建结果显示HTML
        const resultHTML = `
            <div class="analysis-result">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">分析结果</h5>
                        <p class="card-text">
                            <strong>情感类别：</strong> ${data.sentiment}<br>
                            <strong>置信度：</strong> ${(data.confidence * 100).toFixed(2)}%
                        </p>
                        <div class="probabilities-container">
                            <h6>情感概率分布：</h6>
                            <div class="progress-bars">
                                ${createProbabilityBars(data.probabilities)}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        resultsDiv.innerHTML = resultHTML;
    } catch (error) {
        console.error('Error details:', error);
        resultsDiv.innerHTML = `
            <div class="alert alert-danger" role="alert">
                <h5>分析过程中发生错误：</h5>
                <p>${error.message}</p>
                <small>请检查控制台获取更多信息</small>
            </div>
        `;
    } finally {
        document.getElementById('loadingIndicator').classList.add('d-none');
    }
}

// 创建概率条形图的辅助函数
function createProbabilityBars(probabilities) {
    // 打印接收到的数据，用于调试
    console.log('Received probabilities:', probabilities);
    
    // 确保probabilities是数组，如果不是则转换
    let probs = Array.isArray(probabilities) ? probabilities : Object.values(probabilities);
    
    // 确保所有值都是数字，并且在0-1之间
    probs = probs.map(prob => {
        const value = parseFloat(prob);
        // 如果值大于1，假设它已经是百分比形式
        return value > 1 ? value / 100 : value;
    });
    
    console.log('Processed probabilities:', probs);
    
    const emotions = ['积极', '中性', '消极']; // 根据实际API返回的顺序调整
    
    return probs.map((prob, index) => {
        // 计算���分比，确保是有效数字
        const percentage = (prob * 100).toFixed(2);
        
        // 调试输出
        console.log(`${emotions[index]}: ${percentage}%`);
        
        return `
            <div class="prob-item">
                <span class="emotion-label">${emotions[index]}</span>
                <div class="progress">
                    <div class="progress-bar" role="progressbar" 
                         style="width: ${percentage}%"
                         aria-valuenow="${percentage}" 
                         aria-valuemin="0" 
                         aria-valuemax="100">
                        ${percentage}%
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

// 生成并播放TTS
async function generateTTS() {
    const text = document.getElementById('inputText').value;
    if (!text.trim()) {
        alert('请输入要转换的文本');
        return;
    }

    try {
        // 显示加载状态
        const button = document.getElementById('generateTTS');
        const originalText = button.innerHTML;
        button.disabled = true;
        button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 生成中...';

        // 构建URL（包含所有参数）
        const url = `/api/tts?text=${encodeURIComponent(text)}`;
        
        // 获取音频数据
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error('TTS生成失败');
        }

        // 获取音频blob
        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);

        // 设置音频源并显示播放器
        const audio = document.getElementById('ttsAudio');
        audio.src = audioUrl;
        
        // 显示音频区域
        document.querySelector('.audio-section').style.display = 'block';

    } catch (error) {
        console.error('TTS Error:', error);
        alert('生成语音时发生错误：' + error.message);
    } finally {
        // 恢复按钮状态
        const button = document.getElementById('generateTTS');
        button.disabled = false;
        button.innerHTML = '<i class="fas fa-file-audio"></i> 生成语音';
    }
}

// 事件监听器
document.addEventListener('DOMContentLoaded', function() {
    initSpeechRecognition();
    
    document.getElementById('startVoice').addEventListener('click', startRecording);
    document.getElementById('stopVoice').addEventListener('click', stopRecording);
    document.getElementById('analyzeBtn').addEventListener('click', analyzeText);
    document.getElementById('generateTTS').addEventListener('click', generateTTS);
}); 