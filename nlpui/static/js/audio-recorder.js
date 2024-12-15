class AudioRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.initFileUpload();
    }

    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(stream);
            this.audioChunks = [];

            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };

            this.mediaRecorder.start();
        } catch (err) {
            console.error('录音失败:', err);
            throw err;
        }
    }

    stopRecording() {
        return new Promise((resolve) => {
            this.mediaRecorder.onstop = () => {
                const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
                resolve(audioBlob);
            };
            this.mediaRecorder.stop();
        });
    }

    async uploadAudio(audioBlob) {
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.webm');

        try {
            const response = await fetch('/api/speech-to-text', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            return result.transcript;
        } catch (err) {
            console.error('上传音频失败:', err);
            throw err;
        }
    }

    initFileUpload() {
        const fileInput = document.getElementById('fileUpload');
        const statusBadge = document.querySelector('.status-badge');
        const messageInput = document.getElementById('inputText');

        if (fileInput) {
            fileInput.addEventListener('change', async (event) => {
                const file = event.target.files[0];
                if (!file) return;

                // 检查文件类型
                const validVideoTypes = ['video/mp4', 'video/mpeg', 'video/quicktime', 'video/avi', 'video/x-matroska'];
                const validAudioTypes = ['audio/wav', 'audio/mp3', 'audio/aac', 'audio/ogg', 'audio/flac', 'audio/aiff'];
                
                if (!validVideoTypes.includes(file.type) && !validAudioTypes.includes(file.type)) {
                    alert('不支持的文件格式。\n支持的视频格式：MP4, MPEG, MOV, AVI, MKV\n支持的音频格式：WAV, MP3, AAC, OGG, FLAC, AIFF');
                    return;
                }

                // 检查文件大小（1GB限制）
                if (file.size > 1024 * 1024 * 1024) {
                    alert('文件大小不能超过1GB');
                    return;
                }

                try {
                    statusBadge.textContent = '正在转写...';
                    messageInput.value = '正在转写...';

                    const formData = new FormData();
                    formData.append('file', file);

                    const response = await fetch('/api/speech-to-text', {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }

                    const result = await response.json();
                    displayTranscriptSegments(result);
                    statusBadge.textContent = '等待输入';
                    fileInput.value = '';

                } catch (error) {
                    console.error('文件处理失败:', error);
                    alert('文件处理失败: ' + error.message);
                    statusBadge.textContent = '等待输入';
                    fileInput.value = '';
                }
            });
        }
    }
}

// 显示转写结果的函数
function displayTranscriptSegments(data) {
    const inputText = document.getElementById('inputText');
    const segmentsContainer = document.getElementById('segmentsContainer');
    const segmentsList = segmentsContainer.querySelector('.segments-list');
    
    // 隐藏输入框，显示段落容器
    inputText.style.display = 'none';
    segmentsContainer.style.display = 'block';
    
    // 清空并填充段落列表
    segmentsList.innerHTML = '';
    data.segments.forEach((segment, index) => {
        const segmentElement = document.createElement('div');
        segmentElement.className = 'segment-item';
        segmentElement.textContent = segment;
        segmentElement.style.animationDelay = `${index * 0.1}s`;
        segmentsList.appendChild(segmentElement);
    });
    
    // 将完整文本设置到隐藏的输入框中（用于后续分析）
    inputText.value = data.transcript;
}

// 移除之前添加的重复的文件上传处理代码
window.AudioRecorder = AudioRecorder; 