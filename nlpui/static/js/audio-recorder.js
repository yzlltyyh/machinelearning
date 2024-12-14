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
            return result.text;
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
                    messageInput.value = result.text;
                    statusBadge.textContent = '等待输入';

                    // 清除文件选择，允许重复上传相同文件
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

// 导出供其他模块使用
window.AudioRecorder = AudioRecorder; 