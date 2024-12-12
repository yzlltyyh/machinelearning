// 语音识别模块
export class SpeechRecognition {
    constructor(textInputElement) {
        this.textInput = textInputElement;
        this.recognition = null;
        this.isRecording = false;
        this.initializeSpeechRecognition();
    }

    initializeSpeechRecognition() {
        if ('webkitSpeechRecognition' in window) {
            this.recognition = new webkitSpeechRecognition();
            this.setupRecognitionConfig();
        } else {
            console.error('Speech recognition is not supported in this browser');
        }
    }

    setupRecognitionConfig() {
        this.recognition.continuous = true;
        this.recognition.interimResults = true;
        this.recognition.lang = 'zh-CN';

        this.recognition.onresult = (event) => this.handleRecognitionResult(event);
        this.recognition.onerror = (event) => this.handleRecognitionError(event);
        this.recognition.onend = () => this.handleRecognitionEnd();
    }

    handleRecognitionResult(event) {
        let finalTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
            if (event.results[i].isFinal) {
                finalTranscript += event.results[i][0].transcript;
            }
        }
        if (finalTranscript) {
            this.appendTranscript(finalTranscript);
        }
    }

    handleRecognitionError(event) {
        console.error('Speech Recognition Error:', event.error);
        this.stopRecording();
        this.updateUI('error');
    }

    handleRecognitionEnd() {
        if (this.isRecording) {
            this.recognition.start();
        }
    }

    appendTranscript(text) {
        const currentText = this.textInput.value;
        this.textInput.value = currentText + text;
    }

    startRecording() {
        if (this.recognition) {
            this.recognition.start();
            this.isRecording = true;
            this.updateUI('recording');
        }
    }

    stopRecording() {
        if (this.recognition) {
            this.recognition.stop();
            this.isRecording = false;
            this.updateUI('stopped');
        }
    }

    updateUI(status) {
        const startButton = document.getElementById('startVoice');
        const stopButton = document.getElementById('stopVoice');
        const statusBadge = document.querySelector('.status-badge');

        switch (status) {
            case 'recording':
                startButton.disabled = true;
                stopButton.disabled = false;
                statusBadge.textContent = '正在录音...';
                statusBadge.className = 'status-badge recording';
                break;
            case 'stopped':
                startButton.disabled = false;
                stopButton.disabled = true;
                statusBadge.textContent = '等待输入';
                statusBadge.className = 'status-badge';
                break;
            case 'error':
                startButton.disabled = false;
                stopButton.disabled = true;
                statusBadge.textContent = '录音错误';
                statusBadge.className = 'status-badge error';
                break;
        }
    }
}