// 文字转语音模块
export class TTSGenerator {
    constructor() {
        this.audioSection = document.querySelector('.audio-section');
        this.audioElement = document.getElementById('ttsAudio');
        this.generateButton = document.getElementById('generateTTS');
    }

    async generateSpeech(text) {
        if (!text.trim()) {
            throw new Error('请输入要转换的文本');
        }

        try {
            this.showGenerating();
            const audioUrl = await this.requestTTS(text);
            this.updateAudioPlayer(audioUrl);
            this.showAudioPlayer();
        } catch (error) {
            this.handleError(error);
        } finally {
            this.resetGenerateButton();
        }
    }

    async requestTTS(text) {
        const url = `/api/tts?text=${encodeURIComponent(text)}`;
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error('语音生成失败');
        }

        const audioBlob = await response.blob();
        return URL.createObjectURL(audioBlob);
    }

    updateAudioPlayer(audioUrl) {
        this.audioElement.src = audioUrl;
        this.audioElement.load();
    }

    showGenerating() {
        this.generateButton.disabled = true;
        this.generateButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 生成中...';
    }

    resetGenerateButton() {
        this.generateButton.disabled = false;
        this.generateButton.innerHTML = '<i class="fas fa-file-audio"></i> 生成语音';
    }

    showAudioPlayer() {
        this.audioSection.style.display = 'block';
        this.audioSection.classList.add('animate-fade-in');
    }

    handleError(error) {
        console.error('TTS Error:', error);
        alert('生成语音时发生错误：' + error.message);
    }
}