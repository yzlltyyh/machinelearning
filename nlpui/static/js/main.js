// 主应用程序
import { SpeechRecognition } from './modules/speech-recognition.js';
import { SentimentAnalyzer } from './modules/sentiment-analyzer.js';
import { TTSGenerator } from './modules/tts-generator.js';
import { AudioController } from './modules/audio-controller.js';

class App {
    constructor() {
        this.initializeComponents();
        this.setupEventListeners();
    }

    initializeComponents() {
        const textInput = document.getElementById('inputText');
        this.speechRecognition = new SpeechRecognition(textInput);
        this.sentimentAnalyzer = new SentimentAnalyzer();
        this.ttsGenerator = new TTSGenerator();
        this.audioController = new AudioController();
    }

    setupEventListeners() {
        // 语音识别控制
        document.getElementById('startVoice').addEventListener('click', () => {
            this.speechRecognition.startRecording();
        });

        document.getElementById('stopVoice').addEventListener('click', () => {
            this.speechRecognition.stopRecording();
        });

        // 情感分析
        document.getElementById('analyzeBtn').addEventListener('click', () => {
            const text = document.getElementById('inputText').value;
            this.sentimentAnalyzer.analyzeText(text);
        });

        // TTS生成
        document.getElementById('generateTTS').addEventListener('click', () => {
            const text = document.getElementById('inputText').value;
            this.ttsGenerator.generateSpeech(text);
        });
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    new App();
});