class VideoAnalyzer {
    constructor() {
        this.initializeElements();
        this.setupEventListeners();
    }

    initializeElements() {
        this.dropZone = document.getElementById('dropZone');
        this.fileInput = document.getElementById('fileUpload');
        this.videoPlayer = document.getElementById('videoPlayer');
        this.playerSection = document.querySelector('.player-section');
        this.analysisSection = document.querySelector('.analysis-section');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.resultsContainer = document.querySelector('.results-container');
    }

    setupEventListeners() {
        // 文件拖放处理
        this.dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.dropZone.classList.add('drag-over');
        });

        this.dropZone.addEventListener('dragleave', () => {
            this.dropZone.classList.remove('drag-over');
        });

        this.dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.dropZone.classList.remove('drag-over');
            const files = e.dataTransfer.files;
            if (files.length) {
                this.handleFileUpload(files[0]);
            }
        });

        // 点击上传
        this.dropZone.addEventListener('click', () => {
            this.fileInput.click();
        });

        this.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length) {
                this.handleFileUpload(e.target.files[0]);
            }
        });

        // 分析按钮
        this.analyzeBtn.addEventListener('click', () => {
            this.analyzeSentiment();
        });
    }

    async handleFileUpload(file) {
        if (!this.validateFile(file)) return;

        this.showLoading('正在处理文件...');
        
        try {
            // 显示视频预览
            if (file.type.startsWith('video/')) {
                this.showVideoPreview(file);
            }

            // 上传并处理文件
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/api/video/process', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || '处理失败');
            }

            const result = await response.json();
            console.log("Process Result:", result);  // 调试输出
            
            if (!result.segments || !result.segments.length) {
                throw new Error('未获取到有效的转写结果');
            }
            
            this.displayTranscription(result);
            
        } catch (error) {
            console.error('File Processing Error:', error);  // 调试输出
            alert('文件处理失败: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    validateFile(file) {
        const validTypes = {
            video: ['video/mp4', 'video/mpeg', 'video/quicktime', 'video/avi', 'video/x-matroska'],
            audio: ['audio/wav', 'audio/mp3', 'audio/aac', 'audio/ogg', 'audio/flac', 'audio/aiff']
        };

        if (![...validTypes.video, ...validTypes.audio].includes(file.type)) {
            alert('不支持的文件格式');
            return false;
        }

        if (file.size > 1024 * 1024 * 1024) {
            alert('文件大小不能超过1GB');
            return false;
        }

        return true;
    }

    showVideoPreview(file) {
        const videoUrl = URL.createObjectURL(file);
        this.videoPlayer.src = videoUrl;
        this.playerSection.style.display = 'block';
    }

    displayTranscription(result) {
        console.log("Displaying Transcription:", result);  // 调试输出
        this.analysisSection.style.display = 'block';
        const segmentsList = document.querySelector('.segments-list');
        segmentsList.innerHTML = '';

        // 如果没有分段，使用原文本按句号分段
        let segments = result.segments;
        if (!segments || segments.length === 0) {
            segments = result.transcript.split('。').filter(text => text.trim());
        }

        segments.forEach((text, index) => {
            const segmentElement = document.createElement('div');
            segmentElement.className = 'segment-item';
            segmentElement.innerHTML = `
                <div class="segment-text">${text.trim()}</div>
                <div class="segment-emotion" data-segment="${index}"></div>
            `;
            segmentsList.appendChild(segmentElement);
            
            // 添加动画延迟
            segmentElement.style.animationDelay = `${index * 0.1}s`;
        });

        // 显示分析按钮
        this.analyzeBtn.style.display = 'block';
    }

    async analyzeSentiment() {
        const segments = Array.from(document.querySelectorAll('.segment-item'));
        if (!segments.length) return;
        
        try {
            // 禁用分析按钮
            this.analyzeBtn.disabled = true;
            
            // 为每个段落添加加载动画
            segments.forEach(segment => {
                const emotionElement = segment.querySelector('.segment-emotion');
                emotionElement.innerHTML = '<div class="spinner-small"></div>';
            });
            
            // 逐个分析每个段落
            for (let i = 0; i < segments.length; i++) {
                const segment = segments[i];
                const textElement = segment.querySelector('.segment-text');
                const emotionElement = segment.querySelector('.segment-emotion');
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            text: textElement.textContent
                        })
                    });

                    if (!response.ok) throw new Error('分析失败');

                    const result = await response.json();
                    
                    // 更新情感图标
                    emotionElement.innerHTML = this.getEmotionIcon(result.sentiment);
                    emotionElement.title = `情感倾向: ${result.sentiment}\n置信度: ${(result.confidence * 100).toFixed(1)}%`;
                    
                    // 添加情感类名用于样式
                    segment.classList.add(`emotion-${result.sentiment.toLowerCase()}`);
                    
                    // 添加动画效果
                    emotionElement.classList.add('emotion-fade-in');
                } catch (error) {
                    console.error(`段落 ${i + 1} 分析失败:`, error);
                    emotionElement.innerHTML = this.getEmotionIcon('error');
                    emotionElement.title = '分析失败';
                }
                
                // 短暂延迟，避免请求过快
                await new Promise(resolve => setTimeout(resolve, 300));
            }
            
        } catch (error) {
            alert('情感分析失败: ' + error.message);
        } finally {
            // 重新启用分析按钮
            this.analyzeBtn.disabled = false;
        }
    }

    getEmotionIcon(emotion) {
        const icons = {
            positive: '😊',
            neutral: '😐',
            negative: '😔'
        };
        return icons[emotion] || '❓';
    }

    showLoading(message) {
        this.loadingOverlay.querySelector('.loading-text').textContent = message;
        this.loadingOverlay.style.display = 'flex';
    }

    hideLoading() {
        this.loadingOverlay.style.display = 'none';
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    new VideoAnalyzer();
}); 