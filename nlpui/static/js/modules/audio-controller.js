// 音频控制器模块
export class AudioController {
    constructor() {
        this.audioElement = document.getElementById('ttsAudio');
        this.playButton = document.querySelector('.play-button');
        this.progressBar = document.querySelector('.audio-progress-bar');
        this.progressHandle = document.querySelector('.audio-progress-handle');
        this.currentTime = document.querySelector('.current-time');
        this.duration = document.querySelector('.duration');
        this.volumeSlider = document.querySelector('.volume-slider');
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // 播放/暂停控制
        this.playButton.addEventListener('click', () => this.togglePlay());
        
        // 进度条控制
        this.audioElement.addEventListener('timeupdate', () => this.updateProgress());
        this.audioElement.addEventListener('loadedmetadata', () => this.setDuration());
        
        // 音量控制
        this.volumeSlider.addEventListener('input', (e) => this.setVolume(e.target.value));
        
        // 进度条拖动
        this.progressHandle.addEventListener('mousedown', (e) => this.startDragging(e));
    }

    togglePlay() {
        if (this.audioElement.paused) {
            this.audioElement.play();
            this.playButton.innerHTML = '<i class="fas fa-pause"></i>';
        } else {
            this.audioElement.pause();
            this.playButton.innerHTML = '<i class="fas fa-play"></i>';
        }
    }

    updateProgress() {
        const progress = (this.audioElement.currentTime / this.audioElement.duration) * 100;
        this.progressBar.style.width = `${progress}%`;
        this.progressHandle.style.left = `${progress}%`;
        this.currentTime.textContent = this.formatTime(this.audioElement.currentTime);
    }

    setDuration() {
        this.duration.textContent = this.formatTime(this.audioElement.duration);
    }

    setVolume(value) {
        this.audioElement.volume = value / 100;
    }

    startDragging(e) {
        const handleDrag = (e) => {
            const progressRect = this.progressBar.parentElement.getBoundingClientRect();
            const percent = Math.min(Math.max(0, (e.clientX - progressRect.left) / progressRect.width), 1);
            this.audioElement.currentTime = percent * this.audioElement.duration;
        };

        const stopDragging = () => {
            document.removeEventListener('mousemove', handleDrag);
            document.removeEventListener('mouseup', stopDragging);
        };

        document.addEventListener('mousemove', handleDrag);
        document.addEventListener('mouseup', stopDragging);
    }

    formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }
}