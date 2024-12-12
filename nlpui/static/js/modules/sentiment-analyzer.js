// 情感分析模块
export class SentimentAnalyzer {
    constructor() {
        this.loadingIndicator = document.getElementById('loadingIndicator');
        this.resultsContainer = document.getElementById('results');
        this.isAnalyzing = false;
    }

    async analyzeText(text) {
        if (!text.trim()) {
            alert('请输入要分析的文本');
            return;
        }

        if (this.isAnalyzing) {
            return;
        }

        this.isAnalyzing = true;
        this.showLoading();

        try {
            const response = await this.sendAnalysisRequest(text);
            const data = await this.validateResponse(response);
            this.displayResults(data);
        } catch (error) {
            this.handleError(error);
        } finally {
            this.hideLoading();
            this.isAnalyzing = false;
        }
    }

    async sendAnalysisRequest(text) {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text })
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`分析请求失败: ${response.status}, ${errorText}`);
        }

        return response;
    }

    async validateResponse(response) {
        const data = await response.json();
        
        if (!data.probabilities || !Array.isArray(data.probabilities)) {
            throw new Error('收到无效的概率数据');
        }

        // 确保所有概率都是有效的数字
        data.probabilities = data.probabilities.map(prob => {
            const value = parseFloat(prob);
            return isNaN(value) ? 0 : value;
        });

        // 检查是否只有两个概率值
        if (data.probabilities.length !== 2) {
            console.warn('概率数组长度异常:', data.probabilities.length);
            // 如果数据异常，使用默认值
            data.probabilities = [1, 0];
        }

        const sum = data.probabilities.reduce((a, b) => a + b, 0);
        if (Math.abs(sum - 1) > 0.1) {
            console.warn('概率总和异常:', sum);
            // 归一化概率值
            data.probabilities = data.probabilities.map(prob => prob / sum);
        }

        return data;
    }

    displayResults(data) {
        const resultHTML = this.createResultHTML(data);
        this.resultsContainer.innerHTML = resultHTML;
        // 延迟一帧执行动画，确保DOM更新完成
        requestAnimationFrame(() => this.animateResults());
    }

    createResultHTML(data) {
        return `
            <div class="result-wrapper">
                <div class="sentiment-section">
                    <div class="sentiment-item">
                        <span class="label">情感倾向</span>
                        <span class="value">${data.sentiment}</span>
                    </div>
                    <div class="sentiment-item">
                        <span class="label">置信度</span>
                        <span class="value">${(data.confidence * 100).toFixed(1)}%</span>
                    </div>
                </div>
                <div class="probability-section">
                    <h4>情感概率分布</h4>
                    <div class="probability-bars">
                        ${this.createProbabilityBars(data.probabilities)}
                    </div>
                </div>
            </div>
        `;
    }

    createProbabilityBars(probabilities) {
        // 确保emotions的顺序与后端返回的概率数组顺序一致
        const emotions = ['积极', '中性', '消极'];
        
        return probabilities.map((prob, index) => {
            const percentage = Math.max(0, Math.min(100, prob * 100)).toFixed(1);
            const barClass = `progress-bar-${emotions[index].toLowerCase()}`;
            
            return `
                <div class="prob-item">
                    <span class="emotion-label">${emotions[index]}</span>
                    <div class="progress">
                        <div class="progress-bar ${barClass}"
                             style="width: 0%"
                             role="progressbar"
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

    animateResults() {
        const elements = this.resultsContainer.querySelectorAll('.progress-bar');
        elements.forEach((element, index) => {
            const percentage = element.getAttribute('aria-valuenow');
            setTimeout(() => {
                element.style.transition = 'width 0.6s ease-in-out';
                element.style.width = `${percentage}%`;
            }, index * 150);
        });
    }

    showLoading() {
        this.loadingIndicator.style.display = 'flex';
        this.resultsContainer.innerHTML = '';
    }

    hideLoading() {
        this.loadingIndicator.style.display = 'none';
    }

    handleError(error) {
        console.error('分析错误:', error);
        this.resultsContainer.innerHTML = `
            <div class="alert alert-danger" role="alert">
                <h5>分析过程中发生错误：</h5>
                <p>${error.message}</p>
            </div>
        `;
    }
}
