// 情感分析模块
export class SentimentAnalyzer {
    constructor() {
        this.loadingIndicator = document.getElementById('loadingIndicator');
        this.resultsContainer = document.getElementById('results');
    }

    async analyzeText(text) {
        if (!text.trim()) {
            throw new Error('请输入要分析的文本');
        }

        this.showLoading();

        try {
            const response = await this.sendAnalysisRequest(text);
            const data = await this.validateResponse(response);
            this.displayResults(data);
        } catch (error) {
            this.handleError(error);
        } finally {
            this.hideLoading();
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

        const sum = data.probabilities.reduce((a, b) => parseFloat(a) + parseFloat(b), 0);
        if (Math.abs(sum - 1) > 0.1) {
            console.warn('概率总和异常:', sum);
        }

        return data;
    }

    displayResults(data) {
        const resultHTML = this.createResultHTML(data);
        this.resultsContainer.innerHTML = resultHTML;
        this.animateResults();
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
                        <span class="value">${(data.confidence * 100).toFixed(2)}%</span>
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
        const emotions = ['积极', '中性', '消极'];
        
        return probabilities.map((prob, index) => {
            const percentage = (prob * 100).toFixed(2);
            const barClass = `progress-bar-${emotions[index].toLowerCase()}`;
            
            return `
                <div class="prob-item">
                    <span class="emotion-label">${emotions[index]}</span>
                    <div class="progress">
                        <div class="progress-bar ${barClass}"
                             style="width: ${percentage}%"
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
            setTimeout(() => {
                element.style.width = element.getAttribute('aria-valuenow') + '%';
            }, index * 100);
        });
    }

    showLoading() {
        this.loadingIndicator.classList.remove('d-none');
        this.resultsContainer.innerHTML = '';
    }

    hideLoading() {
        this.loadingIndicator.classList.add('d-none');
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