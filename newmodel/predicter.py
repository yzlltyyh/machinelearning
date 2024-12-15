import torch

class EmotionPredictor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def predict(self, text):
        """对文本进行情感预测"""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )
            
            # 将输入数据移动到相应设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.sigmoid(outputs.logits)
            
            # 将预测结果移回 CPU
            predictions = predictions.cpu()
            
            emotion_map = {
                "valence": (predictions[0][0].item() * 2) - 1,  # 转换到 [-1, 1] 范围
                "arousal": predictions[0][1].item() * 100,
                "dominance": predictions[0][2].item() * 100,
                "anxiety": predictions[0][3].item() * 21,
                "depression": predictions[0][4].item() * 21,
                "stress": predictions[0][5].item() * 40,
                "emotional_salience": predictions[0][6].item() * 100,
                "emotional_complexity": predictions[0][7].item() * 100,
                "loneliness_index": predictions[0][8].item() * 100,
                "self_reflection": predictions[0][9].item() * 100
            }
            
            return {
                "code": 200,
                "message": "success",
                "data": emotion_map
            }
            
        except Exception as e:
            return {
                "code": 500,
                "message": f"预测失败: {str(e)}",
                "data": None
            }

    def format_result(self, result):
        """格式化并打印预测结果"""
        if result["code"] != 200:
            print(f"错误: {result['message']}")
            return
            
        data = result["data"]
        print("\n分析结果:")
        print("-" * 50)
        
        print("\nPAD情感维度:")
        print(f"情感效价 (Valence): {data['valence']:.2f} [-1到1]")
        print(f"情感唤醒度 (Arousal): {data['arousal']:.2f} [0-100]")
        print(f"控制感 (Dominance): {data['dominance']:.2f} [0-100]")
        
        print("\n临床评估维度:")
        print(f"焦虑水平 (Anxiety): {data['anxiety']:.2f} [0-21]")
        print(f"抑郁水平 (Depression): {data['depression']:.2f} [0-21]")
        print(f"压力水平 (Stress): {data['stress']:.2f} [0-40]")
        
        print("\n情绪特征维度:")
        print(f"情绪显著性 (Emotional Salience): {data['emotional_salience']:.2f} [0-100]")
        print(f"情绪复杂性 (Emotional Complexity): {data['emotional_complexity']:.2f} [0-100]")
        
        print("\n心理认知维度:")
        print(f"孤独指数 (Loneliness Index): {data['loneliness_index']:.2f} [0-100]")
        print(f"自我认知程度 (Self Reflection): {data['self_reflection']:.2f} [0-100]")
        
        print("-" * 50)

if __name__ == "__main__":
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import os
    
    # 加载模型和分词器
    model_path = os.getenv("EMOTION_MODEL_PATH", "emotion_model")
    if os.path.exists(model_path):
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        model_name = "hfl/chinese-roberta-wwm-ext"
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=10)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 创建预测器实例
    predictor = EmotionPredictor(model, tokenizer)
    
    # 测试预测
    test_text = "今天是个好天气，阳光明媚，心情愉快。"
    result = predictor.predict(test_text)
    predictor.format_result(result)