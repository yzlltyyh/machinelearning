import os
from data_generator import EmotionDataGenerator
from data_processor import EmotionDataProcessor
from model_trainer import EmotionModelTrainer
from model_predictor import EmotionPredictor

def main():
    # 从环境变量获取配置信息
    API_KEY = os.getenv("EMOTION_API_KEY")
    if not API_KEY:
        raise ValueError("请设置环境变量 EMOTION_API_KEY")
        
    API_BASE = os.getenv("EMOTION_API_BASE", "https://api.aigclink.xyz/")
    MODEL_NAME = os.getenv("EMOTION_MODEL_NAME", "hfl/chinese-roberta-wwm-ext")
    MODEL_PATH = os.getenv("EMOTION_MODEL_PATH", "emotion_model")
    DATASET_PATH = os.getenv("EMOTION_DATASET_PATH", "dataset")

    # 初始化各个模块
    data_generator = EmotionDataGenerator(API_KEY, API_BASE, DATASET_PATH)
    data_processor = EmotionDataProcessor(DATASET_PATH)
    model_trainer = EmotionModelTrainer(MODEL_NAME, MODEL_PATH)

    # 询问是否需要训练新模型
    choice = input("是否需要训练新模型？(y/n): ")
    
    if choice.lower() == 'y':
        print("开始训练新模型流程...")
        
        # 1. 生成新的训练数据
        print("正在生成训练数据...")
        texts, labels = data_generator.generate_data(num_samples=500)  # 可调整样本数量
        
        # 2. 保存生成的数据
        data_generator.save_data(texts, labels)
        
        # 3. 加载完整数据集（包括新生成的和历史数据）
        all_texts, all_labels = data_processor.load_dataset()
        
        # 4. 准备训练数据
        tokenizer = model_trainer.get_tokenizer()
        dataset = data_processor.prepare_training_data(all_texts, all_labels, tokenizer)
        
        # 5. 划分数据集
        train_test_split = data_processor.split_dataset(dataset)
        
        # 6. 训练模型
        model_trainer.train(train_test_split['train'], train_test_split['test'])
        print("模型训练完成！")
    
    # 初始化预测器
    predictor = EmotionPredictor(model_trainer.model, model_trainer.tokenizer)
    
    # 交互式预测
    print("\n进入交互式预测模式...")
    while True:
        text = input("\n请输入要分析的文本(输入 'q' 退出): ")
        if text.lower() == 'q':
            break
        
        result = predictor.predict(text)
        predictor.format_result(result)

if __name__ == "__main__":
    main()