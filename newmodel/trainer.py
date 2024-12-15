import os
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)

class EmotionModelTrainer:
    def __init__(self, model_name="hfl/chinese-roberta-wwm-ext", model_path="emotion_model"):
        self.model_name = model_name
        self.model_path = model_path
        self._initialize_model()

    def _initialize_model(self):
        """初始化模型和分词器"""
        try:
            if os.path.exists(self.model_path):
                print(f"正在加载本地模型: {self.model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            else:
                print(f"本地模型不存在，加载预训练模型: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=10
                )
            self.model.eval()
            print("模型加载成功！")
        except Exception as e:
            print(f"模型加载出错: {str(e)}")
            print("加载预训练模型作为备选")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=10
            )
            self.model.eval()

    def train(self, train_dataset, eval_dataset=None):
        """训练模型"""
        training_args = TrainingArguments(
            output_dir="emotion_model",
            learning_rate=1e-5,
            per_device_train_batch_size=32,
            num_train_epochs=5,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="logs",
            save_strategy="epoch",
            evaluation_strategy="epoch" if eval_dataset else "no",
            report_to=[],
            run_name="emotion_model_training",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            max_grad_norm=1.0,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        trainer.train()
        
        print("保存最佳模型...")
        os.makedirs(self.model_path, exist_ok=True)
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)

    def get_tokenizer(self):
        """获取tokenizer用于数据处理"""
        return self.tokenizer

if __name__ == "__main__":
    from dotenv import load_dotenv
    from dataprocessor import EmotionDataProcessor
    
    # 加载环境变量
    load_dotenv()
    
    # 初始化训练器
    model_name = os.getenv("EMOTION_MODEL_NAME", "hfl/chinese-roberta-wwm-ext")
    model_path = os.getenv("EMOTION_MODEL_PATH", "emotion_model")
    trainer = EmotionModelTrainer(model_name, model_path)
    
    # 加载数据
    processor = EmotionDataProcessor()
    texts, labels = processor.load_dataset()
    
    if texts and labels:
        # 准备训练数据
        dataset = processor.prepare_training_data(texts, labels, trainer.tokenizer)
        train_test_split = processor.split_dataset(dataset)
        
        # 训练模型
        trainer.train(train_test_split['train'], train_test_split['test'])
        print("模型训练完成！")
    else:
        print("没有找到训练数据！")