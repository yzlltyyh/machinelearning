import os
import json
from datasets import Dataset

class EmotionDataProcessor:
    def __init__(self, dataset_path="dataset"):
        self.dataset_path = dataset_path
        os.makedirs(dataset_path, exist_ok=True)

    def load_dataset(self, filename=None):
        """加载数据集文件"""
        if filename is None:
            files = os.listdir(self.dataset_path)
            dataset_files = [f for f in files if f.startswith('dataset_') and f.endswith('.json')]
            if not dataset_files:
                return [], []
            filename = max(dataset_files)
            
        file_path = os.path.join(self.dataset_path, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
                
            texts = [item["text"] for item in dataset]
            labels = [item["label"] for item in dataset]
            
            print(f"成功加载数据集: {file_path}")
            print(f"数据集大小: {len(texts)}条")
            
            return texts, labels
            
        except Exception as e:
            print(f"加载数据集失败: {str(e)}")
            return [], []

    def prepare_training_data(self, texts, labels, tokenizer):
        """准备用于训练的数据集"""
        dataset = Dataset.from_dict({
            "text": texts,
            "labels": labels
        })
        
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=256
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset

    def split_dataset(self, dataset, test_size=0.1):
        """划分训练集和测试集"""
        return dataset.train_test_split(test_size=test_size)

if __name__ == "__main__":
    # 测试代码
    processor = EmotionDataProcessor()
    texts, labels = processor.load_dataset()
    print(f"加载数据集完成，共{len(texts)}条数据")