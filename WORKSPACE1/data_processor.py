import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
import re
import json
from imblearn.over_sampling import SMOTE
from collections import Counter

class DataProcessor:
    def __init__(self, sentiment_dict_path=None, stopwords_path=None):
        # 加载情感词典和停用词
        self.sentiment_dict = self._load_sentiment_dict(sentiment_dict_path) if sentiment_dict_path else None
        self.stopwords = self._load_stopwords(stopwords_path) if stopwords_path else None
        
    def _load_sentiment_dict(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def _load_stopwords(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return set([line.strip() for line in f])
    
    def clean_text(self, text):
        if pd.isna(text):  # 处理空值
            return ""
        # 清理文本
        text = str(text)  # 确保是字符串
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def process_data(self, data_path, test_size=0.2, balance_data=True):
        # 读取数据
        df = pd.read_csv(data_path)
        print(f"1. 原始数据量: {len(df)}")
        print("原始标签分布:")
        print(df['label'].value_counts())
        
        # 使用正确的列名
        text_column = 'review'
        label_column = 'label'
        
        # 数据清理
        # 1. 确保标签为数值类型
        df[label_column] = pd.to_numeric(df[label_column], errors='coerce')
        print(f"\n2. 转换为数值后的数据量: {len(df)}")
        
        # 2. 清理文本
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        df = df[df['cleaned_text'].str.len() > 0]
        print(f"\n3. 文本清理后的数据量: {len(df)}")
        
        # 划分训练集和测试集
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42,
            stratify=df[label_column]
        )
        
        if balance_data:
            print("\n执行数据平衡...")
            print("平衡前的训练集标签分布:")
            print(train_df[label_column].value_counts())
            
            # 获取少数类的样本数
            min_class_count = train_df[label_column].value_counts().min()
            
            # 对多数类进行下采样
            df_majority = train_df[train_df[label_column] == 1]
            df_minority = train_df[train_df[label_column] == 0]
            
            # 随机采样多数类，使其数量与少数类相同
            df_majority_downsampled = df_majority.sample(n=len(df_minority), random_state=42)
            
            # 合并下采样后的多数类和少数类
            train_df = pd.concat([df_majority_downsampled, df_minority])
            
            print("\n平衡后的训练集标签分布:")
            print(train_df[label_column].value_counts())
        
        print(f"\n最终处理结果:")
        print(f"训练集大小: {len(train_df)}")
        print(f"测试集大小: {len(test_df)}")
        print("\n最终标签分布:")
        print(train_df[label_column].value_counts())
        
        return train_df, test_df

class DataAugmenter:
    def __init__(self):
        self.nlp = None  # 按需加载
        
    def eda_augment(self, text, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1):
        """EDA数据增强"""
        augmented = []
        words = list(jieba.cut(text))
        num_words = len(words)
        
        # 同义词替换
        n_sr = max(1, int(alpha_sr * num_words))
        augmented.append(self.synonym_replacement(words, n_sr))
        
        # 随机插入
        n_ri = max(1, int(alpha_ri * num_words))
        augmented.append(self.random_insertion(words, n_ri))
        
        # 随机交换
        n_rs = max(1, int(alpha_rs * num_words))
        augmented.append(self.random_swap(words, n_rs))
        
        # 随机删除
        augmented.append(self.random_deletion(words, p_rd))
        
        return augmented
    
    def synonym_replacement(self, words, n):
        """同义词替换"""
        # 实现同义词替换逻辑
        return "".join(words)
    
    def random_insertion(self, words, n):
        """随机插入"""
        # 实现随机插入逻辑
        return "".join(words)
    
    def random_swap(self, words, n):
        """随机交换"""
        # 实现随机交换逻辑
        return "".join(words)
    
    def random_deletion(self, words, p):
        """随机删除"""
        # 实现随机删除逻辑
        return "".join(words)

class FeatureEngineer:
    def __init__(self):
        # 添加基本的正面和负面词汇
        self.positive_words = set(['好', '优秀', '棒', '赞', '喜欢', '开心', '快乐'])
        self.negative_words = set(['坏', '差', '糟', '烂', '讨厌', '生气', '难过'])
    
    def get_linguistic_features(self, text):
        """获取语言学特征"""
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'avg_word_length': sum(len(word) for word in text.split()) / len(text.split()) if text else 0
        }
        return features
    
    def get_semantic_features(self, text):
        """获取语义特征"""
        features = {
            'contains_emoji': bool(re.search(r'[\U0001F300-\U0001F9FF]', text)),
            'contains_punctuation': bool(re.search(r'[,.!?;:]', text))
        }
        return features
    
    def get_sentiment_features(self, text):
        """获取情感特征"""
        features = {
            'positive_words': len([w for w in text.split() if w in self.positive_words]),
            'negative_words': len([w for w in text.split() if w in self.negative_words])
        }
        return features
    
    def get_contextual_features(self, text):
        """获取上下文特征"""
        features = {
            'sentence_count': len(text.split('。')),
            'has_negation': bool(re.search(r'不|没|否', text))
        }
        return features