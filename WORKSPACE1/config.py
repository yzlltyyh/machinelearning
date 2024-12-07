import os

class Config:
    # 基础路径
    BASE_PATH = "/content/drive/MyDrive/hotel_sentiment"
    
    # 数据相关
    DATA_PATH = '/content/ChnSentiCorp_htl_all.csv'
    SENTIMENT_DICT_PATH = os.path.join(BASE_PATH, 'sample_data/sentiment_dict.json')
    STOPWORDS_PATH = os.path.join(BASE_PATH, 'sample_data/stopwords.txt')
    
    # 模型相关
    MODEL_NAME = 'hfl/chinese-roberta-wwm-ext-large'
    MAX_LENGTH = 512
    NUM_LABELS = 2
    
    # 训练相关
    BATCH_SIZE = 8
    NUM_EPOCHS = 3
    LEARNING_RATE = 1.7121714747471662e-05
    WEIGHT_DECAY = 0.011653956691056129
    
    # 路径相关
    MODEL_SAVE_PATH = os.path.join(BASE_PATH, 'models')
    LOG_DIR = os.path.join(BASE_PATH, 'logs')
    RESULT_DIR = os.path.join(BASE_PATH, 'results')
    
    # 添加新的配置项
    SEED = 42  # 随机种子
    EARLY_STOPPING_PATIENCE = 3  # 早停耐心值
    GRADIENT_CLIP_VALUE = 1.0  # 梯度裁剪值
    WARMUP_RATIO = 0.1  # 预热比例
    
    # 模型配置
    HIDDEN_DROPOUT = 0.14871350999617416
    ATTENTION_HEADS = 16
    HIDDEN_DIM = 1024