import optuna
from optuna.trial import Trial
import torch
from torch.utils.data import DataLoader
from model import MultiDimensionalSentimentModel
import gc
import os
import numpy as np
from sklearn.model_selection import KFold
import json
from config import Config
from tqdm import tqdm

# 设置CUDA环境变量
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 添加早停类
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

max_epochs = 3  # 定义最大训练轮数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def objective(trial: Trial, train_dataset, valid_dataset):
    try:
        # 清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # 超参数搜索空间
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 2e-5, log=True),
            'batch_size': 8,  # 固定批次大小
            'hidden_dropout': trial.suggest_float('hidden_dropout', 0.1, 0.2),
            'weight_decay': trial.suggest_float('weight_decay', 0.01, 0.03),
        }
        
        print(f"\nTrial {trial.number + 1}")
        print("Parameters:", params)
        
        # 创建模型
        model = create_model(trial)
        optimizer = create_optimizer(model, trial)
        early_stopping = EarlyStopping(patience=3)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            valid_dataset,
            batch_size=params['batch_size']
        )
        
        # 训练循环
        for epoch in range(max_epochs):
            print(f"\nEpoch {epoch + 1}/{max_epochs}")
            train_loss = train_epoch(model, train_loader, optimizer)
            val_loss = evaluate(model, val_loader)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Valid Loss: {val_loss:.4f}")
            
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
        # 清理内存
        del model, optimizer
        torch.cuda.empty_cache()
        gc.collect()
        
        return val_loss
        
    except Exception as e:
        print(f"Trial failed with error: {str(e)}")
        return 1e6

def run_hyperparameter_search(train_dataset, valid_dataset, n_trials=5):
    print("开始超参数搜索...")
    print(f"计划进行 {n_trials} 次试验")
    
    study = optuna.create_study(direction='minimize')
    try:
        study.optimize(
            lambda trial: objective(trial, train_dataset, valid_dataset),
            n_trials=n_trials,
            catch=(RuntimeError,)
        )
        
        print("\n搜索完成!<(￣︶￣)↗[GO!]")
        print("Best parameters:", study.best_params)
        print("Best validation loss:", study.best_value)
        
        # 保存最佳参数
        params_path = os.path.join(Config.RESULT_DIR, 'best_params.json')
        with open(params_path, 'w') as f:
            json.dump(study.best_params, f)
        
        return study.best_params
    except Exception as e:
        print(f"Hyperparameter search failed: {str(e)}")
        return None

def create_model(trial):
    """创建模型"""
    model = MultiDimensionalSentimentModel(
        pretrained_model_name='hfl/chinese-roberta-wwm-ext-large',
        num_dimensions=3
    )
    return model.to(device)

def create_optimizer(model, trial):
    """创建优化器"""
    return torch.optim.AdamW(
        model.parameters(),
        lr=trial.suggest_float('learning_rate', 1e-5, 2e-5, log=True),
        weight_decay=trial.suggest_float('weight_decay', 0.01, 0.03)
    )

def train_epoch(model, train_loader, optimizer):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    # 添加进度条
    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(batch['input_ids'], batch['attention_mask'])
        loss = criterion(outputs, batch['labels'])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # 更新进度条
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)

def evaluate(model, val_loader):
    """评估模型"""
    model.eval()
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    # 添加进度条
    progress_bar = tqdm(val_loader, desc='Validation', leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch['input_ids'], batch['attention_mask'])
            loss = criterion(outputs, batch['labels'])
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(val_loader)