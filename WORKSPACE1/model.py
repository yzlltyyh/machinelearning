import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import jieba
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

class MultiDimensionalSentimentModel(nn.Module):
    def __init__(self, pretrained_model_name='hfl/chinese-roberta-wwm-ext-large', num_dimensions=2):
        super().__init__()
        # 加载预训练模型
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        hidden_size = self.bert.config.hidden_size  # RoBERTa-large的hidden_size为1024
        
        # 特征提取层
        self.feature_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 多维度情感分析层
        self.dimension_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Linear(128, 1)
            ) for _ in range(num_dimensions)
        ])
        
        # 注意力机制层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=16,  # 增加注意力头数
            dropout=0.1
        )
        
        # 情感增强层
        self.sentiment_enhancement = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, input_ids, attention_mask):
        # 获取BERT输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # 使用最后一层的隐藏状态
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # 使用 [CLS] token 的表示
        cls_output = hidden_states[:, 0, :]  # [batch_size, hidden_size]
        
        # 特征提取
        features = self.feature_layer(cls_output)  # [batch_size, hidden_size]
        
        # 多维度情感分析
        dimension_scores = []
        for head in self.dimension_heads:
            score = head(features)  # [batch_size, 1]
            dimension_scores.append(score)
        
        # 合并所有维度的分数
        final_scores = torch.cat(dimension_scores, dim=1)  # [batch_size, 3]
        
        # 应用 log_softmax
        return F.log_softmax(final_scores, dim=1)  # [batch_size, 3]

class ChineseSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # BERT tokenization
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_model(model, train_loader, valid_loader, criterion, optimizer, 
                n_epochs, device, scheduler=None):
    best_valid_loss = float('inf')
    
    for epoch in range(n_epochs):
        # 训练模式
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            train_loss += loss.item()
        
        # 验证
        valid_loss = evaluate_model(model, valid_loader, criterion, device)
        
        print(f'Epoch: {epoch+1}')
        print(f'\tTrain Loss: {train_loss/len(train_loader):.3f}')
        print(f'\tValid Loss: {valid_loss:.3f}')
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pt')

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    
    return total_loss / len(data_loader)

def create_model_and_tokenizer(num_dimensions=5):
    model_name = 'hfl/chinese-roberta-wwm-ext-large'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = MultiDimensionalSentimentModel(
        pretrained_model_name=model_name,
        num_dimensions=num_dimensions
    )
    
    # 冻结部分BERT层以节省显存
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for i in range(12):  # 冻结前12层
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = False
            
    return model, tokenizer

class SentimentTrainer:
    def __init__(self):
        self.scaler = torch.amp.GradScaler('cuda')  # 更新为新的写法
        self.accumulation_steps = 4
        
    def train_step(self, batch, model, criterion):
        # 1. 混合精度训练
        with torch.amp.autocast('cuda'):  # 更新为新的写法
            # 生成对抗样本
            embeds = model.bert.embeddings(batch['input_ids'])
            delta = torch.zeros_like(embeds).uniform_(-self.eps, self.eps)
            delta.requires_grad = True
            
            # 前向传播
            outputs = model(batch['input_ids'], batch['attention_mask'])
            loss = criterion(outputs, batch['labels'])
            
            # 梯度累积
            loss = loss / self.accumulation_steps
            
        # 反向传播
        self.scaler.scale(loss).backward()
        
        if (self.step + 1) % self.accumulation_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
        return loss.item() * self.accumulation_steps
    
    def quantize_model(self, model):
        # 模型量化
        return torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )

class EnsembleModel(nn.Module):
    def __init__(self, model_configs):
        super().__init__()
        self.models = nn.ModuleList([
            MultiDimensionalSentimentModel(**config)
            for config in model_configs
        ])
        self.attention = nn.Linear(len(model_configs), 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = []
        for model in self.models:
            output = model(input_ids, attention_mask)
            outputs.append(output)
        
        # 堆叠所有模型的输出
        stacked_outputs = torch.stack(outputs, dim=0)  # [num_models, batch_size, num_classes]
        
        # 计算注意力权重
        weights = F.softmax(self.attention(stacked_outputs.permute(1, 2, 0)), dim=-1)
        
        # 加权平均
        weighted_outputs = (stacked_outputs.permute(1, 2, 0) * weights).sum(-1)
        
        return weighted_outputs