from data_processor import DataProcessor
from model_evaluator import ModelEvaluator
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import torch
from model import MultiDimensionalSentimentModel, create_model_and_tokenizer, ChineseSentimentDataset
from tqdm import tqdm
import time
import datetime
import argparse
import logging
from config import Config
import os
import json
from transformers import BertTokenizer
from hyperparameter_tuning import run_hyperparameter_search

def format_time(elapsed):
    '''将秒数转换为 hh:mm:ss 格式'''
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def parse_args():
    parser = argparse.ArgumentParser(description='情感分析模型训练')
    parser.add_argument('--data_path', type=str, default=Config.DATA_PATH,
                      help='训练数据路径')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE,
                      help='批次大小')
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS,
                      help='训练轮数')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE,
                      help='学习率')
    parser.add_argument('--weight_decay', type=float, default=Config.WEIGHT_DECAY,
                      help='权重衰减')
    parser.add_argument('--model_path', type=str, default=Config.MODEL_SAVE_PATH,
                      help='模型保存路径')
    parser.add_argument('--log_dir', type=str, default=Config.LOG_DIR,
                      help='日志目录')
    return parser.parse_args()

def setup_logging(log_dir):
    if not log_dir:
        log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'train.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def main():
    args = parse_args()
    setup_logging(args.log_dir)
    logging.info("开始训练...")
    print("=== 开始模型训练 ===")
    start_time = time.time()
    
    # 使用找到的最佳超参数
    best_params = run_hyperparameter_search(
        train_dataset=train_dataset,
        valid_dataset=test_dataset,
        n_trials=5
    )
    print("\n最佳超参数:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    print("\n1. 数据处理")
    # 初始化数据处理器
    processor = DataProcessor(
        sentiment_dict_path='sample_data/sentiment_dict.json',
        stopwords_path='sample_data/stopwords.txt'
    )
    
    # 处理数据
    train_df, test_df = processor.process_data('nCoV_100k_train.labled-utf8.csv')
    
    print("\n2. 模型初始化")
    # 创建模型和tokenizer
    model, tokenizer = create_model_and_tokenizer(num_dimensions=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model = model.to(device)
    
    print("\n3. 创建数据集和加载器")
    # 创建数据集
    train_dataset = ChineseSentimentDataset(
        texts=train_df['cleaned_text'].values,
        labels=train_df['情感倾向'].values,
        tokenizer=tokenizer
    )
    
    test_dataset = ChineseSentimentDataset(
        texts=test_df['cleaned_text'].values,
        labels=test_df['情感倾向'].values,
        tokenizer=tokenizer
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"训练批次数: {len(train_loader)}")
    print(f"测试批次数: {len(test_loader)}")
    
    # 使用最佳参数创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )
    
    # 创建学习率调度器
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_training_steps * 0.1,
        num_training_steps=num_training_steps
    )
    
    print("\n4. 开始训练")
    criterion = torch.nn.CrossEntropyLoss()
    total_train_loss = 0
    best_valid_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0
        
        # 创建进度条
        progress_bar = tqdm(train_loader, desc='Training', leave=True)
        for batch_idx, batch in enumerate(progress_bar):
            # 将数据移到设备上
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(batch['input_ids'], batch['attention_mask'])
            loss = criterion(outputs, batch['labels'])
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{train_loss/(batch_idx+1):.4f}'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        valid_loss = 0
        print("\n验证中...")
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Validation'):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(batch['input_ids'], batch['attention_mask'])
                loss = criterion(outputs, batch['labels'])
                valid_loss += loss.item()
        
        avg_valid_loss = valid_loss / len(test_loader)
        
        # 打印每个epoch的统计信息
        epoch_time = format_time(time.time() - epoch_start_time)
        print(f'\nEpoch 统计:')
        print(f'  训练损失: {avg_train_loss:.4f}')
        print(f'  验证损失: {avg_valid_loss:.4f}')
        print(f'  耗时: {epoch_time}')
        
        # 保存最佳模型
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            model_path = os.path.join(Config.MODEL_SAVE_PATH, 'best_model.pt')
            torch.save(model.state_dict(), model_path)
            print("  保存新的最佳模型")
    
    print("\n5. 模型评估")
    evaluator = ModelEvaluator(model, tokenizer, device)
    metrics = evaluator.evaluate(test_loader)
    
    print("\n分类报告:")
    print(metrics['classification_report'])
    
    # 保存最终模型
    torch.save(model.state_dict(), 'final_model.pt')
    
    # 打印总训练时间
    total_time = format_time(time.time() - start_time)
    print(f"\n训练完成！总耗时: {total_time}")
    print("最终模型已保存为 'final_model.pt'")
    print("最佳模型已保存为 'best_model.pt'")

if __name__ == "__main__":
    main() 