from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from config import Config

class ModelEvaluator:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.sentiment_map = {0: "负面", 1: "中性", 2: "正面"}
        
    def evaluate(self, test_loader):
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']
                
                outputs = self.model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return self.calculate_metrics(all_labels, all_predictions, all_probs)
    
    def calculate_metrics(self, true_labels, predictions, probabilities):
        # 基础指标
        report = classification_report(true_labels, predictions, output_dict=True)
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        # 计算每个类别的准确率、召回率、F1值
        metrics = {
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'accuracy': (predictions == true_labels).mean(),
            'class_probabilities': np.mean(probabilities, axis=0)
        }
        
        # 绘制混淆矩阵
        self.plot_confusion_matrix(conf_matrix)
        
        # 绘制ROC曲线
        self.plot_roc_curves(true_labels, probabilities)
        
        return metrics
    
    def plot_confusion_matrix(self, conf_matrix):
        save_path = os.path.join(Config.RESULT_DIR, 'confusion_matrix.png')
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d',
            cmap='Blues',
            xticklabels=list(self.sentiment_map.values()),
            yticklabels=list(self.sentiment_map.values())
        )
        plt.title('混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.savefig(save_path)
        plt.close()
    
    def plot_roc_curves(self, true_labels, probabilities):
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        
        # 将标签二值化
        y_bin = label_binarize(true_labels, classes=[0, 1, 2])
        n_classes = 3
        
        plt.figure(figsize=(10, 8))
        
        # 计算每个类别的ROC曲线
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], probabilities[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{self.sentiment_map[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正例率')
        plt.ylabel('真正例率')
        plt.title('ROC曲线')
        plt.legend(loc="lower right")
        plt.savefig('roc_curves.png')
        plt.close()
    
    def analyze_errors(self, test_loader):
        """分析预测错误的案例"""
        self.model.eval()
        errors = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']
                texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                
                outputs = self.model(input_ids, attention_mask)
                predictions = torch.argmax(outputs, dim=1)
                
                # 找出预测错误的样本
                mask = predictions.cpu() != labels
                if mask.any():
                    wrong_predictions = predictions[mask]
                    wrong_labels = labels[mask]
                    wrong_texts = [texts[i] for i, m in enumerate(mask) if m]
                    
                    for text, pred, label in zip(wrong_texts, wrong_predictions, wrong_labels):
                        errors.append({
                            'text': text,
                            'predicted': self.sentiment_map[pred.item()],
                            'actual': self.sentiment_map[label.item()]
                        })
        
        # 保存错误分析结果
        df = pd.DataFrame(errors)
        df.to_csv('error_analysis.csv', index=False)
        return errors
    
    def predict_text(self, text):
        self.model.eval()
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            scores = torch.sigmoid(outputs).cpu().numpy()[0]
            
        return {
            'dimension_scores': scores,
            'predicted_class': np.argmax(scores)
        } 