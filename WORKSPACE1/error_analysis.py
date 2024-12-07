def analyze_errors(model, test_loader, tokenizer, device):
    """分析模型预测错误的案例"""
    model.eval()
    errors = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            outputs = model(input_ids, attention_mask)
            predictions = torch.argmax(outputs, dim=1)
            
            # 找出预测错误的样本
            mask = predictions != labels
            if mask.any():
                wrong_predictions = predictions[mask]
                wrong_labels = labels[mask]
                wrong_texts = tokenizer.batch_decode(input_ids[mask])
                
                for text, pred, label in zip(wrong_texts, wrong_predictions, wrong_labels):
                    errors.append({
                        'text': text,
                        'predicted': pred.item(),
                        'actual': label.item()
                    })
    
    return errors 