def predict_sentiment(text, model, tokenizer, device):
    """预测单个文本的情感"""
    model.eval()
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        predictions = torch.argmax(outputs, dim=1)
    
    sentiment_map = {0: "负面", 1: "中性", 2: "正面"}
    return sentiment_map[predictions.item()] 