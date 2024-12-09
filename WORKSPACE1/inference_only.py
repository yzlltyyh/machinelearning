import torch
from transformers import BertTokenizer
from model import MultiDimensionalSentimentModel
from predict import predict_sentiment
import time
import argparse

def load_model(model_path):
    model = MultiDimensionalSentimentModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    return model

def main():
    parser = argparse.ArgumentParser(description='情感推理')
    parser.add_argument('text', type=str, help='需要预测情感的文本')
    parser.add_argument('--model-path', type=str, default='WORKSPACE1/best_model.pt', help='模型保存路径')
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = load_model(args.model_path)
    device = torch.device('cpu')
    model.to(device)

    start_time = time.time()
    sentiment, probabilities, confidence = predict_sentiment(args.text, model, tokenizer, device)
    end_time = time.time()
    
    inference_time = end_time - start_time
    print('预测情感:', sentiment)
    print('推理时间: {:.4f}秒'.format(inference_time))
    print('详细概率输出:', probabilities)
    print('置信度:', confidence)

if __name__ == '__main__':
    main()
