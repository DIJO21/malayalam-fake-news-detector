import torch
from transformers import BertForSequenceClassification, BertTokenizer
import os

def get_mbert_tokenizer(model_name="bert-base-multilingual-cased"):
    return BertTokenizer.from_pretrained(model_name)

def build_mbert_model(model_name="bert-base-multilingual-cased", learning_rate=2e-5):
    # Load pre-trained mBERT model for sequence classification (binary)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    return model
