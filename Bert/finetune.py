from datasets import load_dataset
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import torch.nn as nn
from torch.utils.data import dataloader, Dataset

data = load_dataset("glue", "sst2")
bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# steps to follow  
# 1. Tokenize the full train and validation sets
# 2. Create PyTorch datasets
# 3. Set up optimizer (AdamW is standard for BERT)
# 4. Training loop — forward pass, compute loss, backward, step
# 5. Evaluate on validation set

# creating the pytorch datasets -> dataset and dataloader


def tokenize_func(data):
    return tokenizer(data['sentence'], padding="max_length", truncation=True, max_length=128)

tokenize_train = data['train'].map(tokenize_func, batched=True)
tokenize_val = data['validation'].map(tokenize_func, batched=True)

class sst2data(Dataset):
    
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
    

train_ds = sst2data(tokenize_train)
val_ds = sst2data(tokenize_val)

print(train_ds[0])