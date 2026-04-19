from datasets import load_dataset
from transformers import BertForSequenceClassification, BertTokenizer
from torch.optim import AdamW
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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

# setting all the cols to tensors
# set_format makes only those columns available when somenoes looks at it. so we already dropped cols like idx, input_ids etc that we did not need
   
tokenize_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
tokenize_val.set_format("torch", columns=["input_ids", "attention_mask", "label"])

train_ds = sst2data(tokenize_train)
val_ds = sst2data(tokenize_val)

# print(train_ds[0])

# output
# {'label': tensor(0), 
# 'input_ids': tensor([  101,  5342,  2047,  3595,  8496,  2013,  1996, 18643,  3197,   102,
#             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#             0,     0,     0,     0,     0,     0,     0,     0]), 
# 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0])}


# dataloaders

train_data = DataLoader(train_ds, batch_size=32, shuffle=True)
val_data = DataLoader(val_ds, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = bert.to(device)

learning_rate = 2e-5 # 2 x 10^(-5)

optimizer = AdamW(model.parameters(), lr = learning_rate)

epochs = 3

# training

for epoch in range(epochs):

    model.train()

    for batch in train_data:
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        output = model(input_ids, attention_mask = attention_mask, labels=labels)

        loss = output.loss

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        # validation loop

    model.eval()

    correct = 0
    total = 0


    with torch.no_grad():
        for batch in val_data:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            output = model(input_ids, attention_mask = attention_mask, labels=labels)
            preds = torch.argmax(output.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)


        print(f"Epoch {epoch} Accuracy: {correct/total:.4f}")