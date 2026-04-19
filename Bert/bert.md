# BERT Fine-tuning on SST-2

BERT stands for **Bidirectional Encoder Representations from Transformers**. It's a transformer-based NLP model that reads text in both directions simultaneously, which is what makes it different from earlier models like GPT that read left to right only.

## Model Variants

BERT comes in two versions:

- **BERT-base**: 110M parameters, 12 encoder layers, hidden size of 768  
- **BERT-large**: 340M parameters, 24 encoder layers, hidden size of 1024  

This project uses **`bert-base-uncased`**, meaning the model lowercases all input text before processing.

## Key Concepts

- **CLS token**: A special token added at the start of every input. It has no word meaning of its own, so through all 12 layers of self-attention it builds up a summary representation of the entire sequence. This is what gets passed to the classification head.  

- **SEP token**: Separates two sentences when the input has a pair. For single sentence tasks like SST-2, it just marks the end of the sequence.  

- **token_type_ids**: Tells the model which sentence each token belongs to (0 for sentence A, 1 for sentence B). For SST-2, everything is 0 since there's only one sentence per input.  

## Classification

For classification, the CLS token's final representation is passed through a linear layer to produce logits over the number of classes.

## Results

- **Validation accuracy on SST-2 after 3 epochs**: ~92.5%