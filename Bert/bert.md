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

- **PAD token**: When sentences are shorter than the maximum length (128 in this project), the remaining positions are filled with padding tokens (id = 0). The attention mask is set to 0 for these positions so the model knows to ignore them during attention computation.  

## Pretraining Tasks

BERT was pretrained on two tasks before any fine-tuning:

- **Masked Language Modeling (MLM)**: 15% of tokens are randomly selected. Of those, 80% are replaced with a `[MASK]` token, 10% are replaced with a random token from the entire vocabulary, and 10% are kept unchanged. The model learns to predict the original token in all three cases. This is what gives BERT its deep language understanding.  

- **Next Sentence Prediction (NSP)**: BERT is given two sentences A and B and has to predict whether B actually follows A in the original text or is a random sentence. This taught BERT to understand relationships between sentences. `token_type_ids` exists specifically for this task — 0 marks tokens from sentence A and 1 marks tokens from sentence B. For downstream tasks like SST-2 that only have one sentence, NSP is irrelevant and `token_type_ids` is all zeros.  

## Classification

For classification, the CLS token's final representation is passed through a linear layer to produce logits over the number of classes.

## Results

- **Validation accuracy on SST-2 after 3 epochs**: ~92.5%


# In my own simpler words: 

BERT stands for Bidirectional Encoder Representations from Transformers, which is an NLP model that reads text in both directions at the same time unlike older models that only read left to right.

BERT comes in two versions. The smaller one is BERT-base with 110M parameters and 12 encoder blocks with a hidden size of 768. The larger one is BERT-large with 340M parameters and 24 encoder blocks with a hidden size of 1024. This project uses bert-base-uncased which means it lowercases everything before processing.

There are a few special tokens. The CLS token is added at the start of every input and since it has no word meaning of its own, it builds up a summary of the entire sequence through all the attention layers, which is why it's used for classification. The SEP token separates two sentences when the input has a pair, or just marks the end for single sentence tasks like SST-2.

For classification, the final CLS representation gets passed through a linear layer which outputs logits over the number of classes. Validation accuracy on SST-2 after 3 epochs was around 92.5%.