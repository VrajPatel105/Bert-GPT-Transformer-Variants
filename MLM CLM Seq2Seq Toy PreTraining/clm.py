# CLM (GPT-style)

# Take a sequence of tokens
# Apply a causal mask (upper triangular) so position i can only attend to positions ≤ i
# Predict the next token at every position
# Compute cross-entropy loss across all positions (shifted by 1)

# NOTE : the code in MLM and CLM is 99% the same. in CLM we are just introducing the causal mask that's all.

import numpy as np
import math
import random
import torch
import torch.nn as nn

input_sentence = 'On a quiet evening, I sat by the window, watching distant lights flicker while my thoughts wandered slowly. The city hummed softly below, and a gentle breeze carried faint music through the open air. I breathed deeply, gathering calm from the moment, promising myself to keep moving forward with hope. Stars twinkled above like scattered diamonds, whispering secrets of forgotten dreams. Leaves rustled outside, dancing in rhythm with my heartbeat. A stray cat meowed softly, seeking warmth nearby. Memories flooded in waves, bittersweet yet comforting. Tomorrow awaited with possibilities untold. I smiled faintly, embracing the serene night fully.'
# print(f"input sentence length: {len(input_sentence.split())}")


pad_id = 0
clf_id = 1
mask_id = 2
sep_id = 3

worddict = {
    "[PAD]" : pad_id,
    "[CLF]" : clf_id,
    "[MASK]" : mask_id,
    "[SEP]" : sep_id
}

def tokenize(sentence):
    sentence = sentence.split()
    counter=4
    for i in sentence:
        if i not in worddict:
            worddict[i] = counter
            counter += 1
    # sep token at the end even though we have one sentence in input
    # now i will direclty return the encoded list
    encoded_list = []
    for word in sentence:
        encoded_list.append(worddict.get(word))
    encoded_list.insert(0,clf_id)
    encoded_list.append(sep_id)
    return encoded_list

encoded_input = tokenize(input_sentence)
# print(encoded_input)


# print(tokenized_input)
# {'[CLF]': 0, 'On': 1, 'a': 2, 'quiet': 3, 'evening,': 4, 'I': 5, 'sat': 6, 'by': 7,
#  'the': 8, 'window,': 9, 'watching': 10, 'distant': 11, 'lights': 12, 'flicker': 13,
#  'while': 14, 'my': 15, 'thoughts': 16, 'wandered': 17, 'slowly.': 18, 'The': 19,
#  'city': 20, 'hummed': 21, 'softly': 22, 'below,': 23, 'and': 24, 'gentle': 25,
#  'breeze': 26, 'carried': 27, 'faint': 28, 'music': 29, 'through': 30, 'open': 31,
#  'air.': 32, 'breathed': 33, 'deeply,': 34, 'gathering': 35, 'calm': 36, 'from': 37,
#  'moment,': 38, 'promising': 39, 'myself': 40, 'to': 41, 'keep': 42, 'moving': 43,
#  'forward': 44, 'with': 45, 'hope.': 46, '[SEP]': 47}

# build labels — CLM: shift input left by 1, last position is -100
labels = encoded_input[1:] + [-100]

max_seq_len = 128
padding_len = max_seq_len - (len(input_sentence.split())+2) # the +2 is for CLF and SEP token
# print(f"padding length {padding_len}")
for i in range(padding_len):
    encoded_input.insert(len(encoded_input) + i, pad_id)
labels.extend([-100] * padding_len)
# print(encoded_input)


# [2, 4, 5, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 2, 20, 66, 2, 23, 24, 25, 26, 27, 5, 28, 29, 30, 31, 32, 33, 11, 34, 2, 2, 27, 37, 38, 39, 40, 11, 41, 42, 43,
#  44, 2, 46, 47, 48, 49, 50, 2, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 48, 18, 67, 68, 69, 70, 71, 2, 73, 74, 75, 76, 77, 65, 78, 2, 80, 2, 82, 83,
#  48, 84, 85, 8, 86, 87, 88, 11, 89, 90, 91, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# print(len(encoded_input))
# Length = 128

# Causal mask
seq_len = len(encoded_input)
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
print(causal_mask)

# now the model part 
vocab_size = len(worddict)
encoded_input = torch.tensor(encoded_input, dtype=torch.long)
embeddings = nn.Embedding(vocab_size,64)
l1 = nn.Linear(64, vocab_size)

embeded_input = embeddings(encoded_input)
output = l1(embeded_input)

# loss

labels = torch.tensor(labels, dtype=torch.long)
criterion = nn.CrossEntropyLoss()
loss = criterion(output, labels)
print(loss)
# tensor(4.8471, grad_fn=<NllLossBackward0>)