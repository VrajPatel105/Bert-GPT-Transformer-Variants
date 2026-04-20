# MLM (BERT-style)

# Take a sentence, randomly select 15% of tokens
# Replace 80% with [MASK], 10% with a random token, 10% leave unchanged
# Run through a small transformer encoder
# Compute cross-entropy loss only on the masked positions
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

# add padding. lets say the max seq len is 64


# now the mask token logic
# 15% of tokens are randomly selected. Of those, 80% are replaced with a [MASK] token, 
# 10% are replaced with a random token from the entire vocabulary, and 10% are kept unchanged.

random_total_token_count = math.ceil(0.15 * len(input_sentence.split())) # this is 0.15 * 50 -> 7.5 then ceil -> 8
# print(random_total_token_count) # 15

# 15% is 15
# now 80% of 15 is mask token, 10% is random token and other 10% is kept original

eighty_percent_mask = math.floor(0.8*random_total_token_count) # 0.8 * 15 = 12 floor
ten_percent_random_token = math.ceil(0.15*random_total_token_count) # 0.15 * 15 = ceil

# print(eighty_percent_mask) 12
# print(ten_percent_random_token) 3

# print(len(worddict)) = 92

random_mask_token_position_list = []
# save original tokens before any masking
original_encoded_input = encoded_input.copy()

# unified position sampling (deduplicated)
idx_positions = random.sample(range(1, len(encoded_input) - 2), random_total_token_count)

# split positions into three groups
mask_positions = idx_positions[:eighty_percent_mask]
random_positions = idx_positions[eighty_percent_mask:eighty_percent_mask + ten_percent_random_token]
unchanged_positions = idx_positions[eighty_percent_mask + ten_percent_random_token:]

for i in mask_positions:
    encoded_input[i] = mask_id

for i in random_positions:
    encoded_input[i] = random.randint(4, len(worddict) - 1)  # skip special tokens

# unchanged positions — do nothing

# build labels
labels = [-100] * len(encoded_input)
for i in idx_positions:
    labels[i] = original_encoded_input[i]

# print(encoded_input)
# print(labels)
    
# print(random_mask_token_position_list)
# [71, 12, 37, 54, 62, 84, 26, 12, 73, 0, 6, 53]
# print(random_token_position_ten_percent_list)
# [44, 20, 42]
# print(encoded_input)
# [2, 4, 5, 6, 7, 8, 2, 10, 11, 12, 13, 14, 2, 16, 17, 18, 19, 20, 21, 22, 17, 24, 25, 26, 27, 5, 2, 29, 30, 31, 32, 33, 11, 34, 35, 8, 36, 2, 38, 39, 40, 11, 35, 42, 16, 44, 45, 46, 47, 48, 49, 50, 51, 2, 2, 54, 55, 56, 57, 58, 59, 60, 2, 62, 63, 64, 65, 66, 48, 18, 67, 2, 69, 2, 71, 72, 73, 74, 75, 76, 77, 65, 78, 79, 2, 81, 82, 83, 48, 84, 85, 8, 86, 87, 88, 11, 89, 90, 91, 3]
# print(len(encoded_input)) # 100


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


# now the model part 
vocab_size = len(worddict)
encoded_input = torch.tensor(encoded_input, dtype=torch.long)
embeddings = nn.Embedding(vocab_size,64)
l1 = nn.Linear(64, vocab_size)

embeded_input = embeddings(encoded_input)
output = l1(embeded_input)

# print(output)
# print(output.shape)

# tensor([[-0.0297, -0.3355,  0.3355,  ...,  0.4011, -0.1936,  0.1121],
#         [ 0.7970,  0.8507, -0.8082,  ..., -0.2070, -1.2305,  0.9316],
#         [ 1.1147,  0.8800,  0.7090,  ..., -1.0979, -0.1262, -0.6238],
#         ...,
#         [ 0.2233,  0.2582,  0.7141,  ...,  0.0614,  0.3974,  0.7608],
#         [ 0.2233,  0.2582,  0.7141,  ...,  0.0614,  0.3974,  0.7608],
#         [ 0.2233,  0.2582,  0.7141,  ...,  0.0614,  0.3974,  0.7608]],
#        grad_fn=<AddmmBackward0>)
# torch.Size([128, 92])

# loss

labels = torch.tensor(labels, dtype=torch.long)
criterion = nn.CrossEntropyLoss()
loss = criterion(output, labels)
print(loss)