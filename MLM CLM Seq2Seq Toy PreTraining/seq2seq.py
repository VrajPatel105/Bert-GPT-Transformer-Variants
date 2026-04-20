# Seq2Seq (T5-style)

# Encoder reads the full input (no masking, bidirectional attention)
# Decoder generates output tokens autoregressively with causal masking + cross-attention to encoder output
# Compute cross-entropy loss on the decoder's predicted tokens


# NOTE : seq2seq is almost same as CLM, just here we also need decoder block that gets encoder's output as input
# since this is toy data, all we want to do is to compute the loss and see if the loss is being computed 
# which is why we just introduce the decoder and add the enc's output as tensor addition and then compute the final loss
import numpy as np
import math
import random
import torch
import torch.nn as nn

input_sentence = 'On a quiet evening, I sat by the window, watching distant lights flicker while my thoughts wandered slowly. The city hummed softly below, and a gentle breeze carried faint music through the open air. I breathed deeply, gathering calm from the moment, promising myself to keep moving forward with hope. Stars twinkled above like scattered diamonds, whispering secrets of forgotten dreams. Leaves rustled outside, dancing in rhythm with my heartbeat. A stray cat meowed softly, seeking warmth nearby. Memories flooded in waves, bittersweet yet comforting. Tomorrow awaited with possibilities untold. I smiled faintly, embracing the serene night fully.'
# print(f"input sentence length: {len(input_sentence.split())}")
dec_input_sentence = 'On a quiet evening, I sat by the window, watching distant lights flicker while my thoughts wandered slowly. The city hummed softly below, and a gentle breeze carried faint music through the open air. I breathed deeply, gathering calm from the moment, promising myself to keep moving forward with hope. Stars twinkled above like scattered diamonds, whispering secrets of forgotten dreams. Leaves rustled outside, dancing in rhythm with my heartbeat. A stray cat meowed softly, seeking warmth nearby. Memories flooded in waves, bittersweet yet comforting. Tomorrow awaited with possibilities untold. I smiled faintly, embracing the serene night fully.'
# keeping the input and output same since this is toy

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
decoder_input = tokenize(dec_input_sentence)


labels = encoded_input[1:] + [-100]
labels = decoder_input[1:] + [-100]

max_seq_len = 128
padding_len = max_seq_len - (len(input_sentence.split())+2) # the +2 is for CLF and SEP token
# print(f"padding length {padding_len}")
for i in range(padding_len):
    encoded_input.insert(len(encoded_input) + i, pad_id)
    decoder_input.insert(len(encoded_input) + i, pad_id)
labels.extend([-100] * padding_len)


# Causal mask
seq_len = len(encoded_input)
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
print(causal_mask)

# now the model part 
vocab_size = len(worddict)
encoded_input = torch.tensor(encoded_input, dtype=torch.long)
decoder_input = torch.tensor(decoder_input, dtype=torch.long)

# encoder 
embeddings = nn.Embedding(vocab_size,64)
l1 = nn.Linear(64, vocab_size)

# decoder
decoder_embeddings = nn.Embedding(vocab_size,64)
l2 = nn.Linear(64, vocab_size)

embeded_input = embeddings(encoded_input)
output = l1(embeded_input)

decoder_embedded = decoder_embeddings(decoder_input) + embeded_input

output2 = l2(decoder_embedded)
# loss

labels = torch.tensor(labels, dtype=torch.long)
criterion = nn.CrossEntropyLoss()
loss = criterion(output2, labels)
print(loss)
# output for seq2seq
#tensor([[1., 0., 0.,  ..., 0., 0., 0.],
#         [1., 1., 0.,  ..., 0., 0., 0.],
#         [1., 1., 1.,  ..., 0., 0., 0.],
#         ...,
#         [1., 1., 1.,  ..., 1., 0., 0.],
#         [1., 1., 1.,  ..., 1., 1., 0.],
#         [1., 1., 1.,  ..., 1., 1., 1.]])
# tensor(4.9350, grad_fn=<NllLossBackward0>)