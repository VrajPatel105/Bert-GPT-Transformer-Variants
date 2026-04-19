# BERT vs GPT

## BERT — Bidirectional Encoder Representations from Transformer — Encoder Only
## GPT — Generative Pre-trained Transformer — Decoder Only

---

**Attention Mask**

`causal_m = torch.tril(torch.ones(tgt_len, tgt_len, device=device)).bool()`

```
causal_m --> [1 0 0 0]
             [1 1 0 0]
             [1 1 1 0]
             [1 1 1 1]
```

In the causal mask, the zeros come from applying softmax to negative infinity which drives those attention scores to zero. This mask is used in GPT's masked multi-head attention to prevent the model from looking at future tokens during training.

BERT works differently. The B in BERT stands for bidirectional, meaning it doesn't need to hide future tokens. Instead MLM masks random tokens in the input and BERT uses both left and right context to predict those masked tokens.

---

**Where Classification Happens**

BERT uses the [CLS] token at position 0. This token has no word meaning of its own so through all 12 attention layers it builds up a summary of the entire sequence. That final [CLS] vector of shape (d_model,) gets passed through a linear layer which outputs raw logits over the number of classes. Softmax is only applied at inference when you need probabilities, not during training since CrossEntropyLoss expects raw logits.

The internal classification head is simply `nn.Linear(768, num_labels)` applied to the [CLS] output.

GPT's decoder on the other hand outputs a vector at every position and passes all of them through the projection layer at once:

```python
class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.linear_layer(x), dim=-1)
```

---

**Loss Computation**

In BERT pre-training, MLM loss is computed only over the 15% masked token positions, not the entire sequence. There's no point computing loss over tokens the model wasn't asked to predict.

In BERT fine-tuning, loss is computed over just the CLS token output since that's the only position being used for classification.

In GPT, loss is computed over all target token positions since the model is predicting the next token at every single position during training.
