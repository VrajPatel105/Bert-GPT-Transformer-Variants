# ALBERT — A Lite BERT (Google Research, 2019)

## What is ALBERT?
ALBERT is a lighter, more parameter-efficient version of BERT. Google Research showed that BERT's size was a bottleneck — too many parameters meant slow training, high memory cost, and difficulty scaling. ALBERT reduced BERT's 110M parameters down to 12M using two architectural techniques, while maintaining competitive or better performance.

---

## Core Problem with BERT that ALBERT Solved
BERT was large and expensive. Simply making BERT bigger didn't always help and often caused degradation due to GPU/TPU memory limits and communication overhead in distributed training. ALBERT asked: can we get the same or better performance with far fewer parameters?

---

## Two Key Techniques

**1. Factorized Embedding Parameterization**
In BERT, the token embedding dimension and the hidden layer dimension are tied together at 768. This is wasteful because token embeddings are context-free — they represent a token in isolation, not in context — so they don't need to be as large as the hidden layers which do the heavy contextual reasoning.

ALBERT decouples these two dimensions. It uses a small embedding dimension (128) and projects up to the full hidden dimension (768) via a small matrix multiplication. Since the vocabulary is large (30k tokens), this saves enormous space: 30k × 768 vs 30k × 128. The projection matrix handles the rest.

**2. Cross-Layer Parameter Sharing**
BERT has 12 encoder layers, each with its own independent weights. ALBERT shares the same weights across all encoder layers — every layer is literally the same set of parameters run repeatedly. This is what drives the parameter count from 110M to 12M.

Important nuance: ALBERT is smaller in memory and parameter count, but **not faster in inference**. You still run N layers — they just share weights. Smaller model, same compute cost at inference time.

---

## Training Objective Change: NSP → SOP

**Sentence Order Prediction (SOP)** replaced NSP for the same reason RoBERTa dropped it — NSP was too easy. But where RoBERTa simply removed it, ALBERT replaced it with a harder, more meaningful task.

SOP gives the model two sentences from the *same document* in either correct or swapped order. The model predicts which order is correct. Since both sentences are from the same document, topic detection is useless — the model must learn real discourse coherence and sentence-level reasoning. This is a direct and meaningful upgrade over NSP.

---

## What ALBERT Did NOT Change
- Transformer encoder architecture — still identical to BERT
- Masked Language Modeling (MLM) as primary pretraining objective
- Bidirectional attention — still encoder-only

---

## Key Differences at a Glance

|                          | BERT                  | ALBERT                    |
|--------------------------|-----------------------|---------------------------|
| Parameters               | 110M                  | 12M                       |
| Embedding Dim            | Tied to hidden (768)  | Decoupled (128 → 768)     |
| Layer Weights            | Independent per layer | Shared across all layers  |
| Pretraining Objective 2  | NSP                   | SOP                       |
| Memory Cost              | High                  | Low                       |
| Inference Speed          | Baseline              | Same (not faster)         |

---

## The Core Insight
ALBERT proved that **parameter count and model quality are not the same thing**. Smart parameter sharing and embedding factorization let you build a model that is 9x smaller than BERT but learns richer representations. The SOP objective also showed that better pretraining tasks matter as much as architecture decisions.

---

## When to Use ALBERT over BERT
Use ALBERT when memory and parameter count are constraints — edge deployment, limited GPU memory, or when you need to run many model instances. If raw inference speed is the bottleneck, ALBERT won't help since compute cost is similar.