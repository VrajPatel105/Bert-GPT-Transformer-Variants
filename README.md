# Transformer-Variants
Exploring different variants of Transformer

This repo covers:

1. BERT
   - Exploring data
   - tokenizer
   - Fine Tuning BERT
2. BERT vs GPT contrast
3. RoBERTa Explained
4. ALBERT Explained
5. T5 Explained

Also contains Toy dataset pretrainig implenentatiosn for MLM, CLM and seq2seq


## Key Differences at a Glance

|                       | BERT                      | RoBERTa                   | ALBERT                    | T5                              |
|-----------------------|---------------------------|---------------------------|---------------------------|---------------------------------|
| Architecture          | Encoder-only              | Encoder-only              | Encoder-only              | Encoder-Decoder                 |
| Pretraining Objective | MLM + NSP                 | MLM                       | MLM + SOP                 | Span Corruption                 |
| Positional Encoding   | Absolute learned          | Absolute learned          | Absolute learned          | Relative                        |
| Layer Norm            | Post-Norm                 | Post-Norm                 | Post-Norm                 | Pre-Norm                        |
| Core Contribution     | Bidirectional pretraining | Better training recipe    | Parameter efficiency      | Unified text-to-text framework  |
| Output Format         | Task-specific heads       | Task-specific heads       | Task-specific heads       | Always text                     |

---
