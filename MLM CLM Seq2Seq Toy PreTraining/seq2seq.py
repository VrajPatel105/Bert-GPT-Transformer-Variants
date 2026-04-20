# Seq2Seq (T5-style)

# Encoder reads the full input (no masking, bidirectional attention)
# Decoder generates output tokens autoregressively with causal masking + cross-attention to encoder output
# Compute cross-entropy loss on the decoder's predicted tokens

