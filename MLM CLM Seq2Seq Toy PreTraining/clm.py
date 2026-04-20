# CLM (GPT-style)

# Take a sequence of tokens
# Apply a causal mask (upper triangular) so position i can only attend to positions ≤ i
# Predict the next token at every position
# Compute cross-entropy loss across all positions (shifted by 1)