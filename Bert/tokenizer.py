from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

encoded = tokenizer("to smack of a hallmark hall of fame , with a few four letter words thrown in that are generally not heard on television")

print(encoded)

# output : 
# {'input_ids': [101, 2000, 21526, 1997, 1037, 25812, 2534, 1997, 4476, 1010, 2007, 1037, 2261, 2176, 3661, 2616, 6908, 1999, 2008, 2024, 3227, 2025, 2657, 2006, 2547, 102],
# 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

print(len(encoded['input_ids']))
# 26