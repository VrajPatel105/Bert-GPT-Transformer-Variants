from datasets import load_dataset

dataset = load_dataset("glue", "sst2")

# sst2 is simply movie review dataset for sentiment analysis

print(dataset)


print(dataset["train"][0])
print(dataset["train"][4230])
print(dataset["train"][230])
print(dataset["train"][430])

# Response for the above print statements
# {'sentence': 'hide new secretions from the parental units ', 'label': 0, 'idx': 0}
# {'sentence': 'to smack of a hallmark hall of fame , with a few four letter words thrown in that are generally not heard on television ', 'label': 0, 'idx': 4230}
# {'sentence': "'s at once laughable and compulsively watchable , ", 'label': 1, 'idx': 230}
# {'sentence': 'it seeks excitement in manufactured high drama ', 'label': 0, 'idx': 430}

print("Train length")
print(len(dataset['train']))
print("validatino length")
print(len(dataset['validation']))

# Train length
# 67349
# validatino length
# 872

