import tiktoken

# tokenizer = tiktoken.get_encoding("gpt2")
# text = (
#     "Hello, do you like tea? <|endoftext|> In the sunlit terraces "
#     "of someunknownPlace."
# )
# ints = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# print(ints)
# print(tokenizer.decode(ints))

import torch
from torch.utils.data import Dataset, DataLoader
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i+max_length]
            target_chunk = token_ids[i+1 : i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

vocab_size = 0
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.max_token_value
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader, vocab_size

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
#enc_text = tokenizer.encode(raw_text)
#print(len(enc_text))

# Load the tokenized data as inputs and targets. The tensors will be 'batch_size' lists of size 'context_length'.
# Each cell is a token id. The targets are the inputs, shifted by 1, with the intention that target[X] is the
# word that follows input[0:X]. GPT is gonna predict target[X] after seeing input[0:X].

context_length = 4
dataloader, vocab_size = create_dataloader_v1(raw_text, batch_size=8, max_length=context_length, stride=context_length, shuffle=False)
#batches = iter(dataloader)
#print(next(batches))

# Now convert each cell containing a token_id (which is an index into our vocab of 'vocab_size') to an
# embedding vector of length 'output_dim'
output_dim = 256

torch.manual_seed(123)
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
#print(vocab_size)
#print(embedding_layer.weight)
#print(embedding_layer(torch.tensor([1, 0, 0, 1])))

inputs, targets = next(iter(dataloader)) # first batch
token_embeddings = token_embedding_layer(inputs)
#print("Token IDS:", inputs.shape, "\n", inputs)  # 8x4
#print(embeddings.shape)  # 8x4x256

# Add positional encodings to the embeddings.
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
#print(pos_embeddings.shape)  # 8x4
input_embeddings = token_embeddings + pos_embeddings
# print(input_embeddings.shape)  # 8x4x256
# print(token_embeddings)
# print(input_embeddings)