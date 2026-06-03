from tokens import create_dataloader_v1
from transformer import *
import torch
import torch.nn as nn

GPT_CONFIG_124M["context_length"] = 256  # be gentle on our laptop

torch.manual_seed(123)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
train_ratio = 0.90
split_idx = int(train_ratio * len(raw_text))
train_data = raw_text[:split_idx]
val_data = raw_text[split_idx:]

train_loader, _ = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0)

val_loader, _ = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0)

# print("Train loader:")
# for x,y in train_loader:
#     print(x.shape, y.shape)

# print("Validation loader:")
# for x,y in val_loader:
#     print(x.shape, y.shape)

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    # input_batch.shape and target_batch.shape = 2x256 (2 batches, 256 tokens in context window)
    # logits.shape = 2x256x50257 (2 batches, 256 tokens in context window, 50257-length embedding vector)
    # Want to maximize logits[batch][token_position][target_token] where target_token = our predicted next token
    #   (e.g. target_token = target_batch[batch][token_position])
    # cross_entropy takes the logits, turns them into probabilities (via softmax), grabs the computed probability
    # of the target_token, and takes its negative log. Ideal case is loss = -log(1.0) = 0, minimal loss, maximal
    # probability. Bad case is e.g loss = -log(0.0001) = 9.21, high loss, low probability.
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    num_batches = min(num_batches or len(data_loader), len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches: break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(train_loader, model, device)
print("Training loss:", train_loss)
print("Validation loss:", val_loss)