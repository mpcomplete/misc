from tokens import create_dataloader_v1
from transformer import *
import torch
import torch.nn as nn
import tiktoken

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

# torch.manual_seed(123)
# model = GPTModel(GPT_CONFIG_124M)
# model.to(device)
# with torch.no_grad():
#     train_loss = calc_loss_loader(train_loader, model, device)
#     val_loss = calc_loss_loader(train_loader, model, device)
# print("Training loss:", train_loss)
# print("Validation loss:", val_loss)

# 5.3 pre-training
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional: just tracks our progress.
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")
        # Optional: test our final results.
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen

# Debugging, show our loss as we progress.
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()  # Temporarily disable training mode.
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()  # Temporarily disable training mode.
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model=model, idx=encoded, max_new_tokens=50, context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

tokenizer = tiktoken.get_encoding("gpt2")
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
train_losses, val_losses, track_tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=10, eval_freq=5, eval_iter=5, start_context="Every effort moves you", tokenizer=tokenizer)

from chatty import main
#main(model)