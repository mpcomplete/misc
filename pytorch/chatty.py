# Chapter 4

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

import torch
import torch.nn as nn
from transformer import *

import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

def generate_and_stream(model, prompt, context_size):
    encoded = tokenizer.encode(prompt)
    idx = torch.tensor(encoded).unsqueeze(0).to(device)
    print(prompt, end="", flush=True)

    try:
        while True:
            idx = generate(
                model=model,
                idx=idx,
                max_new_tokens=1,
                context_size=context_size,
                top_k=25,
                temperature=1.4)
            new_token = idx[0][-1]
            decoded = tokenizer.decode([new_token])
            print(decoded, end="", flush=True)

            if new_token == tokenizer.eot_token:
                break
    except KeyboardInterrupt:
        print("\n[Interrupted]")

def main(model, context_size):
    print("Interactive LLM - Enter a prompt (Ctrl+C to exit)")
    while True:
        try:
            prompt = input("\n> ")
            if prompt:
                generate_and_stream(model, prompt, context_size)
        except KeyboardInterrupt:
            print("\nExiting.")
            break

if __name__ == "__main__":
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    model.eval()

    main(model, GPT_CONFIG_124M["context_length"])