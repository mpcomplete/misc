# Ch3
import torch
import torch.nn as nn

if torch.backends.mps.is_available():
    device = torch.device("mps")   # Apple Silicon GPU (Metal)
elif torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU
else:
    device = torch.device("cpu")   # CPU fallback

print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")
batch_size = 8
context_len = 1024
embed_dim = 768
embeddings = torch.randn((batch_size, context_len, embed_dim), device=device)

# 3.6.2
# Parallelize MultiHeadAttentionWrapper by stitching the multiple heads into one giant matrix.
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert(d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        # keys.shape = (b, num_tokens, d_out)

        # Split each n*d_out matrix into num_heads copies.
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # Reshape into b, num_heads, num_tokens, head_dim.
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # query_i dot key_j for each i,j in head.
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values

        # Reshape back into b, num_tokens, num_heads, head_dim.
        context_vec = context_vec.transpose(1, 2)
        # Then combine heads to b, num_tokens, d_out.
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # And do this weird projection step.
        context_vec = self.out_proj(context_vec)
        return context_vec


# Some fake input. Each row is a 3-d embedding vector representing a token.
inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your
     [0.55, 0.87, 0.66],  # journey
     [0.57, 0.85, 0.64],  # starts
     [0.22, 0.58, 0.33],  # with
     [0.77, 0.25, 0.10],  # one
     [0.05, 0.80, 0.55]]  # step
)
# print("inputshape: ", inputs.shape)

torch.manual_seed(123)
batch = torch.stack((inputs, inputs), dim=0)
batch_size, context_length, d_in = batch.shape # 2x6x3
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
# print("mha context_vecs shape=",context_vecs.shape)
# print(context_vecs)

# # Exercise 3.3. GPT2 sized attention heads.
# d_in, d_out = 768, 768
# context_length = 1024
# torch.manual_seed(123)
# inputs = torch.randn(context_length, d_in)
# print("inputshape: ", inputs.shape)
# print(inputs)

# torch.manual_seed(123)
# batch = torch.stack((inputs, inputs, inputs, inputs, inputs, inputs, inputs, inputs, inputs, inputs, inputs, inputs), dim=0)
# mha = MultiHeadAttention(d_in, d_out*12, context_length, 0.0, num_heads=12)
# context_vecs = mha(batch)
# print("mha context_vecs shape=",context_vecs.shape)
# print(context_vecs)

mha_ch03 = MultiHeadAttention(
    d_in=embed_dim,
    d_out=embed_dim*12,
    context_length=context_len,
    dropout=0.0,
    num_heads=12,
    qkv_bias=False
).to(device)

out = mha_ch03(embeddings)
# print("shape=", out.shape)
# print(out)