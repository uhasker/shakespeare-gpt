import tiktoken
import torch

from transformers import GPT2Model

from model import GPTModel
from tokenizer import Tokenizer

torch.manual_seed(42)

tokenizer = Tokenizer()

batch = tokenizer.text_to_token_ids("Hello, I am")

print(batch)
print(batch.shape)

# Vocabulary size (BPE tokenizer)
vocab_size = 50257

# Maximum context length
context_len = 256

# Embedding dimension
emb_dim = 768

# Number of attention heads
n_heads = 12

# Number of layers
n_layers = 12

# Dropout rate
drop_rate = 0.1

# Whether to use bias in query, key and value layers
qkv_bias = False

model = GPTModel(
    vocab_size=vocab_size, emb_dim=emb_dim, drop_rate=drop_rate, n_layers=n_layers, context_len=context_len, n_heads=n_heads, qkv_bias=qkv_bias
)
output = model(batch)

print(output)
print(output.shape)

total_params = sum(p.numel() for p in model.parameters())
print(total_params)

out = model.generate_text(
    idx=batch,
    max_new_tokens=6,
    context_len=context_len
)
print(out)

"""start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)

print(encoded_tensor)

model.eval()



print(out)
print(tokenizer.decode(out.squeeze(0).tolist()))"""