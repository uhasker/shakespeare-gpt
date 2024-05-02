import torch

from dataset import create_next_token_dataloader
from model import GPTModel
from tokenizer import TiktokenTokenizer

train_filename = "verdict.txt"
val_filename = "verdict.txt"

batch_size = 2

context_len = 256

emb_dim = 768

drop_rate = 0.1

n_layers = 6

n_heads = 12

with open(train_filename, "r", encoding="utf-8") as f:
    train_data = f.read()

with open(val_filename, "r", encoding="utf-8") as f:
    val_data = f.read()

tokenizer = TiktokenTokenizer()
vocab_size = tokenizer.vocab_size

train_token_ids = tokenizer.encode(train_data)
val_token_ids = tokenizer.encode(val_data)

train_dataloader = create_next_token_dataloader(
    token_ids=train_token_ids,
    batch_size=batch_size,
    context_len=context_len,
    shuffle=True,
    drop_last=True
)

val_dataloader = create_next_token_dataloader(
    token_ids=val_token_ids,
    batch_size=batch_size,
    context_len=context_len,
    shuffle=True,
    drop_last=True
)

model = GPTModel(
    vocab_size=vocab_size, emb_dim=emb_dim, drop_rate=drop_rate, n_layers=n_layers, context_len=context_len,
    n_heads=n_heads, qkv_bias=False
)

# print(torch.tensor(train_token_ids[:4]))
# print(model.generate_token_ids(token_ids=torch.tensor(train_token_ids[:4]), max_new_tokens=4, context_len=4))

train_loss, val_loss = model.get_loss(train_dataloader, val_dataloader)
print(train_loss, val_loss)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
model.train_loop(
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    optimizer=optimizer,
    n_epochs=10
)
