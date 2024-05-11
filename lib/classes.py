import torch
import torch.nn as nn
from torch.utils.data import Dataset

from lib.config import config
from lib.tokenizer import VOCAB_SIZE


class NextTokenDataset(Dataset):
    def __init__(self, token_ids):
        self.input_ids = []
        self.target_ids = []

        for i in range(
            0, len(token_ids) - config.CONTEXT_LENGTH, config.CONTEXT_LENGTH
        ):
            self.input_ids.append(
                torch.tensor(token_ids[i : i + config.CONTEXT_LENGTH])
            )
            self.target_ids.append(
                torch.tensor(token_ids[i + 1 : i + 1 + config.CONTEXT_LENGTH])
            )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()

        self.head_dim = d_out // config.NUMBER_OF_HEADS

        self.d_in = d_in
        self.d_out = d_out

        self.W_K = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_Q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_V = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.W_out = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(config.DROP_RATE)

        self.register_buffer(
            "mask", torch.tril(torch.ones(config.CONTEXT_LENGTH, config.CONTEXT_LENGTH))
        )

    def forward(self, X):
        batch_size, n_tokens, d_in = X.shape

        K = self.W_K(X)
        Q = self.W_Q(X)
        V = self.W_V(X)

        K = K.view(batch_size, n_tokens, config.NUMBER_OF_HEADS, self.head_dim)
        K = K.transpose(1, 2)

        V = V.view(batch_size, n_tokens, config.NUMBER_OF_HEADS, self.head_dim)
        V = V.transpose(1, 2)

        Q = Q.view(batch_size, n_tokens, config.NUMBER_OF_HEADS, self.head_dim)
        Q = Q.transpose(1, 2)

        S = torch.matmul(Q, K.transpose(2, 3))

        mask = (self.mask == 0)[:n_tokens, :n_tokens]
        S_masked = S.masked_fill(mask, float("-inf"))

        S_masked = S_masked / K.shape[-1] ** 0.5

        W = torch.softmax(S_masked, dim=-1)

        W = self.dropout(W)

        C = torch.matmul(W, V)
        C = C.transpose(1, 2)
        C = C.contiguous().view(batch_size, n_tokens, self.d_out)

        R = self.W_out(C)
        return R


class GPTFeedForward(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.layer1 = nn.Linear(emb_dim, 4 * emb_dim)
        self.activation = nn.GELU()
        self.layer2 = nn.Linear(4 * emb_dim, emb_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        return self.layer2(x)


class GPTBlock(nn.Module):
    def __init__(self, qkv_bias=False):
        super().__init__()

        self.attention = MultiHeadAttention(
            d_in=config.EMBEDDING_DIMENSION,
            d_out=config.EMBEDDING_DIMENSION,
            qkv_bias=qkv_bias,
        )

        self.feed_forward = GPTFeedForward(emb_dim=config.EMBEDDING_DIMENSION)

        self.layer_norm1 = nn.LayerNorm(config.EMBEDDING_DIMENSION)
        self.layer_norm2 = nn.LayerNorm(config.EMBEDDING_DIMENSION)

        self.dropout = nn.Dropout(config.DROP_RATE)

    def forward(self, X):
        X_old = X

        X = self.layer_norm1(X)
        X = self.attention(X)
        X = self.dropout(X)

        X = X + X_old

        X_old = X

        X = self.layer_norm2(X)
        X = self.feed_forward(X)
        X = self.dropout(X)

        X = X + X_old

        return X


class GPTModel(nn.Module):
    def __init__(self, qkv_bias=False):
        super().__init__()

        self.token_emb = nn.Embedding(VOCAB_SIZE, config.EMBEDDING_DIMENSION)
        self.pos_emb = nn.Embedding(VOCAB_SIZE, config.EMBEDDING_DIMENSION)

        self.emb_dropout = nn.Dropout(config.DROP_RATE)

        self.transformer_blocks = nn.Sequential(
            *[GPTBlock(qkv_bias=qkv_bias) for _ in range(config.NUMBER_OF_LAYERS)]
        )

        self.layer_norm = nn.LayerNorm(config.EMBEDDING_DIMENSION)

        self.logits_layer = nn.Linear(
            config.EMBEDDING_DIMENSION, VOCAB_SIZE, bias=False
        )

    def forward(self, X):
        batch_size, seq_len = X.shape
        X_pos = torch.arange(seq_len)

        token_embeds = self.token_emb(X)
        pos_embeds = self.pos_emb(X_pos)

        embeds = token_embeds + pos_embeds
        embeds_dropout = self.emb_dropout(embeds)

        Y = self.transformer_blocks(embeds_dropout)
        Y_norm = self.layer_norm(Y)

        logits = self.logits_layer(Y_norm)
        return logits

    def generate_token_ids(self, token_ids, max_new_tokens):
        token_ids = token_ids.unsqueeze(0)

        for _ in range(max_new_tokens):
            # Needed in case max_new_tokens > context_len
            context_tokens = token_ids[:, -config.CONTEXT_LENGTH :]

            with torch.no_grad():
                logits = self(context_tokens)

            last_logits = logits[:, -1, :]
            next_token_id = torch.argmax(last_logits, dim=-1, keepdim=True)
            token_ids = torch.cat((token_ids, next_token_id), dim=1)

        return token_ids[0]

    def get_batch_loss(self, input_ids, target_ids):
        logits = self(input_ids)

        # Flatten the batches
        logits_flat = logits.flatten(0, 1)
        target_ids_flat = target_ids.flatten()

        loss = torch.nn.functional.cross_entropy(logits_flat, target_ids_flat)
        return loss

    def get_dataloader_loss(self, dataloader):
        total_loss = 0.0

        for input_ids, target_ids in dataloader:
            loss = self.get_batch_loss(input_ids=input_ids, target_ids=target_ids)
            total_loss += loss.item()

        num_batches = len(dataloader)
        if num_batches == 0:
            print("no batches")
            return 0
        return total_loss / num_batches

    def get_loss(self, train_dataloader, val_dataloader):
        self.eval()

        with torch.no_grad():
            train_loss = self.get_dataloader_loss(train_dataloader)
            val_loss = self.get_dataloader_loss(val_dataloader)

        self.train()
        return train_loss, val_loss
