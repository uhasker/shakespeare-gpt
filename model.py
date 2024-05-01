import torch
import torch.nn as nn

from gpt_block import GPTBlock


class GPTModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, drop_rate, n_layers, context_len, n_heads, qkv_bias=False):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(vocab_size, emb_dim)

        self.emb_dropout = nn.Dropout(drop_rate)

        self.transformer_blocks = nn.Sequential(
            *[GPTBlock(emb_dim=emb_dim, context_len=context_len, n_heads=n_heads, drop_rate=drop_rate,
                       qkv_bias=qkv_bias) for _ in range(n_layers)]
        )

        self.layer_norm = nn.LayerNorm(emb_dim)

        self.logits_layer = nn.Linear(emb_dim, vocab_size, bias=False)

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

    def generate_text(self, idx, max_new_tokens, context_len):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_len:]
            with torch.no_grad():
                logits = self(idx_cond)

            logits = logits[:, -1, :]
            probas = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
