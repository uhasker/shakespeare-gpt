import torch
import torch.nn as nn

from multi_head_attention import MultiHeadAttention


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
    def __init__(self, emb_dim, context_len, n_heads, drop_rate, qkv_bias=False):
        super().__init__()

        self.attention = MultiHeadAttention(
            d_in=emb_dim,
            d_out=emb_dim,
            context_len=context_len,
            n_heads=n_heads,
            drop_rate=drop_rate,
            qkv_bias=qkv_bias
        )

        self.feed_forward = GPTFeedForward(emb_dim=emb_dim)

        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.layer_norm2 = nn.LayerNorm(emb_dim)

        self.dropout = nn.Dropout(drop_rate)

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


if __name__ == '__main__':
    batch_size = 2
    context_len = 5
    d = 4

    layer = GPTBlock(emb_dim=d, context_len=context_len, drop_rate=0.1, n_heads=2)

    X = torch.randn(batch_size, context_len, d)
    print(X.shape)

    Y = layer(X)
    print(Y.shape)
