import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_len, drop_rate, n_heads, qkv_bias=False):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = d_out // self.n_heads

        self.d_in = d_in
        self.d_out = d_out

        self.W_K = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_Q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_V = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.W_out = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(drop_rate)

        self.register_buffer("mask", torch.tril(torch.ones(context_len, context_len)))

    def forward(self, X):
        batch_size, n_tokens, d_in = X.shape

        K = self.W_K(X)
        Q = self.W_Q(X)
        V = self.W_V(X)

        K = K.view(batch_size, n_tokens, self.n_heads, self.head_dim)
        K = K.transpose(1, 2)

        V = V.view(batch_size, n_tokens, self.n_heads, self.head_dim)
        V = V.transpose(1, 2)

        Q = Q.view(batch_size, n_tokens, self.n_heads, self.head_dim)
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


if __name__ == '__main__':
    batch_size = 2
    context_len = 5
    d = 4

    layer = MultiHeadAttention(d_in=d, d_out=d, context_len=context_len, drop_rate=0.1, n_heads=2)

    X = torch.randn(batch_size, context_len, d)
    print(X.shape)
    print(X)

    Y = layer(X)
    print(Y.shape)
    print(Y)
