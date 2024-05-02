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

    def generate_token_ids(self, token_ids, max_new_tokens, context_len):
        token_ids = token_ids.unsqueeze(0)

        for _ in range(max_new_tokens):
            # Needed in case max_new_tokens > context_len
            context_tokens = token_ids[:, -context_len:]

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
        return total_loss / num_batches

    def get_loss(self, train_dataloader, val_dataloader):
        self.eval()

        with torch.no_grad():
            train_loss = self.get_dataloader_loss(train_dataloader)
            val_loss = self.get_dataloader_loss(val_dataloader)

        self.train()
        return train_loss, val_loss

    def train_loop(self, train_dataloader, val_dataloader, optimizer, n_epochs):
        for epoch in range(n_epochs):
            self.train()

            for input_batch, target_batch in train_dataloader:
                print("batch")
                # Standard training loop
                optimizer.zero_grad()
                loss = self.get_batch_loss(input_batch, target_batch)
                loss.backward()
                optimizer.step()

            train_loss, val_loss = self.get_loss(train_dataloader, val_dataloader)
            print(f"Epoch {epoch}: Train loss={round(train_loss, 2)}, Validation loss={round(val_loss, 2)}")

            torch.save(self.state_dict(), f"model_{epoch}.pth")