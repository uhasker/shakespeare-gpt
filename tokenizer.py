import tiktoken
import torch


class TiktokenTokenizer:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def encode(self, text):
        return self.tokenizer.encode(text, allowed_special={'<|endoftext|>'})

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def vocab_size(self):
        return self.tokenizer.n_vocab

    """def text_to_token_ids(self, text):
        encoded = self.tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)
        return encoded_tensor

    def token_ids_to_text(self, token_ids):
        return self.tokenizer.decode(token_ids.squeeze(0).tolist())"""
