import tiktoken
import torch


class Tokenizer:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def text_to_token_ids(self, text):
        encoded = self.tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)
        return encoded_tensor

    def token_ids_to_text(self, token_ids):
        return self.tokenizer.decode(token_ids.squeeze(0).tolist())
