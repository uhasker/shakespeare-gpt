import tiktoken


class TiktokenTokenizer:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def encode(self, text):
        return self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def vocab_size(self):
        return self.tokenizer.n_vocab


# Not pretty but circular import otherwise
tokenizer = TiktokenTokenizer()
VOCAB_SIZE = tokenizer.vocab_size
