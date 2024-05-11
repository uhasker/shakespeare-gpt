import torch

from lib.classes import GPTModel
from lib.tokenizer import tokenizer
from lib.config import config


def generate(start, checkpoint_path):
    model = GPTModel(qkv_bias=False)

    token_ids = tokenizer.encode(start)

    model.load_state_dict(torch.load(checkpoint_path))

    model.eval()
    generated_ids = model.generate_token_ids(torch.tensor(token_ids), max_new_tokens=50)

    print(tokenizer.decode(generated_ids.tolist()))
