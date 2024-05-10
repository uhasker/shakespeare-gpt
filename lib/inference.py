import torch

from lib.classes import GPTModel
from lib.tokenizer import tokenizer
from lib.const import config


def generate(start, runtime, checkpoint_path=None):
    model = GPTModel(qkv_bias=False)

    token_ids = tokenizer.encode(start)

    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        model.load_state_dict(torch.load(runtime.latest_checkpoint))

    model.eval()
    generated_ids = model.generate_token_ids(torch.tensor(token_ids), max_new_tokens=50)

    print(tokenizer.decode(generated_ids.tolist()))
