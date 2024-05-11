from lib.classes import NextTokenDataset
from torch.utils.data import DataLoader

from lib.config import config


def create_next_token_dataloader(token_ids, shuffle=True, drop_last=True):
    dataset = NextTokenDataset(token_ids=token_ids)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return dataloader
