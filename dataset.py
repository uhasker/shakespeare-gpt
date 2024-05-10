import torch
from torch.utils.data import Dataset, DataLoader


class NextTokenDataset(Dataset):
    def __init__(self, token_ids, context_len):
        self.input_ids = []
        self.target_ids = []

        for i in range(0, len(token_ids) - context_len, context_len):
            self.input_ids.append(torch.tensor(token_ids[i:i + context_len]))
            self.target_ids.append(torch.tensor(token_ids[i + 1:i + 1 + context_len]))

        print("stop")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_next_token_dataloader(token_ids, batch_size, context_len, shuffle=True, drop_last=True):
    dataset = NextTokenDataset(token_ids=token_ids, context_len=context_len)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader
