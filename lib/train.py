import torch

from lib.const import config
from lib.data_manager import Runtime
from lib.tokenizer import tokenizer
from lib.util import create_next_token_dataloader
from lib.classes import GPTModel


def train(runtime):
    with open(runtime.dataset_path, "r", encoding="utf-8") as f:
        data = f.read()

    n_split = int(0.9 * len(data))
    train_data = data[:n_split]
    val_data = data[n_split:]
    print(f"Total data:\t\t{len(data):_} Characters!")
    print(f"Training data:\t\t{len(train_data):_} Characters!")
    print(f"Validation data:\t{len(val_data):_} Characters!")

    train_token_ids = tokenizer.encode(train_data)
    val_token_ids = tokenizer.encode(val_data)

    # CREATE DATALOADERS
    train_dataloader = create_next_token_dataloader(
        token_ids=train_token_ids,
        shuffle=True,
        drop_last=True,
    )
    val_dataloader = create_next_token_dataloader(
        token_ids=val_token_ids,
        shuffle=True,
        drop_last=True,
    )

    # CREATE MODEL
    model = GPTModel(qkv_bias=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    number_of_batches = len(train_dataloader)

    print("## STARTING TRAINING ##")
    for epoch in range(config.NUMBER_OF_EPOCHS):
        model.train()

        for i, (input_batch, target_batch) in enumerate(train_dataloader, start=1):
            print(f"Batch {i} (out of {number_of_batches})")
            # Standard training loop
            optimizer.zero_grad()
            loss = model.get_batch_loss(input_batch, target_batch)
            loss.backward()
            optimizer.step()

        train_loss, val_loss = model.get_loss(train_dataloader, val_dataloader)
        print(
            f"Epoch {epoch}: Train loss={round(train_loss, 2)}, Validation loss={round(val_loss, 2)}"
        )

        runtime.add_loss(train_loss, val_loss)
        runtime.save_checkpoint(model.state_dict())
