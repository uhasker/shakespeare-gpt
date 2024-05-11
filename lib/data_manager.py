from datetime import datetime
import json
import os

import torch
from lib.const import config as project_config

DATASET_KEY = "dataset"
DATE_KEY = "date"
VALUES_KEY = "values"
LOSSES_KEY = "losses"


class Runtime:
    def __init__(self, dataset=None, config=None):
        if dataset is config is None:
            raise ValueError("Either dataset or config must be provided")

        if config is not None:
            self.config = config
            self.dataset = config[DATASET_KEY]
            self.date = config[DATE_KEY]
            self.next_checkpoint = len(
                [
                    f
                    for f in os.listdir(os.path.join(self.folder_path))
                    if f.startswith("checkpoint_")
                ]
            )
            project_config.update_from_config(config[VALUES_KEY])
        else:
            self.dataset = dataset.replace(".txt", "")
            self.date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.next_checkpoint = 0
            self.config = {
                DATASET_KEY: self.dataset,
                DATE_KEY: self.date,
                VALUES_KEY: {
                    "BATCH_SIZE": project_config.BATCH_SIZE,
                    "CONTEXT_LENGTH": project_config.CONTEXT_LENGTH,
                    "EMBEDDING_DIMENSION": project_config.EMBEDDING_DIMENSION,
                    "DROP_RATE": project_config.DROP_RATE,
                    "NUMBER_OF_LAYERS": project_config.NUMBER_OF_LAYERS,
                    "NUMBER_OF_HEADS": project_config.NUMBER_OF_HEADS,
                    "NUMBER_OF_EPOCHS": project_config.NUMBER_OF_EPOCHS,
                },
                LOSSES_KEY: [],
            }
            if not os.path.exists(
                os.path.join(project_config.CHECKPOINTS_DIR, self.folder_name)
            ):
                os.makedirs(
                    os.path.join(project_config.CHECKPOINTS_DIR, self.folder_name)
                )

            if not os.path.exists(self.dataset_path):
                raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
            self.save_config()

    @property
    def folder_name(self):
        return f"{self.dataset} {self.date}"

    @property
    def folder_path(self):
        return os.path.join(project_config.CHECKPOINTS_DIR, self.folder_name)

    @property
    def dataset_path(self):
        return os.path.join(project_config.DATASETS_DIR, f"{self.dataset}.txt")

    @property
    def latest_checkpoint(self):
        return os.path.join(
            self.folder_path, f"checkpoint_{self.next_checkpoint - 1}.pth"
        )

    @staticmethod
    def from_folder_name(folder_name):
        folder_path = os.path.join(project_config.CHECKPOINTS_DIR, folder_name)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder {folder_name} not found")

        with open(os.path.join(folder_path, "config.json"), "r") as f:
            config = json.load(f)

        return Runtime(config=config)

    def save_config(self):
        with open(os.path.join(self.folder_path, f"config.json"), "w") as f:
            json.dump(self.config, f, indent=4)

    def save_checkpoint(self, state):
        torch.save(
            state,
            os.path.join(self.folder_path, f"checkpoint_{self.next_checkpoint}.pth"),
        )
        self.next_checkpoint += 1

    def add_loss(self, training_loss, validation_loss):
        # left is the training loss, right is the validation loss
        self.config["losses"].append((training_loss, validation_loss))
        self.save_config()
