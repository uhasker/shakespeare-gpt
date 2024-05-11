from dataclasses import dataclass
from datetime import datetime
import json
import torch
import os

CONFIG_NAME = "config.json"
DEFAULT_CONFIG_PATH = CONFIG_NAME
CHECKPOINTS_DIR = "checkpoints"
DATASETS_DIR = "datasets"

DATASET_KEY = "dataset"
DATE_KEY = "date"
PARAMS_KEY = "params"
LOSSES_KEY = "losses"

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
CHECKPOINT_PREFIX = "checkpoint_"


@dataclass
class Config:
    def __init__(self):
        try:
            with open(DEFAULT_CONFIG_PATH, "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Config file not found")

        self.DATASET = config[DATASET_KEY]
        self.DATE = datetime.now()
        self.NEXT_CHECKPOINT = 0

        self.FOLDER_NAME = f"{self.DATASET} {self.date_str}"

        for key, value in config[PARAMS_KEY].items():
            setattr(self, key, value)

        self.config = config
        self.config.update({LOSSES_KEY: []})
        self.config.update({DATE_KEY: self.DATE.strftime(DATE_FORMAT)})

        if not os.path.exists(CHECKPOINTS_DIR):
            os.makedirs(CHECKPOINTS_DIR)

        if not os.path.exists(DATASETS_DIR):
            os.makedirs(DATASETS_DIR)

        if self.EMBEDDING_DIMENSION % self.NUMBER_OF_HEADS != 0:
            raise ValueError(
                f"EMBEDDING_DIMENSION ({self.EMBEDDING_DIMENSION}) must be divisible by NUMBER_OF_HEADS ({self.NUMBER_OF_HEADS})"
            )

    @property
    def date_str(self):
        return self.DATE.strftime(DATE_FORMAT)

    @property
    def folder_path(self):
        return os.path.join(CHECKPOINTS_DIR, self.FOLDER_NAME)

    @property
    def dataset_path(self):
        return os.path.join(DATASETS_DIR, f"{self.DATASET}.txt")

    @property
    def latest_checkpoint(self):
        return self.NEXT_CHECKPOINT - 1

    def update_from_foldername(self, folder_name):
        self.FOLDER_NAME = folder_name
        folder_path = os.path.join(CHECKPOINTS_DIR, folder_name)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder {folder_name} not found")

        with open(os.path.join(folder_path, CONFIG_NAME), "r") as f:
            config = json.load(f)

        return self.update_from_config(config)

    def update_from_config(self, config):
        self.config.update(config)

        self.DATASET = config[DATASET_KEY]
        self.DATE = datetime.strptime(config[DATE_KEY], DATE_FORMAT)
        self.NEXT_CHECKPOINT = len(
            [
                f
                for f in os.listdir(os.path.join(self.folder_path))
                if f.startswith(CHECKPOINT_PREFIX)
            ]
        )

        for key, value in config[PARAMS_KEY].items():
            if hasattr(self, key):
                setattr(self, key, value)

        if self.EMBEDDING_DIMENSION % self.NUMBER_OF_HEADS != 0:
            raise ValueError(
                f"EMBEDDING_DIMENSION ({self.EMBEDDING_DIMENSION}) must be divisible by NUMBER_OF_HEADS ({self.NUMBER_OF_HEADS})"
            )

        with open(os.path.join(self.folder_path, f"config.json"), "w") as f:
            json.dump(self.config, f, indent=4)

        return self

    def save_config(self):
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        with open(os.path.join(self.folder_path, f"config.json"), "w") as f:
            json.dump(self.config, f, indent=4)

    def save_checkpoint(self, state, training_loss, validation_loss):
        torch.save(
            state,
            os.path.join(self.folder_path, f"checkpoint_{self.NEXT_CHECKPOINT}.pth"),
        )
        self.NEXT_CHECKPOINT += 1

        # first is the training loss, second is the validation loss
        self.config["losses"].append((training_loss, validation_loss))
        self.save_config()

    def checkpoint_path(self, checkpoint):
        return os.path.join(self.folder_path, f"{CHECKPOINT_PREFIX}{checkpoint}.pth")


config = Config()
