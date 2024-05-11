from dataclasses import dataclass
import os


@dataclass
class Config:
    CHECKPOINTS_DIR: str = "checkpoints"
    DATASETS_DIR: str = "datasets"

    BATCH_SIZE: int = 32
    CONTEXT_LENGTH: int = 32

    EMBEDDING_DIMENSION: int = 128  # must be divisible by NUMBER_OF_HEADS

    DROP_RATE: float = 0.1

    NUMBER_OF_LAYERS: int = 2
    NUMBER_OF_HEADS: int = 2
    NUMBER_OF_EPOCHS: int = 3

    def __post_init__(self):
        if not os.path.exists(self.CHECKPOINTS_DIR):
            os.makedirs(self.CHECKPOINTS_DIR)

        if not os.path.exists(self.DATASETS_DIR):
            os.makedirs(self.DATASETS_DIR)

        if self.EMBEDDING_DIMENSION % self.NUMBER_OF_HEADS != 0:
            raise ValueError(
                f"EMBEDDING_DIMENSION ({self.EMBEDDING_DIMENSION}) must be divisible by NUMBER_OF_HEADS ({self.NUMBER_OF_HEADS})"
            )

    def update_from_config(self, config):
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


config = Config()
