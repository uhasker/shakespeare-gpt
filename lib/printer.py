from dataclasses import dataclass
import time


class Printer:
    def __init__(self, total_batches, total_epochs) -> None:
        self.batch: int = 1
        self.epoch: int = 1
        self.total_batches: int = total_batches
        self.total_epochs: int = total_epochs
        self.start_time = time.time()
        self.max_len = 0

    def __call__(self):
        self._print()

    def _print(self):
        elapsed = time.time() - self.start_time
        remaining = elapsed / self.batch * (self.total_batches - self.batch)
        text = f"Batch: {self.batch}/{self.total_batches} - Elapsed: {elapsed:.2f}s - Remaining: {remaining:.2f}s"
        self.max_len = max(self.max_len, len(text))
        print(f"{text: <{self.max_len}}", end="\r")
        self.batch += 1

    def new_epoch(self, training_loss, validation_loss):
        print(
            f"Epoch {self.epoch}/{self.total_epochs} - Elapsed: {time.time() - self.start_time:.2f}s - Training Loss: {training_loss:.4f} - Validation Loss: {validation_loss:.4f}"
        )
        self.start_time = time.time()
        self.batch = 1
        self.epoch += 1
        self.max_len = 0
        self._print()


def main():
    total = 100
    printer = Printer(1, total)
    for i in range(1, total):
        printer.batch = i
        printer()
        time.sleep(0.1)
    print()


if __name__ == "__main__":
    main()
