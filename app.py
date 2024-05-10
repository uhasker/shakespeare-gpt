from argparse import ArgumentParser
import os

from lib.data_manager import Runtime
from lib.inference import generate
from lib.train import train
from lib.const import config


def parse_args():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", help="Mode to run", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model")

    run_parser = subparsers.add_parser("run", help="Run a model")
    run_parser.add_argument(
        "folder",
        help="Folder name from which to load the model and config. Only the name is required, not the full path!",
    )
    run_parser.add_argument("start", help="Starting text to generate from")
    run_parser.add_argument(
        "-c",
        "--checkpoint",
        metavar="CHECKPOINT",
        help="Checkpoint number to load from (Defaults to latest checkpoint in folder)",
        default=None,
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    runtime = Runtime.from_folder_name(args.folder) if args.folder else None

    if args.mode == "train":
        print("### Training ###")
        train(runtime=runtime)
    elif args.mode == "run":
        if args.checkpoint is None:
            args.checkpoint = runtime.next_checkpoint - 1
        checkpoint = os.path.join(
            runtime.folder_path, f"checkpoint_{args.checkpoint}.pth"
        )

        print("### Running ###")
        generate(args.start, runtime, checkpoint_path=checkpoint)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
