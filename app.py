from argparse import ArgumentParser
import os

from lib.inference import generate
from lib.train import train
from lib.config import config


def parse_args():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", help="Mode to run", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "-f",
        "--folder",
        help='Folder name from which to load the model and config. Only the name is required, not the full path! To use a custom checkpoint, use "<foldername>":<checkpoint> format. default: latest checkpoint. Currently does nothing .)',
        default=None,
    )
    train_parser.add_argument(
        "-c",
        "--checkpoint",
        help="Checkpoint to load. Only possible if a folder is specified. Only the number is required, not the full path! default: latest checkpoint",
        default=None,
    )

    run_parser = subparsers.add_parser("run", help="Run a model")
    run_parser.add_argument(
        "folder",
        help="Folder name from which to load the model and config. Only the name is required, not the full path!",
        default=None,
    )
    run_parser.add_argument("start", help="Starting text to generate from")
    run_parser.add_argument(
        "-c",
        "--checkpoint",
        help="Checkpoint to load. Only possible if a folder is specified. Only the number is required, not the full path! default: latest checkpoint",
        default=None,
    )

    args = parser.parse_args()

    if args.folder is None and args.checkpoint is not None:
        raise parser.error("Checkpoint can only be specified if a folder is specified")

    return args


def main():
    args = parse_args()
    if args.folder is not None:
        config.update_from_foldername(args.folder)
    else:
        config.save_config()

    if args.checkpoint is not None:
        checkpoint = int(args.checkpoint)
    else:
        checkpoint = config.latest_checkpoint

    checkpoint_path = config.checkpoint_path(checkpoint)

    if args.mode == "train":
        print("### Training ###")
        train()
    elif args.mode == "run":
        print("### Running ###")
        generate(args.start, checkpoint_path=checkpoint_path)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
