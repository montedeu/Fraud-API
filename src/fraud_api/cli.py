"""Command line: ``fraud-api download`` and ``fraud-api train``."""

import argparse
import json
from pathlib import Path

from fraud_api.data import DATA_DIR, FILE_NAME, download
from fraud_api.model_version import MODEL_DIR
from fraud_api.training import SEED, train


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fraud-api")
    commands = parser.add_subparsers(dest="command", required=True)

    download_parser = commands.add_parser(
        "download", help="download and verify Base.csv from Kaggle"
    )
    download_parser.add_argument("--dir", type=Path, default=DATA_DIR)

    train_parser = commands.add_parser("train", help="train a Model Version from Base.csv")
    train_parser.add_argument("--data", type=Path, default=DATA_DIR / FILE_NAME)
    train_parser.add_argument("--output", type=Path, default=MODEL_DIR)
    train_parser.add_argument("--seed", type=int, default=SEED)

    args = parser.parse_args(argv)
    if args.command == "download":
        print(f"verified {download(args.dir)}")
    else:
        model = train(args.data, args.output, seed=args.seed)
        print(f"wrote Model Version {model.id} to {args.output}")
        print(f"Threshold {model.threshold:.6f}")
        print(json.dumps(model.metadata["metrics"], indent=2))
