"""Command line: ``fraud-api download`` and ``fraud-api train``."""

import argparse
import json
from pathlib import Path

from fraud_api.data import DATA_DIR, FILE_NAME
from fraud_api.model_version import MODEL_DIR
from fraud_api.training import SEED


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fraud-api")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="download and verify Base.csv from Kaggle")
    download.add_argument("--dir", type=Path, default=DATA_DIR)

    train = commands.add_parser("train", help="train a Model Version from Base.csv")
    train.add_argument("--data", type=Path, default=DATA_DIR / FILE_NAME)
    train.add_argument("--output", type=Path, default=MODEL_DIR)
    train.add_argument("--seed", type=int, default=SEED)

    args = parser.parse_args(argv)
    if args.command == "download":
        from fraud_api.data import download as run_download

        print(f"verified {run_download(args.dir)}")
    else:
        from fraud_api.training import train as run_train

        model = run_train(args.data, args.output, seed=args.seed)
        print(f"wrote Model Version {model.id} to {args.output}")
        print(f"Threshold {model.threshold:.6f}")
        print(json.dumps(model.metadata["metrics"], indent=2))
