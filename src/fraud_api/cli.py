"""Command line: ``fraud-api download``, ``fraud-api train`` and ``fraud-api serve``."""

import argparse
import json
import os
from pathlib import Path

import uvicorn

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

    serve_parser = commands.add_parser("serve", help="run the scoring service")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=os.environ.get("PORT", "8000"))

    args = parser.parse_args(argv)
    if args.command == "serve":
        uvicorn.run("fraud_api.service:app", host=args.host, port=args.port)
    elif args.command == "download":
        print(f"verified {download(args.dir)}")
    else:
        model = train(args.data, args.output, seed=args.seed)
        print(f"wrote Model Version {model.id} to {args.output}")
        print(f"Threshold {model.threshold:.6f}")
        print(json.dumps(model.metadata["metrics"], indent=2))
