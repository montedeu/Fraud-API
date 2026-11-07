"""Command line: ``fraud-api download``, ``fraud-api train`` and ``fraud-api serve``."""

import argparse
import os
from pathlib import Path

import uvicorn

from fraud_api.data import DATA_DIR, FILE_NAME, download
from fraud_api.model_version import MODEL_DIR

# Kept in step with fraud_api.training, fraud_api.tuning and fraud_api.reports, which are
# imported only for `train`: serving must not need Optuna or matplotlib.
SEED = 42
N_TRIALS = 50
REPORTS_DIR = Path("reports")


def _trials(value: str) -> int:
    trials = int(value)
    if trials < 2:
        raise argparse.ArgumentTypeError("at least 2 trials are needed")
    return trials


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
    train_parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    train_parser.add_argument("--seed", type=int, default=SEED)
    train_parser.add_argument(
        "--trials", type=_trials, default=N_TRIALS, help="Optuna trials (at least 2)"
    )

    serve_parser = commands.add_parser("serve", help="run the scoring service")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=os.environ.get("PORT", "8000"))

    args = parser.parse_args(argv)
    if args.command == "serve":
        uvicorn.run("fraud_api.service:app", host=args.host, port=args.port)
    elif args.command == "download":
        print(f"verified {download(args.dir)}")
    else:
        from fraud_api.training import train

        model = train(
            args.data, args.output, reports_dir=args.reports, seed=args.seed, n_trials=args.trials
        )
        summary = model.metadata["optuna"]
        print(
            f"Tuning Month Recall@5%FPR: default {summary['tuning_month_default_score']:.3f}, "
            f"tuned {summary['tuning_month_tuned_score']:.3f} ({summary['n_trials']} trials)"
        )
        print(f"wrote Model Version {model.id} to {args.output}")
        print(f"Threshold {model.threshold:.6f}\n")
        models = {"lightgbm": model.metadata} | model.metadata["baselines"]
        print(f"{'Recall@5%FPR':24} {'validation':>10} {'test':>10}")
        for name, result in models.items():
            recall = {s: result["metrics"][s]["recall_at_5pct_fpr"] for s in ("validation", "test")}
            print(f"{name:24} {recall['validation']:10.3f} {recall['test']:10.3f}")


if __name__ == "__main__":
    main()
