from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from c2meta.detection import scan_csv, score_with_model, train_model


DISPLAY_COLUMNS = [
    "src_ip",
    "dst_ip",
    "dst_port",
    "protocol",
    "flow_count",
    "avg_interval_sec",
    "interval_cv",
    "avg_total_bytes",
    "risk_score",
    "verdict",
]


def _print_table(frame: pd.DataFrame, top: int) -> None:
    visible_columns = [column for column in DISPLAY_COLUMNS if column in frame.columns]
    print(frame[visible_columns].head(top).to_string(index=False))


def scan_command(args: argparse.Namespace) -> None:
    if args.model:
        result = score_with_model(args.csv, args.model)
    else:
        result = scan_csv(args.csv, args.rules)

    _print_table(result, args.top)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
        print(f"\nReport saved to: {output}")


def train_command(args: argparse.Namespace) -> None:
    report = train_model(args.csv, args.model)
    print(report)
    print(f"Model saved to: {args.model}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="c2meta",
        description="Detect C2-like encrypted traffic channels using flow metadata.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="Score network flow metadata from CSV")
    scan.add_argument("csv", help="Path to input flow CSV")
    scan.add_argument("--rules", help="Path to JSON rules config")
    scan.add_argument("--model", help="Path to trained joblib model")
    scan.add_argument("--top", type=int, default=15, help="Number of rows to print")
    scan.add_argument("--output", help="Optional CSV report path")
    scan.set_defaults(func=scan_command)

    train = subparsers.add_parser("train", help="Train a simple classifier from labeled CSV")
    train.add_argument("csv", help="Path to labeled flow CSV")
    train.add_argument("--model", default="artifacts/c2_model.joblib", help="Output model path")
    train.set_defaults(func=train_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
