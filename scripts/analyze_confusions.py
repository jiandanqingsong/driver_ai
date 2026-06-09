from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List top off-diagonal confusion pairs.")
    parser.add_argument("--matrix", required=True, help="Path to confusion_matrix.csv.")
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def read_confusion_matrix(path: str | Path) -> tuple[list[str], list[list[int]]]:
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        class_names = header[1:]
        matrix = []
        for row in reader:
            matrix.append([int(value) for value in row[1:]])
    return class_names, matrix


def main() -> None:
    args = parse_args()
    class_names, matrix = read_confusion_matrix(args.matrix)
    pairs = []

    for true_idx, row in enumerate(matrix):
        total = sum(row)
        for pred_idx, count in enumerate(row):
            if true_idx == pred_idx or count == 0:
                continue
            pairs.append(
                {
                    "true": class_names[true_idx],
                    "pred": class_names[pred_idx],
                    "count": count,
                    "rate": count / max(total, 1),
                }
            )

    pairs.sort(key=lambda item: item["count"], reverse=True)
    print("true,pred,count,rate")
    for item in pairs[: args.top_k]:
        print(f"{item['true']},{item['pred']},{item['count']},{item['rate']:.4f}")


if __name__ == "__main__":
    main()
