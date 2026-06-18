from __future__ import annotations

import argparse
from pathlib import Path

from src.plotting import make_comparison_grid, plot_loss_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create report-ready result plots.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    loss = subparsers.add_parser("loss")
    loss.add_argument("--metrics", required=True)
    loss.add_argument("--output", required=True)

    grid = subparsers.add_parser("grid")
    grid.add_argument("--images", nargs="+", required=True)
    grid.add_argument("--labels", nargs="+", required=True)
    grid.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "loss":
        plot_loss_curve(args.metrics, args.output)
    elif args.command == "grid":
        make_comparison_grid(
            [Path(path) for path in args.images],
            args.labels,
            args.output,
        )


if __name__ == "__main__":
    main()
