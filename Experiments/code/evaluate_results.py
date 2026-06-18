from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize style-transfer metrics.")
    parser.add_argument("runs", nargs="+", help="Run directories containing metrics.csv")
    parser.add_argument("--output", default="summary.csv")
    return parser.parse_args()


def summarize_run(run_dir: str | Path) -> dict:
    path = Path(run_dir)
    metrics_path = path / "metrics.csv"
    config_path = path / "config.json"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    df = pd.read_csv(metrics_path)
    final = df.iloc[-1].to_dict()
    summary = {
        "run_dir": str(path),
        "steps": int(final["step"]),
        "final_total_loss": final["total_loss"],
        "final_content_loss": final["content_loss"],
        "final_style_loss": final["style_loss"],
        "final_tv_loss": final["tv_loss"],
    }
    for key in [
        "weighted_content_loss",
        "weighted_style_loss",
        "weighted_tv_loss",
    ]:
        if key in final:
            summary[f"final_{key}"] = final[key]

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        for key in [
            "experiment_name",
            "content_weight",
            "style_weight",
            "tv_weight",
            "content_layers",
            "style_layers",
            "optimizer",
            "pooling",
            "init",
            "image_size",
        ]:
            summary[key] = config.get(key)
    return summary


def main() -> None:
    args = parse_args()
    rows = [summarize_run(path) for path in args.runs]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
