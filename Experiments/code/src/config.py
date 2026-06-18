from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config["_config_path"] = str(config_path)
    return config


def save_config(config: dict[str, Any], output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    clean_config = {k: v for k, v in config.items() if not k.startswith("_")}
    path = output_path / "config.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(clean_config, f, indent=2, ensure_ascii=False)
    return path


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    updated = dict(config)
    for key, value in overrides.items():
        if value is not None:
            updated[key] = value
    return updated
