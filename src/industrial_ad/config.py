"""YAML configuration with defaults and command-line overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "data": {
        "root": "datasets/MVTecAD",
        "category": "pill",
        "image_size": [256, 256],
        "calib_ratio": 0.1,
        "seed": 42,
        "max_test_samples": 0,
    },
    "model": {
        "name": "patchcore",
        "patchcore": {
            "backbone": "wide_resnet50_2",
            "layers": ["layer2", "layer3"],
            "coreset_sampling_ratio": 0.1,
            "num_neighbors": 9,
        },
        "padim": {
            "backbone": "resnet18",
            "layers": ["layer1", "layer2", "layer3"],
        },
    },
    "train": {
        "accelerator": "auto",
        "batch_size": 32,
        "num_workers": 0,
    },
    "evaluation": {
        "max_fpr": 0.05,
        "aupro_fpr_limit": 0.3,
        "batch_size": 8,
        "save_all_panels": False,
    },
    "output_dir": "runs",
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = DEFAULTS
    if path is not None:
        with Path(path).open(encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def run_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["output_dir"]) / cfg["model"]["name"] / cfg["data"]["category"]


def category_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["data"]["root"]) / cfg["data"]["category"]
