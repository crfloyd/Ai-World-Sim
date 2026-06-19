"""Configuration loading and merging utilities."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return it as a nested dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge *override* into *base*, returning a new dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def get(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely fetch a nested key with a default."""
    node = config
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key, default)
    return node


DEFAULT_WORLD_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "world.yaml"
DEFAULT_TRAIN_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "train.yaml"
