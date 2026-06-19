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


def resolve_predator_profile(config: dict[str, Any]) -> dict[str, Any]:
    """Merge the active predator profile into config['animals'] and return a new config.

    Reads ``config['animals']['predator_curriculum_phase']`` to select a named
    profile from ``config['animals']['predator_profiles']``, then shallow-merges
    the profile keys over the base animals values.  Non-profile keys are preserved.
    Returns the original config unchanged if no phase is set or no matching profile
    is found.
    """
    anim = config.get("animals", {})
    phase = anim.get("predator_curriculum_phase")
    if not phase:
        return config

    profiles = anim.get("predator_profiles", {})
    profile = profiles.get(phase)
    if not profile:
        return config

    new_anim: dict[str, Any] = {**anim, **profile}
    return {**config, "animals": new_anim}
