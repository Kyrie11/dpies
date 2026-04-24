from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def parse_override_items(items: list[str] | None) -> Dict[str, Any]:
    """Parse CLI overrides like training.lr=3e-4 selection.budget=16."""
    out: Dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw = item.split("=", 1)
        try:
            value = yaml.safe_load(raw)
        except Exception:
            value = raw
        cur = out
        parts = key.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value
    return out
