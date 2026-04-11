from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_CONFIG_PATH = REPO_ROOT / "configs" / "model_config.yaml"

_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?((\d+\.\d*)|(\d*\.\d+)|(\d+))(e[+-]?\d+)?$", re.IGNORECASE)


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if _INT_RE.match(value):
        return int(value)
    if _FLOAT_RE.match(value) and any(ch in value for ch in ".eE"):
        return float(value)
    if value.startswith(("[", "{", "(", "'", '"')):
        return ast.literal_eval(value)
    return value


def _parse_simple_yaml_block(lines: list[str], start: int, indent: int) -> tuple[dict[str, Any], int]:
    mapping: dict[str, Any] = {}
    index = start

    while index < len(lines):
        raw_line = lines[index]
        current_indent = len(raw_line) - len(raw_line.lstrip(" "))
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ValueError(f"Unexpected indentation in config line: {raw_line!r}")

        stripped = raw_line.strip()
        key, separator, raw_value = stripped.partition(":")
        if separator != ":":
            raise ValueError(f"Expected key/value pair, got: {raw_line!r}")

        key = key.strip()
        value = raw_value.strip()
        index += 1

        if value == "":
            child, index = _parse_simple_yaml_block(lines, index, indent + 2)
            mapping[key] = child
        else:
            mapping[key] = _parse_scalar(value)

    return mapping, index


def _load_simple_yaml(text: str) -> dict[str, Any]:
    lines = [
        line.rstrip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    mapping, index = _parse_simple_yaml_block(lines, start=0, indent=0)
    if index != len(lines):
        raise ValueError("Config parser did not consume the entire file.")
    return mapping


def load_yaml_config(config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH) -> tuple[Path, dict[str, Any]]:
    resolved_path = Path(config_path)
    if not resolved_path.is_absolute():
        resolved_path = (REPO_ROOT / resolved_path).resolve()
    else:
        resolved_path = resolved_path.resolve()

    text = resolved_path.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore
    except ImportError:
        config = _load_simple_yaml(text)
    else:
        config = yaml.safe_load(text)

    if not isinstance(config, dict):
        raise ValueError(f"Expected config root to be a mapping, got {type(config).__name__}")
    return resolved_path, config


def load_config_section(
    section: str,
    config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
) -> tuple[Path, dict[str, Any]]:
    resolved_path, config = load_yaml_config(config_path)
    section_value = config.get(section)
    if not isinstance(section_value, dict):
        raise KeyError(f"Missing config section '{section}' in {resolved_path}")
    return resolved_path, section_value


def resolve_repo_path(path_value: str | Path, *, config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH) -> Path:
    raw_path = Path(path_value)
    if raw_path.is_absolute():
        return raw_path.resolve()

    resolved_config_path, _ = load_yaml_config(config_path)
    config_relative_path = (resolved_config_path.parent / raw_path).resolve()
    if config_relative_path.exists():
        return config_relative_path
    return (REPO_ROOT / raw_path).resolve()


def resolve_torch_dtype(dtype_name: str | None) -> torch.dtype | None:
    if dtype_name is None:
        return None

    mapping = {
        "float16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "float": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    key = dtype_name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype_name}'.")
    return mapping[key]


__all__ = [
    "DEFAULT_MODEL_CONFIG_PATH",
    "REPO_ROOT",
    "load_config_section",
    "load_yaml_config",
    "resolve_repo_path",
    "resolve_torch_dtype",
]
