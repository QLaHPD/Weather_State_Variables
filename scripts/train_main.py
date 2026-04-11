from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from weather_state_variables.training import train_main_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the FuXi-style forecast model.")
    parser.add_argument(
        "--config",
        default="configs/model_config.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run model print/summary and a random forward pass without loading remote data or training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_main_model(args.config, smoke_only=args.smoke_only)


if __name__ == "__main__":
    main()
