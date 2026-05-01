from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from weather_state_variables.benchmarking import (
    DEFAULT_EARTH2STUDIO_MODEL_CACHE,
    download_earth2studio_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and cache the Earth2Studio benchmark models so later comparison runs "
            "can reuse the local checkpoints."
        )
    )
    parser.add_argument(
        "--model-cache-dir",
        default=str(DEFAULT_EARTH2STUDIO_MODEL_CACHE),
        help="Directory where Earth2Studio should cache the model files.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["pangu24", "graphcastsmall"],
        help="Which Earth2Studio models to prefetch.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used when instantiating the downloaded models. CPU is enough for pure prefetching.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the download summary as JSON instead of a short human-readable summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = download_earth2studio_models(
        model_cache_dir=args.model_cache_dir,
        model_names=args.models,
        device=args.device,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print(f"[earth2studio-download] cache_dir={report['model_cache_dir']}")
    for model_report in report["models"]:
        print(
            "[earth2studio-download] "
            f"model={model_report['model_name']} "
            f"lead_time_hours={model_report['lead_time_hours']} "
            f"vars={model_report['input_variable_count']} "
            f"grid={model_report['input_lat_size']}x{model_report['input_lon_size']} "
            f"device={model_report['device']}"
        )


if __name__ == "__main__":
    main()
