from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from weather_state_variables.data import DEFAULT_ARCO_ERA5_URL, inspect_arco_era5_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect an ARCO ERA5 Zarr dataset for FuXi-style compatibility.")
    parser.add_argument(
        "--dataset-url",
        default=DEFAULT_ARCO_ERA5_URL,
        help="ARCO ERA5 dataset URL. Supports gs://..., storage.googleapis.com, or console.cloud.google.com URLs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = inspect_arco_era5_dataset(args.dataset_url)
    print(json.dumps(report.summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
