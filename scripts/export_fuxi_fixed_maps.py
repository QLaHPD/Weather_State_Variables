from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from weather_state_variables.data import (  # noqa: E402
    ArcoEra5FuXiDataConfig,
    build_fuxi_derived_static_maps,
    load_arco_static_source_maps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the FuXi fixed maps from an ARCO ERA5-compatible dataset. "
            "By default this writes the three derived maps and records where the "
            "two raw-source maps come from."
        )
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the .npy fixed-map files will be written.",
    )
    parser.add_argument(
        "--dataset-url",
        default=None,
        help="Optional dataset path or gs:// URL override. Defaults to the data config value.",
    )
    parser.add_argument(
        "--config",
        default="configs/model_config.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Also export the raw-source maps land_sea_mask.npy and orography.npy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_config = ArcoEra5FuXiDataConfig.from_yaml(args.config)
    dataset_url = args.dataset_url or data_config.dataset_url
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_maps = load_arco_static_source_maps(
        dataset_url,
        orography_source=data_config.orography_source,
        convert_geopotential_to_height=data_config.convert_geopotential_to_height,
        latitude_descending=data_config.latitude_descending,
        gcs_token=data_config.gcs_token,
    )
    derived_maps = build_fuxi_derived_static_maps(
        source_maps["latitude"],
        source_maps["longitude"],
    )

    np.save(output_dir / "cos_latitude.npy", derived_maps["cos_latitude"])
    np.save(output_dir / "cos_longitude.npy", derived_maps["cos_longitude"])
    np.save(output_dir / "sin_longitude.npy", derived_maps["sin_longitude"])

    if args.include_raw:
        np.save(output_dir / "land_sea_mask.npy", source_maps["land_sea_mask"])
        np.save(output_dir / "orography.npy", source_maps["orography"])

    summary = {
        "dataset_url": str(dataset_url),
        "output_dir": str(output_dir),
        "generated_maps": ["cos_latitude", "cos_longitude", "sin_longitude"],
        "raw_source_maps": {
            "land_sea_mask": "dataset variable 'land_sea_mask'",
            "orography": (
                "dataset variable "
                f"'{data_config.orography_source}'"
                + (
                    " converted from geopotential to height"
                    if data_config.convert_geopotential_to_height
                    and data_config.orography_source == "geopotential_at_surface"
                    else ""
                )
            ),
        },
        "included_raw_exports": args.include_raw,
        "shape": list(derived_maps["cos_latitude"].shape),
        "latitude_size": int(source_maps["latitude"].shape[0]),
        "longitude_size": int(source_maps["longitude"].shape[0]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
