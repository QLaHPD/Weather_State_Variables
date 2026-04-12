from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from weather_state_variables.data import (  # noqa: E402
    download_arco_era5_subset,
    repair_local_zarr_time_consistency,
    resolve_arco_era5_download_window,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the ARCO ERA5 variables, levels, and time window needed by the "
            "current FuXi-style training config into a local Zarr store."
        )
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Local output path for the downloaded Zarr store.",
    )
    parser.add_argument(
        "--config",
        default="configs/model_config.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--dataset-url",
        default=None,
        help="Optional ARCO ERA5 source dataset URL override.",
    )
    parser.add_argument(
        "--start-time",
        default=None,
        help=(
            "Optional raw source start time. If omitted, the script uses the padded union "
            "of the configured training and validation windows."
        ),
    )
    parser.add_argument(
        "--end-time",
        default=None,
        help=(
            "Optional raw source end time. If omitted, the script uses the padded union "
            "of the configured training and validation windows."
        ),
    )
    parser.add_argument(
        "--chunk-hours",
        type=int,
        default=24,
        help="Number of hourly timesteps to copy per write chunk.",
    )
    parser.add_argument(
        "--gcs-token",
        default=None,
        help="Optional gcsfs token override. Defaults to the config value, usually 'anon'.",
    )
    parser.add_argument(
        "--skip-statics",
        action="store_true",
        help="Skip raw static source fields and only write dynamic training variables.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output path if it already exists.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing partial output store if it is a prefix of the requested download.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the inferred download window and exit without copying data.",
    )
    parser.add_argument(
        "--repair-output-only",
        action="store_true",
        help="Repair a partially written local output Zarr store and exit without downloading more data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_window = resolve_arco_era5_download_window(args.config)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "config_path": str(Path(args.config).resolve()),
                    "download_window": download_window.summary(),
                    "start_time_override": args.start_time,
                    "end_time_override": args.end_time,
                    "chunk_hours": args.chunk_hours,
                    "skip_statics": args.skip_statics,
                    "resume": args.resume,
                    "progress_enabled": not args.no_progress,
                    "output_path": str(Path(args.output).resolve()),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.repair_output_only:
        summary = repair_local_zarr_time_consistency(args.output, verbose=True)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    summary = download_arco_era5_subset(
        args.output,
        config_path=args.config,
        dataset_url=args.dataset_url,
        start_time=args.start_time,
        end_time=args.end_time,
        include_static_sources=not args.skip_statics,
        overwrite=args.overwrite,
        resume=args.resume,
        chunk_size=args.chunk_hours,
        gcs_token=args.gcs_token,
        verbose=True,
        show_progress=not args.no_progress,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
