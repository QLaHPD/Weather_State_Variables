from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from weather_state_variables.benchmarking import (
    DEFAULT_BENCHMARK_OUTPUT_DIR,
    DEFAULT_EARTH2STUDIO_MODEL_CACHE,
    compare_forecast_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark our main model against Earth2Studio GraphCastSmall and Pangu24 "
            "on a shared ERA5 evaluation window. Our model and GraphCastSmall are scored "
            "on the 1-degree grid, while Pangu24 is forecast at full resolution and "
            "conservatively remapped to low resolution for evaluation."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/model_config.yaml",
        help="Path to the YAML config file for our model.",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the local full-resolution ERA5 benchmark Zarr store.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint path for our main forecast model.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_BENCHMARK_OUTPUT_DIR),
        help="Directory where metrics JSON/CSV and plots will be written.",
    )
    parser.add_argument(
        "--model-cache-dir",
        default=str(DEFAULT_EARTH2STUDIO_MODEL_CACHE),
        help="Directory where Earth2Studio model files should be cached.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device used for our model and Earth2Studio models. Use auto, cpu, or a cuda device.",
    )
    parser.add_argument(
        "--our-highres-device",
        default=None,
        help=(
            "Deprecated alias for --our-lowres-device. If set and --our-lowres-device is not "
            "provided, this device assignment will be reused for our low-resolution runtime."
        ),
    )
    parser.add_argument(
        "--our-lowres-device",
        default=None,
        help=(
            "Optional override for the low-resolution copy of our model. "
            "Accepts one device like cuda:0 or cpu, or a comma-separated shard list."
        ),
    )
    parser.add_argument(
        "--pangu24-device",
        default=None,
        help="Optional override for Earth2Studio Pangu24, e.g. cuda:3 or cpu.",
    )
    parser.add_argument(
        "--graphcastsmall-device",
        default=None,
        help="Optional override for Earth2Studio GraphCastSmall, e.g. cuda:4 or cpu.",
    )
    parser.add_argument(
        "--start-time",
        default=None,
        help="Optional evaluation init-window start time.",
    )
    parser.add_argument(
        "--end-time",
        default=None,
        help="Optional evaluation init-window end time.",
    )
    parser.add_argument(
        "--init-stride-hours",
        type=int,
        default=24,
        help="Spacing in hours between benchmark initialization times.",
    )
    parser.add_argument(
        "--horizon-hours",
        type=int,
        default=120,
        help="Maximum forecast horizon in hours.",
    )
    parser.add_argument(
        "--max-init-times",
        type=int,
        default=None,
        help="Optional cap on how many initialization times to evaluate.",
    )
    parser.add_argument(
        "--normalization-stats-path",
        default=None,
        help="Optional override for the normalization stats JSON used by our model.",
    )
    parser.add_argument(
        "--highres-metric-step-hours",
        type=int,
        default=24,
        help="Lead-time spacing for the low-resolution Pangu24 comparison.",
    )
    parser.add_argument(
        "--lowres-metric-step-hours",
        type=int,
        default=6,
        help="Lead-time spacing for the low-resolution GraphCast comparison.",
    )
    parser.add_argument(
        "--skip-model-download",
        action="store_true",
        help="Assume the Earth2Studio model cache is already populated and skip the prefetch step.",
    )
    parser.add_argument(
        "--no-cpu-fallback",
        action="store_true",
        help="Disable the automatic CPU fallback if an Earth2Studio model cannot create its GPU ORT session.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the final benchmark summary as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    our_lowres_device = args.our_lowres_device or args.our_highres_device
    report = compare_forecast_models(
        args.config,
        data_path=args.data_path,
        main_checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        model_cache_dir=args.model_cache_dir,
        device=args.device,
        device_map={
            key: value
            for key, value in {
                "ours_lowres": our_lowres_device,
                "pangu24": args.pangu24_device,
                "graphcastsmall": args.graphcastsmall_device,
            }.items()
            if value is not None
        },
        start_time=args.start_time,
        end_time=args.end_time,
        init_stride_hours=args.init_stride_hours,
        horizon_hours=args.horizon_hours,
        max_init_times=args.max_init_times,
        normalization_stats_path=args.normalization_stats_path,
        highres_metric_step_hours=args.highres_metric_step_hours,
        lowres_metric_step_hours=args.lowres_metric_step_hours,
        download_models_first=not args.skip_model_download,
        allow_cpu_fallback=not args.no_cpu_fallback,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print(f"[benchmark] metrics_json={report['metrics_json_path']}")
    print(f"[benchmark] metrics_csv={report['metrics_csv_path']}")
    print(f"[benchmark] variable_metrics_csv={report['variable_metrics_csv_path']}")
    print(f"[benchmark] plot={report['plot_path']}")
    for model_name, plot_path in report.get("individual_plot_paths", {}).items():
        print(f"[benchmark] plot_{model_name}={plot_path}")
    print(f"[benchmark] variable_metric_plot_groups={list(report.get('variable_metric_plot_paths', {}).keys())}")
    print(
        "[benchmark] "
        f"init_times={report['init_time_count']} "
        f"pangu24_leads={report['highres_metric_count']} "
        f"lowres_6h_leads={report['lowres_metric_count']}"
    )
    print(f"[benchmark] device_map_requested={report['device_map_requested']}")
    print(f"[benchmark] device_map_effective={report['device_map_effective']}")


if __name__ == "__main__":
    main()
