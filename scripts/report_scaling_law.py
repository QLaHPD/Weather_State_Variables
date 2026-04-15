from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from weather_state_variables.scaling import build_main_model_scaling_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a CPU-only scaling-law report for the main forecast model using "
            "parameter count and train-split patch tokens."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/model_config.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--skip-forward",
        action="store_true",
        help="Skip the CPU forward pass and only report parameter and token counts.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the final report as JSON only.",
    )
    return parser.parse_args()


def _format_billions(value: float) -> str:
    return f"{value:.3f}B"


def main() -> None:
    args = parse_args()
    report = build_main_model_scaling_report(
        args.config,
        run_forward=not args.skip_forward,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return

    print("CPU-Only Scaling Law Report")
    print(f"config_path: {report.config_path}")
    print(f"model_device: {report.model_device}")
    print(f"model_dtype: {report.model_dtype}")
    print(
        "parameter_count: "
        f"{report.parameter_count:,} "
        f"({report.parameter_count / 1.0e6:.3f}M)"
    )
    print(
        "parameter_size: "
        f"{report.parameter_size_bytes:,} bytes "
        f"({report.parameter_size_mib:.2f} MiB / {report.parameter_size_gib:.2f} GiB)"
    )
    print(
        "patching: "
        f"input={report.input_size} "
        f"resized={report.resized_input_size} "
        f"patch_size={report.patch_size} "
        f"patch_grid={report.patch_grid}"
    )
    print(f"tokens_per_sample: {report.tokens_per_sample:,}")
    print(
        "train_split: "
        f"samples={report.train_samples:,} "
        f"window={report.train_window_start} -> {report.train_window_end}"
    )
    print(
        "unique_train_tokens: "
        f"{report.train_unique_tokens:,} "
        f"({_format_billions(report.train_unique_tokens_billions)})"
    )
    print(
        "chinchilla_target_tokens: "
        f"{report.chinchilla_target_tokens:,} "
        f"({_format_billions(report.chinchilla_target_tokens_billions)}) "
        f"at {report.chinchilla_tokens_per_parameter:.1f} tokens/parameter"
    )
    print(
        "unique_train_ratio_to_chinchilla: "
        f"{report.unique_train_ratio_to_chinchilla:.3f} "
        f"({report.unique_train_verdict})"
    )
    print(
        "epochs_of_full_split_to_reach_chinchilla: "
        f"{report.epochs_of_full_split_to_reach_chinchilla:.2f}"
    )
    print(
        "single_process_schedule: "
        f"batches_per_epoch={report.single_process_batches_per_epoch:,} "
        f"full_split_each_epoch={report.full_split_each_epoch_single_process} "
        f"samples_per_epoch={report.scheduled_samples_per_epoch_single_process:,} "
        f"max_epochs={report.max_epochs}"
    )
    print(
        "scheduled_train_tokens_single_process: "
        f"{report.scheduled_train_tokens_single_process:,} "
        f"({_format_billions(report.scheduled_train_tokens_single_process_billions)})"
    )
    print(
        "scheduled_ratio_to_chinchilla_single_process: "
        f"{report.scheduled_ratio_to_chinchilla_single_process:.3f} "
        f"({report.scheduled_verdict_single_process})"
    )
    if report.forward_ran:
        print(f"forward_forecast_shape: {report.forecast_shape}")
        print(f"forward_second_block_features_shape: {report.second_block_features_shape}")

    print()
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
