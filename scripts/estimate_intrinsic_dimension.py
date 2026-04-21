from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from weather_state_variables.training import estimate_main_model_intrinsic_dimension
from weather_state_variables.training.pipeline import _to_plain_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the intrinsic dimension of a main-model latent using a "
            "nearest-neighbor intrinsic-dimension estimator."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/model_config.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional main-model checkpoint path. Defaults to train_main.output_dir/train_main.checkpoint_name.",
    )
    parser.add_argument(
        "--bottleneck-compressor-checkpoint",
        "--compressor-checkpoint",
        dest="bottleneck_compressor_checkpoint",
        default=None,
        help=(
            "Optional bottleneck-compressor checkpoint path when "
            "--latent-source=bottleneck_compressor. Defaults to the configured best "
            "compressor checkpoint if it exists, otherwise the configured last checkpoint."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="val",
        help="Which configured window to sample latent states from.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch-size override while collecting latent samples.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional num-workers override while collecting latent samples.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=64,
        help="Maximum number of latent samples to collect for the estimator.",
    )
    parser.add_argument(
        "--start-time",
        default=None,
        help="Optional evaluation-window start time override.",
    )
    parser.add_argument(
        "--end-time",
        default=None,
        help="Optional evaluation-window end time override.",
    )
    parser.add_argument(
        "--latent-source",
        choices=("second_block_features", "patch_grid_features", "bottleneck_compressor"),
        default="second_block_features",
        help=(
            "Which latent tensor to flatten into sample vectors. "
            "'bottleneck_compressor' estimates the compressor's z_bottleneck output."
        ),
    )
    parser.add_argument(
        "--method",
        choices=("two_nn", "levina_bickel"),
        default="two_nn",
        help="Intrinsic-dimension estimator to use.",
    )
    parser.add_argument(
        "--k1",
        type=int,
        default=10,
        help="First neighborhood size for Levina-Bickel aggregation. Ignored for Two-NN.",
    )
    parser.add_argument(
        "--k2",
        type=int,
        default=20,
        help="Last neighborhood size for Levina-Bickel aggregation. Ignored for Two-NN.",
    )
    parser.add_argument(
        "--bias-corrected",
        action="store_true",
        help="Use the common MacKay-style bias correction for Levina-Bickel. Ignored for Two-NN.",
    )
    parser.add_argument(
        "--two-nn-discard-fraction",
        type=float,
        default=0.1,
        help=(
            "Discard this top fraction of largest r2/r1 ratios before the Two-NN "
            "origin-constrained line fit. The original paper suggests a high-percentile cutoff such as 90%%."
        ),
    )
    parser.add_argument(
        "--plateau-search",
        action="store_true",
        help=(
            "Run repeated subsample-size analysis from one cached latent pool to search for an effective-ID plateau."
        ),
    )
    parser.add_argument(
        "--plateau-sample-sizes",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional explicit subset sizes for plateau search. If omitted, the script uses a doubling schedule "
            "such as 128 256 512 ... up to max-samples."
        ),
    )
    parser.add_argument(
        "--plateau-min-samples",
        type=int,
        default=128,
        help="Minimum subset size for the automatic plateau-search schedule.",
    )
    parser.add_argument(
        "--plateau-repeats",
        type=int,
        default=8,
        help="How many random prefix-subset repeats to run per plateau-search sample size.",
    )
    parser.add_argument(
        "--plateau-seed",
        type=int,
        default=0,
        help="Random seed for plateau-search subset permutations.",
    )
    parser.add_argument(
        "--plateau-relative-tolerance",
        type=float,
        default=0.1,
        help=(
            "Relative range tolerance used to declare a plateau from the repeated sample-size curve."
        ),
    )
    parser.add_argument(
        "--plateau-min-points",
        type=int,
        default=2,
        help="Minimum number of consecutive sample sizes required for a detected plateau.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Optional job count for the nearest-neighbor search.",
    )
    parser.add_argument(
        "--print-model-summary",
        action="store_true",
        help="Print the model summary before loading the checkpoint and collecting latents.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the final report as JSON only.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the latent-sample generation progress bar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = estimate_main_model_intrinsic_dimension(
        args.config,
        checkpoint_path=args.checkpoint,
        bottleneck_compressor_checkpoint_path=args.bottleneck_compressor_checkpoint,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        start_time=args.start_time,
        end_time=args.end_time,
        latent_source=args.latent_source,
        method=args.method,
        k1=args.k1,
        k2=args.k2,
        bias_correction=args.bias_corrected,
        two_nn_discard_fraction=args.two_nn_discard_fraction,
        plateau_search=args.plateau_search,
        plateau_sample_sizes=args.plateau_sample_sizes,
        plateau_min_samples=args.plateau_min_samples,
        plateau_repeats=args.plateau_repeats,
        plateau_seed=args.plateau_seed,
        plateau_relative_tolerance=args.plateau_relative_tolerance,
        plateau_min_points=args.plateau_min_points,
        show_progress=not args.json and not args.no_progress,
        print_model_summary=args.print_model_summary,
        n_jobs=args.n_jobs,
        print_result=not args.json,
    )
    if args.json and int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(_to_plain_data(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
