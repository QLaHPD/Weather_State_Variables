from __future__ import annotations

import argparse
import json
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
            "Estimate the intrinsic dimension of a main-model latent using the "
            "Levina-Bickel nearest-neighbor estimator."
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
        choices=("second_block_features", "patch_grid_features"),
        default="second_block_features",
        help="Which encoder latent tensor to flatten into sample vectors.",
    )
    parser.add_argument(
        "--k1",
        type=int,
        default=10,
        help="First neighborhood size for Levina-Bickel aggregation.",
    )
    parser.add_argument(
        "--k2",
        type=int,
        default=20,
        help="Last neighborhood size for Levina-Bickel aggregation.",
    )
    parser.add_argument(
        "--bias-corrected",
        action="store_true",
        help="Use the common MacKay-style bias correction (k-2 numerator) instead of the original k-1 form.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = estimate_main_model_intrinsic_dimension(
        args.config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        start_time=args.start_time,
        end_time=args.end_time,
        latent_source=args.latent_source,
        k1=args.k1,
        k2=args.k2,
        bias_correction=args.bias_corrected,
        print_model_summary=args.print_model_summary,
        n_jobs=args.n_jobs,
        print_result=not args.json,
    )
    if args.json:
        print(json.dumps(_to_plain_data(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
