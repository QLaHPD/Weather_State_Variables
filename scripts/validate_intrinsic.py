from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from weather_state_variables.training import validate_intrinsic_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run standalone validation for the intrinsic model, optionally chained through the main decoder."
    )
    parser.add_argument(
        "--config",
        default="configs/model_config.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--intrinsic-checkpoint",
        default=None,
        help="Optional intrinsic checkpoint path. Defaults to train_intrinsic.output_dir/train_intrinsic.checkpoint_name.",
    )
    parser.add_argument(
        "--main-checkpoint",
        default=None,
        help="Optional main-model checkpoint path. Defaults to the path recorded in the intrinsic checkpoint or train_intrinsic.main_checkpoint_path.",
    )
    parser.add_argument(
        "--split",
        choices=("val", "train"),
        default="val",
        help="Which configured window to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch-size override for validation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional num-workers override for validation.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional max-batches override for validation.",
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
        "--print-model-summary",
        action="store_true",
        help="Print the main and intrinsic model summaries before loading checkpoints and running validation.",
    )
    parser.add_argument(
        "--mode",
        choices=("intrinsic", "chain"),
        default="intrinsic",
        help="Run only intrinsic reconstruction validation, or chain main encoder -> intrinsic -> main decoder.",
    )
    parser.add_argument(
        "--save-rollout-plots",
        action="store_true",
        help="Save autoregressive rollout plots in chain mode.",
    )
    parser.add_argument(
        "--rollout-output-dir",
        default=None,
        help="Optional output directory for saved rollout figures and graphs.",
    )
    parser.add_argument(
        "--rollout-samples",
        type=int,
        default=1,
        help="How many validation samples to visualize when rollout plotting is enabled.",
    )
    parser.add_argument(
        "--rollout-passes",
        type=int,
        default=3,
        help="How many autoregressive forward passes to run per plotted sample.",
    )
    parser.add_argument(
        "--rollout-anchor-stride-hours",
        type=int,
        default=None,
        help="Optional anchor-step override for the autoregressive rollout in hours.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_intrinsic_model(
        args.config,
        intrinsic_checkpoint_path=args.intrinsic_checkpoint,
        main_checkpoint_path=args.main_checkpoint,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        start_time=args.start_time,
        end_time=args.end_time,
        print_model_summary=args.print_model_summary,
        mode=args.mode,
        save_rollout_plots=args.save_rollout_plots,
        rollout_output_dir=args.rollout_output_dir,
        rollout_samples=args.rollout_samples,
        rollout_passes=args.rollout_passes,
        rollout_anchor_stride_hours=args.rollout_anchor_stride_hours,
    )


if __name__ == "__main__":
    main()
