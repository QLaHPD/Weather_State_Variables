from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from weather_state_variables.training import rollout_main_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run autoregressive main-model rollout and save one denormalized PNG per channel "
            "for each future lead-time step."
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
        "--intrinsic-checkpoint",
        default=None,
        help=(
            "Optional intrinsic checkpoint path. When set, rollout can trace the intrinsic bottleneck "
            "from the main-model bottleneck. Pair with --intrinsic-frequency to explicitly route "
            "selected autoregressive steps through the intrinsic reconstruction."
        ),
    )
    parser.add_argument(
        "--intrinsic-frequency",
        type=int,
        default=None,
        help=(
            "Explicitly enable intrinsic forecast chaining every N generated rollout steps, counted "
            "1-indexed. For example, --intrinsic-frequency 4 applies the intrinsic reconstruction at "
            "steps 4, 8, 12, ..."
        ),
    )
    parser.add_argument(
        "--save-intrinsic-latent-plot",
        action="store_true",
        help=(
            "Record the intrinsic 2D bottleneck at each rollout step and save a cold-to-hot "
            "latent-space trajectory plot with time-direction arrows. Requires --intrinsic-checkpoint."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="val",
        help="Which configured window to use for the rollout seed sample.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Which valid anchor sample inside the selected window to roll out.",
    )
    parser.add_argument(
        "--future-steps",
        type=int,
        default=8,
        help="How many autoregressive future lead-time steps to generate.",
    )
    parser.add_argument(
        "--start-time",
        default=None,
        help="Optional rollout-window start time override.",
    )
    parser.add_argument(
        "--end-time",
        default=None,
        help="Optional rollout-window end time override.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for channel folders and rollout images.",
    )
    parser.add_argument(
        "--print-model-summary",
        action="store_true",
        help="Print the model summary before loading the checkpoint and generating the rollout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rollout_main_model(
        args.config,
        checkpoint_path=args.checkpoint,
        intrinsic_checkpoint_path=args.intrinsic_checkpoint,
        intrinsic_frequency=args.intrinsic_frequency,
        save_intrinsic_latent_plot=args.save_intrinsic_latent_plot,
        split=args.split,
        sample_index=args.sample_index,
        future_steps=args.future_steps,
        start_time=args.start_time,
        end_time=args.end_time,
        output_dir=args.output_dir,
        print_model_summary=args.print_model_summary,
    )


if __name__ == "__main__":
    main()
