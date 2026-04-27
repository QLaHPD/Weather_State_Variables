from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from weather_state_variables.training import train_latent_dynamics_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the latent neural state vector field on frozen intrinsic trajectories "
            "encoded from the main-model bottleneck."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/model_config.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run model print/summary and a random rollout without loading data or training.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Optional checkpoint path to resume latent-dynamics training from.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_latent_dynamics_model(
        args.config,
        smoke_only=args.smoke_only,
        resume_checkpoint_path=args.resume_checkpoint,
    )


if __name__ == "__main__":
    main()
