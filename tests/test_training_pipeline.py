import unittest

import torch

from weather_state_variables.models import (
    FuXiIntrinsic,
    FuXiIntrinsicConfig,
    FuXiLowerRes,
    FuXiLowerResConfig,
    FuXiLowerResEncoder,
)
from weather_state_variables.training import (
    run_intrinsic_model_smoke_test,
    run_main_model_smoke_test,
)


class TestTrainingSmoke(unittest.TestCase):
    def test_main_smoke_test_runs_on_tiny_model(self) -> None:
        model_config = FuXiLowerResConfig(
            input_size=(17, 32),
            time_steps=2,
            in_chans=8,
            aux_chans=2,
            out_chans=8,
            forecast_steps=2,
            temb_dim=12,
            patch_size=(4, 4),
            embed_dim=16,
            num_heads=4,
            window_size=2,
            depths=(1, 1, 1, 1),
            num_groups=8,
            mlp_hidden_dim=32,
            d_high=6,
            device="cpu",
            dtype=torch.float32,
        )
        model = FuXiLowerRes(model_config)

        report = run_main_model_smoke_test(model, batch_size=2, print_outputs=False)

        self.assertEqual(report["output"]["forecast"]["shape"], [2, 2, 8, 17, 32])
        self.assertEqual(report["output"]["z_high"]["shape"], [2, 6, 2, 4])

    def test_intrinsic_smoke_test_runs_on_tiny_models(self) -> None:
        encoder_config = FuXiLowerResConfig(
            input_size=(17, 32),
            time_steps=2,
            in_chans=8,
            aux_chans=2,
            out_chans=8,
            forecast_steps=2,
            temb_dim=12,
            patch_size=(4, 4),
            embed_dim=16,
            num_heads=4,
            window_size=2,
            depths=(1, 1, 1, 1),
            num_groups=8,
            mlp_hidden_dim=32,
            d_high=6,
            device="cpu",
            dtype=torch.float32,
        )
        intrinsic_config = FuXiIntrinsicConfig(
            d_high=6,
            spatial_size=encoder_config.latent_grid,
            d_intrinsic=3,
            hidden_dims=(24, 12),
            apply_tanh=True,
            device="cpu",
            dtype=torch.float32,
        )
        encoder = FuXiLowerResEncoder(encoder_config)
        intrinsic_model = FuXiIntrinsic(intrinsic_config)

        report = run_intrinsic_model_smoke_test(
            encoder,
            intrinsic_model,
            batch_size=2,
            print_outputs=False,
        )

        self.assertEqual(report["z_high"]["shape"], [2, 6, 2, 4])
        self.assertEqual(report["intrinsic_output"]["z_intrinsic"]["shape"], [2, 3])
        self.assertEqual(report["intrinsic_output"]["z_high_recon"]["shape"], [2, 6, 2, 4])


if __name__ == "__main__":
    unittest.main()
