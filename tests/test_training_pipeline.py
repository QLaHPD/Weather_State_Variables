import unittest

import torch

from weather_state_variables.data import build_fuxi_channel_names
from weather_state_variables.models import (
    FuXiIntrinsic,
    FuXiIntrinsicConfig,
    FuXiLowerRes,
    FuXiLowerResConfig,
    FuXiLowerResEncoder,
)
from weather_state_variables.training import (
    IntrinsicTrainingConfig,
    LatitudeWeightedCharbonnierLoss,
    MainTrainingConfig,
    run_intrinsic_model_smoke_test,
    run_main_model_smoke_test,
)


class TestTrainingSmoke(unittest.TestCase):
    def test_training_configs_expose_gradient_accumulation(self) -> None:
        main_config = MainTrainingConfig.from_yaml()
        intrinsic_config = IntrinsicTrainingConfig.from_yaml()

        self.assertGreaterEqual(main_config.gradient_accumulation_steps, 1)
        self.assertEqual(main_config.forecast_loss, "charbonnier")
        self.assertEqual(main_config.amp_dtype, "bfloat16")
        self.assertGreaterEqual(intrinsic_config.gradient_accumulation_steps, 1)

    def test_charbonnier_loss_downweights_surface_channels(self) -> None:
        criterion = LatitudeWeightedCharbonnierLoss(
            build_fuxi_channel_names(),
            epsilon=1.0e-3,
            upper_air_weight=1.0,
            surface_weight=0.1,
            latitude_descending=True,
        )
        prediction = torch.zeros(1, 1, 70, 1, 1)
        baseline_target = prediction.clone()

        upper_target = prediction.clone()
        upper_target[:, :, 0, 0, 0] = 1.0
        surface_target = prediction.clone()
        surface_target[:, :, -1, 0, 0] = 1.0

        baseline_loss = criterion(prediction, baseline_target)
        upper_loss = criterion(prediction, upper_target)
        surface_loss = criterion(prediction, surface_target)

        upper_delta = float(upper_loss - baseline_loss)
        surface_delta = float(surface_loss - baseline_loss)

        self.assertGreater(upper_delta, surface_delta * 9.0)

    def test_charbonnier_loss_upweights_equatorial_latitudes(self) -> None:
        criterion = LatitudeWeightedCharbonnierLoss(
            build_fuxi_channel_names(),
            epsilon=1.0e-3,
            upper_air_weight=1.0,
            surface_weight=0.1,
            latitude_descending=True,
        )
        prediction = torch.zeros(1, 1, 70, 3, 1)

        pole_target = prediction.clone()
        pole_target[:, :, 0, 0, 0] = 1.0
        equator_target = prediction.clone()
        equator_target[:, :, 0, 1, 0] = 1.0

        pole_loss = criterion(prediction, pole_target)
        equator_loss = criterion(prediction, equator_target)

        self.assertGreater(float(equator_loss), float(pole_loss))

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
            device="cpu",
            dtype=torch.float32,
        )
        model = FuXiLowerRes(model_config)

        report = run_main_model_smoke_test(model, batch_size=2, print_outputs=False)

        self.assertEqual(report["output"]["forecast"]["shape"], [2, 2, 8, 17, 32])
        self.assertEqual(report["output"]["second_block_features"]["shape"], [2, 16, 2, 4])

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
            device="cpu",
            dtype=torch.float32,
        )
        intrinsic_config = FuXiIntrinsicConfig(
            feature_channels=16,
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

        self.assertEqual(report["second_block_features"]["shape"], [2, 16, 2, 4])
        self.assertEqual(report["intrinsic_output"]["z_intrinsic"]["shape"], [2, 3])
        self.assertEqual(report["intrinsic_output"]["second_block_features_recon"]["shape"], [2, 16, 2, 4])


if __name__ == "__main__":
    unittest.main()
