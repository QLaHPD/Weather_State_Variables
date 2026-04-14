from pathlib import Path
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from weather_state_variables.config import load_config_section
from weather_state_variables.data import ArcoEra5FuXiDataConfig, build_fuxi_channel_names
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
import weather_state_variables.training.pipeline as training_pipeline


class _IdentityDenormForecastDataset(Dataset[dict[str, torch.Tensor]]):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        x = torch.zeros(2, 70, 3, 1)
        temb = torch.zeros(12)
        static_features = torch.zeros(5, 3, 1)
        target = torch.zeros(1, 70, 3, 1)
        if index == 0:
            target[0, 0, 1, 0] = 1.0
        return {
            "x": x,
            "temb": temb,
            "static_features": static_features,
            "target": target,
        }

    def denormalize_dynamic_tensor(self, dynamic: torch.Tensor) -> torch.Tensor:
        return dynamic


class _ZeroForecastModel(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        static_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del temb, static_features
        batch_size, _time_steps, channels, height, width = x.shape
        return {
            "forecast": torch.zeros(
                batch_size,
                1,
                channels,
                height,
                width,
                device=x.device,
                dtype=x.dtype,
            )
        }


class TestTrainingSmoke(unittest.TestCase):
    def test_training_configs_expose_gradient_accumulation(self) -> None:
        main_config = MainTrainingConfig.from_yaml()
        intrinsic_config = IntrinsicTrainingConfig.from_yaml()
        _, main_yaml = load_config_section("train_main")
        _, intrinsic_yaml = load_config_section("train_intrinsic")

        self.assertGreaterEqual(main_config.gradient_accumulation_steps, 1)
        self.assertEqual(main_config.forecast_loss, "charbonnier")
        self.assertEqual(main_config.amp_dtype, "bfloat16")
        self.assertEqual(main_config.save_epoch_checkpoint, bool(main_yaml.get("save_epoch_checkpoint", True)))
        self.assertEqual(main_config.save_best_checkpoint, bool(main_yaml.get("save_best_checkpoint", True)))
        self.assertIsNone(main_config.resume_checkpoint_path)
        self.assertEqual(main_config.save_every_train_batches, main_yaml.get("save_every_train_batches"))
        self.assertEqual(main_config.save_every_optimizer_steps, main_yaml.get("save_every_optimizer_steps"))
        self.assertGreaterEqual(intrinsic_config.gradient_accumulation_steps, 1)
        self.assertEqual(
            intrinsic_config.save_epoch_checkpoint,
            bool(intrinsic_yaml.get("save_epoch_checkpoint", True)),
        )
        self.assertEqual(
            intrinsic_config.save_best_checkpoint,
            bool(intrinsic_yaml.get("save_best_checkpoint", True)),
        )
        self.assertIsNone(intrinsic_config.resume_checkpoint_path)
        self.assertEqual(
            intrinsic_config.save_every_train_batches,
            intrinsic_yaml.get("save_every_train_batches"),
        )
        self.assertEqual(
            intrinsic_config.save_every_optimizer_steps,
            intrinsic_yaml.get("save_every_optimizer_steps"),
        )

    def test_step_checkpoint_path_includes_zero_padded_optimizer_step(self) -> None:
        path = training_pipeline._step_checkpoint_path(Path("runs/main"), "main_last.pt", 12)

        self.assertEqual(path, Path("runs/main/main_last_step_00000012.pt"))

    def test_default_rollout_anchor_stride_uses_full_predicted_window_when_inputs_match(self) -> None:
        data_config = ArcoEra5FuXiDataConfig(
            input_time_offsets_hours=(-3, 0),
            lead_time_hours=3,
            forecast_steps=2,
        )

        stride_hours = training_pipeline._default_rollout_anchor_stride_hours(data_config)

        self.assertEqual(stride_hours, 6)

    def test_forecast_rollout_plot_groups_cover_all_channels(self) -> None:
        data_config = ArcoEra5FuXiDataConfig()

        plot_groups = training_pipeline._forecast_rollout_plot_groups(data_config)

        self.assertEqual(len(plot_groups), len(data_config.upper_air_variables) + len(data_config.surface_variables))
        self.assertEqual(sum(len(group.channel_indices) for group in plot_groups), len(data_config.channel_names))
        self.assertEqual(plot_groups[0].row_labels[0], f"{data_config.pressure_levels[0]} hPa")
        self.assertEqual(plot_groups[-1].row_labels, ("surface",))

    def test_resume_state_for_epoch_checkpoint_starts_next_epoch(self) -> None:
        checkpoint = {
            "epoch": 3,
            "optimizer_step": 12,
            "global_batch_step": 96,
            "best_val_loss": 1.25,
            "history": [
                {"epoch": 1.0, "global_batch_steps": 32.0, "optimizer_steps": 4.0, "val_loss": 2.0},
                {"epoch": 2.0, "global_batch_steps": 64.0, "optimizer_steps": 8.0, "val_loss": 1.5},
                {"epoch": 3.0, "global_batch_steps": 96.0, "optimizer_steps": 12.0, "val_loss": 1.25},
            ],
        }

        state = training_pipeline._build_resume_state(
            checkpoint,
            checkpoint_path=Path("runs/main/main_last.pt"),
            total_train_batches=32,
            accumulation_steps=8,
        )

        self.assertEqual(state.start_epoch, 4)
        self.assertIsNone(state.resume_epoch)
        self.assertEqual(state.optimizer_steps, 12)
        self.assertEqual(state.global_batch_steps, 96)
        self.assertAlmostEqual(state.best_val_loss, 1.25)

    def test_resume_state_for_mid_epoch_checkpoint_replays_open_accumulation_window(self) -> None:
        checkpoint = {
            "epoch": 3,
            "batch_index_within_epoch": 20,
            "optimizer_step": 14,
            "history": [
                {"epoch": 1.0, "global_batch_steps": 50.0, "optimizer_steps": 7.0, "val_loss": 2.0},
                {"epoch": 2.0, "global_batch_steps": 100.0, "optimizer_steps": 14.0, "val_loss": 1.8},
            ],
        }

        state = training_pipeline._build_resume_state(
            checkpoint,
            checkpoint_path=Path("runs/main/main_last_step_00000020.pt"),
            total_train_batches=50,
            accumulation_steps=8,
        )

        self.assertEqual(state.start_epoch, 3)
        self.assertEqual(state.resume_epoch, 3)
        self.assertEqual(state.resume_batch_index, 20)
        self.assertEqual(state.replay_start_batch_index, 16)
        self.assertEqual(state.global_batch_steps, 120)

    def test_resume_state_uses_history_best_val_when_checkpoint_value_is_stale(self) -> None:
        checkpoint = {
            "epoch": 3,
            "optimizer_step": 12,
            "global_batch_step": 96,
            "best_val_loss": 1.5,
            "history": [
                {"epoch": 1.0, "global_batch_steps": 32.0, "optimizer_steps": 4.0, "val_loss": 2.0},
                {"epoch": 2.0, "global_batch_steps": 64.0, "optimizer_steps": 8.0, "val_loss": 1.5},
                {"epoch": 3.0, "global_batch_steps": 96.0, "optimizer_steps": 12.0, "val_loss": 1.25},
            ],
        }

        state = training_pipeline._build_resume_state(
            checkpoint,
            checkpoint_path=Path("runs/main/main_best.pt"),
            total_train_batches=32,
            accumulation_steps=8,
        )

        self.assertAlmostEqual(state.best_val_loss, 1.25)

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

    def test_evaluate_main_forecast_model_reports_per_variable_metrics(self) -> None:
        data_config = ArcoEra5FuXiDataConfig(forecast_steps=1)
        criterion = LatitudeWeightedCharbonnierLoss(
            data_config.channel_names,
            epsilon=1.0e-3,
            upper_air_weight=1.0,
            surface_weight=0.1,
            latitude_descending=True,
        )
        loader = DataLoader(_IdentityDenormForecastDataset(), batch_size=2, shuffle=False)
        runtime = training_pipeline.DistributedRuntime(
            enabled=False,
            backend=None,
            rank=0,
            local_rank=0,
            world_size=1,
            device=torch.device("cpu"),
        )

        result = training_pipeline._evaluate_main_forecast_model(
            _ZeroForecastModel(),
            loader,
            criterion=criterion,
            data_config=data_config,
            runtime=runtime,
            use_amp=False,
            amp_dtype=None,
            max_batches=None,
        )

        self.assertIn("loss", result)
        self.assertIn("variable_losses", result)
        self.assertIn("denorm_mae", result)
        self.assertIn("variable_denorm_mae", result)
        self.assertGreater(result["variable_losses"]["geopotential"], result["variable_losses"]["temperature"])
        self.assertGreater(
            result["variable_denorm_mae"]["geopotential"],
            result["variable_denorm_mae"]["temperature"],
        )
        self.assertEqual(result["batches"], 1)


if __name__ == "__main__":
    unittest.main()
