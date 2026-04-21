from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from weather_state_variables.config import load_config_section
from weather_state_variables.data import ArcoEra5FuXiDataConfig, build_fuxi_channel_names
from weather_state_variables.models import (
    FuXiBottleneckCompressor,
    FuXiBottleneckCompressorConfig,
    FuXiIntrinsic,
    FuXiIntrinsicConfig,
    FuXiLowerRes,
    FuXiLowerResConfig,
    FuXiLowerResEncoder,
)
from weather_state_variables.training import (
    BottleneckCompressorTrainingConfig,
    IntrinsicTrainingConfig,
    LatitudeWeightedCharbonnierLoss,
    MainTrainingConfig,
    run_bottleneck_compressor_smoke_test,
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
        compressor_config = BottleneckCompressorTrainingConfig.from_yaml()
        _, main_yaml = load_config_section("train_main")
        _, intrinsic_yaml = load_config_section("train_intrinsic")
        _, compressor_yaml = load_config_section("train_bottleneck_compressor")

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
            intrinsic_config.detach_second_block_features,
            bool(intrinsic_yaml.get("detach_second_block_features", intrinsic_yaml.get("detach_z_high", False))),
        )
        self.assertEqual(
            intrinsic_config.save_every_train_batches,
            intrinsic_yaml.get("save_every_train_batches"),
        )
        self.assertEqual(
            intrinsic_config.save_every_optimizer_steps,
            intrinsic_yaml.get("save_every_optimizer_steps"),
        )
        self.assertGreaterEqual(compressor_config.gradient_accumulation_steps, 1)
        self.assertEqual(
            compressor_config.save_epoch_checkpoint,
            bool(compressor_yaml.get("save_epoch_checkpoint", True)),
        )
        self.assertEqual(
            compressor_config.save_best_checkpoint,
            bool(compressor_yaml.get("save_best_checkpoint", True)),
        )
        self.assertIsNone(compressor_config.resume_checkpoint_path)
        self.assertEqual(
            compressor_config.detach_second_block_features,
            bool(
                compressor_yaml.get(
                    "detach_second_block_features",
                    compressor_yaml.get("detach_encoder_features", True),
                )
            ),
        )
        self.assertEqual(
            compressor_config.save_every_train_batches,
            compressor_yaml.get("save_every_train_batches"),
        )
        self.assertEqual(
            compressor_config.save_every_optimizer_steps,
            compressor_yaml.get("save_every_optimizer_steps"),
        )

    def test_build_intrinsic_training_objects_preserves_intrinsic_feature_controls(self) -> None:
        config_path, intrinsic_yaml = load_config_section("intrinsic_model")
        _, encoder_config, intrinsic_config, _data_config, runtime, _model_dtype, _amp_dtype = (
            training_pipeline._build_intrinsic_training_objects(config_path)
        )

        try:
            expected_resblocks = tuple(
                int(value)
                for value in intrinsic_yaml.get(
                    "resblocks_per_stage",
                    intrinsic_yaml.get("depths"),
                )
            )
            if len(expected_resblocks) == 2:
                expected_resblocks = (
                    expected_resblocks[0],
                    expected_resblocks[1],
                    expected_resblocks[1],
                )

            self.assertEqual(intrinsic_config.input_channels, encoder_config.embed_dim)
            self.assertEqual(intrinsic_config.feature_channels, int(intrinsic_yaml["feature_channels"]))
            self.assertEqual(intrinsic_config.resblocks_per_stage, expected_resblocks)
        finally:
            training_pipeline._cleanup_distributed_runtime(runtime)

    def test_build_bottleneck_compressor_training_objects_targets_encoder_bottleneck(self) -> None:
        config_path, compressor_yaml = load_config_section("bottleneck_compressor_model")
        _, encoder_config, compressor_config, _data_config, runtime, _model_dtype, _amp_dtype = (
            training_pipeline._build_bottleneck_compressor_training_objects(config_path)
        )

        try:
            self.assertEqual(compressor_config.input_channels, encoder_config.embed_dim)
            self.assertEqual(compressor_config.feature_source, compressor_yaml["feature_source"])
            self.assertEqual(compressor_config.spatial_size, encoder_config.latent_grid)
            self.assertEqual(compressor_config.bottleneck_channels, 1)
            self.assertEqual(compressor_config.positional_embedding, "learned_2d")
        finally:
            training_pipeline._cleanup_distributed_runtime(runtime)

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

    def test_forecast_rollout_channel_specs_cover_all_channels(self) -> None:
        data_config = ArcoEra5FuXiDataConfig()

        specs = training_pipeline._forecast_rollout_channel_specs(data_config)

        self.assertEqual(len(specs), len(data_config.channel_names))
        self.assertEqual(specs[0].folder_name, "geopotential_50hpa")
        self.assertEqual(specs[0].channel_name, data_config.channel_names[0])
        self.assertEqual(specs[len(data_config.pressure_levels)].folder_name, "temperature_50hpa")
        self.assertEqual(specs[-1].folder_name, "total_precipitation")

    def test_should_use_intrinsic_for_rollout_step_respects_frequency(self) -> None:
        dummy_model = nn.Identity()

        self.assertFalse(
            training_pipeline._should_use_intrinsic_for_rollout_step(
                rollout_step=0,
                intrinsic_model=None,
                intrinsic_frequency=None,
            )
        )
        self.assertTrue(
            training_pipeline._should_use_intrinsic_for_rollout_step(
                rollout_step=0,
                intrinsic_model=dummy_model,
                intrinsic_frequency=None,
            )
        )
        self.assertFalse(
            training_pipeline._should_use_intrinsic_for_rollout_step(
                rollout_step=0,
                intrinsic_model=dummy_model,
                intrinsic_frequency=4,
            )
        )
        self.assertTrue(
            training_pipeline._should_use_intrinsic_for_rollout_step(
                rollout_step=3,
                intrinsic_model=dummy_model,
                intrinsic_frequency=4,
            )
        )
        self.assertTrue(
            training_pipeline._should_use_intrinsic_for_rollout_step(
                rollout_step=7,
                intrinsic_model=dummy_model,
                intrinsic_frequency=4,
            )
        )

    def test_levina_bickel_estimator_recovers_two_dimensional_plane(self) -> None:
        rng = np.random.default_rng(0)
        base = rng.normal(size=(256, 2)).astype(np.float32)
        embedded = np.concatenate(
            [base, np.zeros((256, 4), dtype=np.float32)],
            axis=1,
        )

        report = training_pipeline._estimate_levina_bickel_dimension(
            embedded,
            k1=10,
            k2=20,
            bias_correction=False,
            n_jobs=1,
        )

        self.assertGreater(report["dimension_estimate"], 1.5)
        self.assertLess(report["dimension_estimate"], 2.5)
        self.assertEqual(report["rounded_dimension_estimate"], 2)

    def test_two_nn_estimator_recovers_two_dimensional_plane(self) -> None:
        rng = np.random.default_rng(0)
        base = rng.normal(size=(512, 2)).astype(np.float32)
        embedded = np.concatenate(
            [base, np.zeros((512, 4), dtype=np.float32)],
            axis=1,
        )

        report = training_pipeline._estimate_two_nn_dimension(
            embedded,
            discard_fraction=0.1,
            n_jobs=1,
        )

        self.assertGreater(report["dimension_estimate"], 1.4)
        self.assertLess(report["dimension_estimate"], 2.6)
        self.assertEqual(report["rounded_dimension_estimate"], 2)
        self.assertGreater(report["used_ratio_count"], 100)

    def test_two_nn_estimator_from_cached_distances_matches_direct_estimator(self) -> None:
        rng = np.random.default_rng(0)
        base = rng.normal(size=(256, 2)).astype(np.float32)
        embedded = np.concatenate(
            [base, np.zeros((256, 4), dtype=np.float32)],
            axis=1,
        )

        direct = training_pipeline._estimate_two_nn_dimension(
            embedded,
            discard_fraction=0.1,
            n_jobs=1,
        )
        distance_matrix, feature_dim = training_pipeline._compute_pairwise_distance_matrix(
            embedded,
            n_jobs=1,
        )
        cached = training_pipeline._estimate_two_nn_dimension_from_distances(
            distance_matrix,
            discard_fraction=0.1,
            feature_dim=feature_dim,
        )

        self.assertAlmostEqual(direct["dimension_estimate"], cached["dimension_estimate"], places=6)
        self.assertEqual(cached["feature_dim"], embedded.shape[1])

    def test_default_plateau_sample_sizes_double_to_max(self) -> None:
        sample_sizes = training_pipeline._default_plateau_sample_sizes(
            max_samples=1024,
            min_samples=128,
        )

        self.assertEqual(sample_sizes, [128, 256, 512, 1024])

    def test_detect_intrinsic_dimension_plateau_prefers_long_stable_window(self) -> None:
        plateau = training_pipeline._detect_intrinsic_dimension_plateau(
            [
                {"sample_size": 64, "mean_dimension_estimate": 1.4, "stderr_dimension_estimate": 0.2},
                {"sample_size": 128, "mean_dimension_estimate": 2.02, "stderr_dimension_estimate": 0.05},
                {"sample_size": 256, "mean_dimension_estimate": 2.06, "stderr_dimension_estimate": 0.04},
                {"sample_size": 512, "mean_dimension_estimate": 2.01, "stderr_dimension_estimate": 0.03},
                {"sample_size": 1024, "mean_dimension_estimate": 2.45, "stderr_dimension_estimate": 0.07},
            ],
            relative_tolerance=0.05,
            min_plateau_points=2,
        )

        self.assertTrue(plateau["found"])
        self.assertEqual(plateau["start_sample_size"], 128)
        self.assertEqual(plateau["end_sample_size"], 512)
        self.assertGreater(plateau["dimension_estimate"], 1.9)
        self.assertLess(plateau["dimension_estimate"], 2.1)

    def test_plateau_search_reuses_single_latent_pool(self) -> None:
        rng = np.random.default_rng(0)
        base = rng.normal(size=(256, 2)).astype(np.float32)
        embedded = np.concatenate(
            [base, np.zeros((256, 4), dtype=np.float32)],
            axis=1,
        )
        distance_matrix, feature_dim = training_pipeline._compute_pairwise_distance_matrix(
            embedded,
            n_jobs=1,
        )
        estimator = training_pipeline._build_intrinsic_dimension_estimator(
            method="two_nn",
            k1=10,
            k2=20,
            bias_correction=False,
            two_nn_discard_fraction=0.1,
        )

        report = training_pipeline._run_intrinsic_dimension_plateau_search(
            distance_matrix,
            estimator=estimator,
            feature_dim=feature_dim,
            sample_sizes=[64, 128, 256],
            repeats=3,
            seed=0,
            relative_tolerance=0.2,
            min_plateau_points=2,
        )

        self.assertEqual(report["latent_pool_sample_count"], 256)
        self.assertEqual(report["sample_sizes"], [64, 128, 256])
        self.assertEqual(len(report["curve"]), 3)
        self.assertEqual(report["curve"][-1]["repeat_count"], 3)
        self.assertIn("single_cached_latent_pool", report["sampling_strategy"])

    def test_fixed_index_shard_sampler_partitions_without_padding_or_duplicates(self) -> None:
        indices = list(range(11))

        shards = [
            list(training_pipeline._FixedIndexShardSampler(indices, num_replicas=5, rank=rank))
            for rank in range(5)
        ]

        self.assertEqual(sum(len(shard) for shard in shards), len(indices))
        self.assertEqual(sorted(index for shard in shards for index in shard), indices)
        self.assertEqual(len(set(index for shard in shards for index in shard)), len(indices))
        self.assertEqual(shards[0], [0, 1, 2])
        self.assertEqual(shards[-1], [9, 10])

    def test_fixed_index_shard_sampler_handles_more_ranks_than_indices(self) -> None:
        indices = [10, 20, 30]

        shards = [
            list(training_pipeline._FixedIndexShardSampler(indices, num_replicas=5, rank=rank))
            for rank in range(5)
        ]

        self.assertEqual(shards[0], [10])
        self.assertEqual(shards[1], [20])
        self.assertEqual(shards[2], [30])
        self.assertEqual(shards[3], [])
        self.assertEqual(shards[4], [])

    def test_to_plain_data_converts_timestamps_numpy_and_tensors(self) -> None:
        plain = training_pipeline._to_plain_data(
            {
                "start_time": pd.Timestamp("2024-01-01 00:00:00"),
                "scalar": np.float32(2.5),
                "vector": np.array([1, 2, 3], dtype=np.int64),
                "tensor": torch.tensor([[1.0, 2.0]]),
            }
        )

        self.assertEqual(plain["start_time"], "2024-01-01 00:00:00")
        self.assertEqual(plain["scalar"], 2.5)
        self.assertEqual(plain["vector"], [1, 2, 3])
        self.assertEqual(plain["tensor"], [[1.0, 2.0]])

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

    def test_resume_compatibility_allows_epoch_checkpoint_batching_changes(self) -> None:
        checkpoint = {
            "epoch": 3,
            "optimizer_step": 12,
            "global_batch_step": 96,
            "train_config": {
                "batch_size": 8,
                "gradient_accumulation_steps": 32,
            },
        }
        resume_state = training_pipeline._build_resume_state(
            checkpoint,
            checkpoint_path=Path("runs/main/main_best.pt"),
            total_train_batches=32,
            accumulation_steps=320,
        )

        warnings = training_pipeline._validate_resume_compatibility(
            checkpoint,
            checkpoint_path=Path("runs/main/main_best.pt"),
            section_name="train_main",
            current_batch_size=2,
            current_accumulation_steps=320,
            resume_state=resume_state,
        )

        self.assertEqual(len(warnings), 2)
        self.assertTrue(any("batch_size" in message for message in warnings))
        self.assertTrue(any("gradient_accumulation_steps" in message for message in warnings))

    def test_resume_compatibility_rejects_mid_epoch_batching_changes(self) -> None:
        checkpoint = {
            "epoch": 3,
            "batch_index_within_epoch": 20,
            "optimizer_step": 14,
            "train_config": {
                "batch_size": 8,
                "gradient_accumulation_steps": 32,
            },
        }
        resume_state = training_pipeline._build_resume_state(
            checkpoint,
            checkpoint_path=Path("runs/main/main_last_step_00000020.pt"),
            total_train_batches=50,
            accumulation_steps=320,
        )

        with self.assertRaisesRegex(ValueError, "mid-epoch checkpoint"):
            training_pipeline._validate_resume_compatibility(
                checkpoint,
                checkpoint_path=Path("runs/main/main_last_step_00000020.pt"),
                section_name="train_main",
                current_batch_size=2,
                current_accumulation_steps=320,
                resume_state=resume_state,
            )

    def test_apply_optimizer_hyperparameter_overrides(self) -> None:
        parameter = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=1.0e-4, weight_decay=0.05)
        checkpoint_optimizer = torch.optim.AdamW([parameter], lr=2.0e-4, weight_decay=0.1)
        optimizer.load_state_dict(checkpoint_optimizer.state_dict())

        training_pipeline._apply_optimizer_hyperparameter_overrides(
            optimizer,
            learning_rate=3.0e-4,
            weight_decay=0.2,
        )

        self.assertEqual(optimizer.param_groups[0]["lr"], 3.0e-4)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.2)

    def test_encode_patch_grid_features_for_intrinsic_keeps_encoder_graph_when_not_detached(self) -> None:
        encoder_config = FuXiLowerResConfig(
            input_size=(33, 64),
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
        encoder = FuXiLowerResEncoder(encoder_config)
        batch = {
            "x": torch.randn(2, 2, 8, 33, 64),
            "temb": torch.randn(2, 12),
            "static_features": torch.randn(2, 2, 33, 64),
        }

        patch_grid_features = training_pipeline._encode_patch_grid_features_for_intrinsic(
            encoder,
            batch,
            detach_features=False,
            clear_encoder_grads=True,
        )
        loss = patch_grid_features.square().mean()
        loss.backward()

        self.assertTrue(patch_grid_features.requires_grad)
        self.assertTrue(
            any(parameter.grad is not None for parameter in encoder.parameters())
        )

    def test_encode_features_for_bottleneck_compressor_uses_second_block_features(self) -> None:
        encoder_config = FuXiLowerResConfig(
            input_size=(33, 64),
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
        encoder = FuXiLowerResEncoder(encoder_config)
        batch = {
            "x": torch.randn(2, 2, 8, 33, 64),
            "temb": torch.randn(2, 12),
            "static_features": torch.randn(2, 2, 33, 64),
        }

        features = training_pipeline._encode_features_for_bottleneck_compressor(
            encoder,
            batch,
            feature_source="second_block_features",
            detach_features=True,
            clear_encoder_grads=True,
        )

        self.assertEqual(features.shape, (2, 16, 4, 8))
        self.assertFalse(features.requires_grad)

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
            input_size=(33, 64),
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
            spatial_size=encoder_config.patch_grid,
            d_intrinsic=3,
            depths=(1, 1),
            num_heads=4,
            num_groups=8,
            mlp_hidden_dim=32,
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

        self.assertEqual(report["patch_grid_features"]["shape"], [2, 16, 8, 16])
        self.assertEqual(report["intrinsic_output"]["z_intrinsic"]["shape"], [2, 3])
        self.assertEqual(report["intrinsic_output"]["patch_grid_features_recon"]["shape"], [2, 16, 8, 16])

    def test_bottleneck_compressor_smoke_test_runs_on_tiny_models(self) -> None:
        encoder_config = FuXiLowerResConfig(
            input_size=(33, 64),
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
        compressor_config = FuXiBottleneckCompressorConfig(
            input_channels=16,
            spatial_size=encoder_config.latent_grid,
            model_dim=16,
            bottleneck_channels=1,
            num_heads=4,
            encoder_depth=1,
            decoder_depth=1,
            mlp_hidden_dim=32,
            positional_embedding="learned_2d",
            feature_source="second_block_features",
            device="cpu",
            dtype=torch.float32,
        )
        encoder = FuXiLowerResEncoder(encoder_config)
        compressor_model = FuXiBottleneckCompressor(compressor_config)

        report = run_bottleneck_compressor_smoke_test(
            encoder,
            compressor_model,
            batch_size=2,
            print_outputs=False,
        )

        self.assertEqual(report["feature_source"], "second_block_features")
        self.assertEqual(report["feature_grid"]["shape"], [2, 16, 4, 8])
        self.assertEqual(report["compressor_output"]["z_bottleneck"]["shape"], [2, 1, 4, 8])
        self.assertEqual(report["compressor_output"]["feature_grid_recon"]["shape"], [2, 16, 4, 8])

    def test_intrinsic_forecast_chain_runs_on_tiny_models(self) -> None:
        encoder_config = FuXiLowerResConfig(
            input_size=(33, 64),
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
            spatial_size=encoder_config.patch_grid,
            d_intrinsic=3,
            depths=(1, 1),
            num_heads=4,
            num_groups=8,
            mlp_hidden_dim=32,
            apply_tanh=True,
            device="cpu",
            dtype=torch.float32,
        )
        main_model = FuXiLowerRes(encoder_config)
        intrinsic_model = FuXiIntrinsic(intrinsic_config)
        chain_model = training_pipeline._IntrinsicForecastChain(main_model, intrinsic_model)

        x, temb, static_features = training_pipeline._make_main_random_inputs(
            encoder_config,
            batch_size=2,
            device=torch.device("cpu"),
        )
        encoded = main_model.encoder(
            x,
            temb,
            static_features=static_features,
            return_patch_grid_features=True,
        )
        outputs = chain_model(x, temb, static_features=static_features)

        self.assertEqual(outputs["forecast"].shape, (2, 2, 8, 33, 64))
        self.assertEqual(outputs["z_intrinsic"].shape, (2, 3))
        self.assertEqual(outputs["patch_grid_features_recon"].shape, (2, 16, 8, 16))
        self.assertEqual(
            outputs["second_block_features"].shape,
            encoded.second_block_features.shape,
        )

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
