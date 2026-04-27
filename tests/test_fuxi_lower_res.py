import unittest

import torch

from weather_state_variables.models import (
    FuXiEncoderOutput,
    FuXiBottleneckCompressor,
    FuXiBottleneckCompressorConfig,
    FuXiIntrinsic,
    FuXiIntrinsicConfig,
    FuXiLowerRes,
    FuXiLowerResConfig,
    FuXiLowerResDecoder,
    FuXiLowerResEncoder,
    LatentDynamicsConfig,
    NeuralLatentDynamics,
)


class TestFuXiLowerRes(unittest.TestCase):
    def test_default_model_exposes_encoder_and_decoder(self) -> None:
        model = FuXiLowerRes()

        self.assertIsInstance(model.encoder, FuXiLowerResEncoder)
        self.assertIsInstance(model.decoder, FuXiLowerResDecoder)

    def test_default_recipe_uses_meta_to_avoid_oom(self) -> None:
        model = FuXiLowerRes()
        summary = model.summary()
        config = model.config

        self.assertEqual(summary["input_size"], list(config.input_size))
        self.assertEqual(summary["resized_input_size"], list(config.resized_input_size))
        self.assertEqual(summary["window_size"], config.window_size)
        self.assertEqual(summary["depths"], list(config.depths))
        self.assertEqual(summary["shared_feature_name"], "second_block_features")
        self.assertFalse(summary["cross_boundary_skip"])
        self.assertFalse(summary["cross_boundary_time_conditioning"])
        self.assertEqual(summary["second_block_feature_shape"], [config.embed_dim, *config.latent_grid])
        self.assertEqual(summary["forecast_steps"], config.forecast_steps)
        self.assertEqual(summary["parameter_device"], "meta")

    def test_tiny_config_runs_forward(self) -> None:
        config = FuXiLowerResConfig(
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
        model = FuXiLowerRes(config)
        x = torch.randn(2, 2, 8, 17, 32)
        temb = torch.randn(2, 12)
        static_features = torch.randn(2, 2, 2, 17, 32)

        outputs = model(x, temb, static_features=static_features)

        self.assertEqual(set(outputs.keys()), {"forecast", "second_block_features"})
        self.assertEqual(outputs["forecast"].shape, (2, 2, 8, 17, 32))
        self.assertEqual(outputs["second_block_features"].shape, (2, 16, 2, 4))

    def test_tiny_config_encode_decode_runs_forward(self) -> None:
        config = FuXiLowerResConfig(
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
        model = FuXiLowerRes(config)
        x = torch.randn(2, 2, 8, 17, 32)
        temb = torch.randn(2, 12)
        static_features = torch.randn(2, 2, 2, 17, 32)

        encoded = model.encode(
            x,
            temb,
            static_features=static_features,
            return_patch_grid_features=True,
        )
        forecast = model.decode(encoded)
        next_forecast = model.predict_next(x, temb, static_features=static_features)
        future_forecast = model.predict_future(x, temb, static_features=static_features)

        self.assertIsInstance(encoded, FuXiEncoderOutput)
        self.assertEqual(encoded.patch_grid_features.shape, (2, 16, 4, 8))
        self.assertEqual(encoded.second_block_features.shape, (2, 16, 2, 4))
        self.assertFalse(hasattr(encoded, "skip"))
        self.assertFalse(hasattr(encoded, "temb_emb"))
        self.assertEqual(forecast.shape, (2, 2, 8, 17, 32))
        self.assertEqual(next_forecast.shape, (2, 8, 17, 32))
        self.assertEqual(future_forecast.shape, (2, 2, 8, 17, 32))

    def test_tiny_config_returns_second_block_features(self) -> None:
        config = FuXiLowerResConfig(
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
        model = FuXiLowerRes(config)
        x = torch.randn(2, 2, 8, 17, 32)
        temb = torch.randn(2, 12)
        static_features = torch.randn(2, 2, 2, 17, 32)

        encoded = model.encode(
            x,
            temb,
            static_features=static_features,
            return_patch_grid_features=True,
        )
        outputs = model(x, temb, static_features=static_features)

        self.assertEqual(encoded.patch_grid_features.shape, (2, 16, 4, 8))
        self.assertEqual(encoded.second_block_features.shape, (2, 16, 2, 4))
        self.assertEqual(set(outputs.keys()), {"forecast", "second_block_features"})
        self.assertEqual(outputs["second_block_features"].shape, (2, 16, 2, 4))
        self.assertEqual(outputs["forecast"].shape, (2, 2, 8, 17, 32))

    def test_decoder_summary_reports_true_bottleneck(self) -> None:
        config = FuXiLowerResConfig(
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
        model = FuXiLowerRes(config)
        summary = model.decoder.summary()

        self.assertFalse(summary["uses_encoder_skip"])
        self.assertFalse(summary["uses_encoder_time_embedding"])
        self.assertEqual(model.decoder.up_resblock.conv1.in_channels, config.embed_dim)


class TestFuXiIntrinsic(unittest.TestCase):
    def test_default_recipe_uses_meta_to_avoid_oom(self) -> None:
        model = FuXiIntrinsic()
        summary = model.summary()
        config = model.config

        self.assertEqual(summary["input_channels"], config.resolved_input_channels)
        self.assertEqual(summary["output_channels"], config.resolved_input_channels)
        self.assertEqual(summary["feature_channels"], config.feature_channels)
        self.assertEqual(summary["spatial_size"], list(config.spatial_size))
        self.assertEqual(summary["first_downsampled_size"], list(config.first_downsampled_size))
        self.assertEqual(summary["second_downsampled_size"], list(config.second_downsampled_size))
        self.assertEqual(summary["bottleneck_spatial_size"], list(config.bottleneck_spatial_size))
        self.assertEqual(summary["d_intrinsic"], config.d_intrinsic)
        self.assertEqual(summary["depths"], list(config.stage_depths))
        self.assertEqual(summary["resblocks_per_stage"], list(config.resblocks_per_stage))
        self.assertEqual(summary["architecture"], "conv_autoencoder")
        self.assertEqual(summary["transformer_type"], "none")
        self.assertFalse(summary["uses_attention"])
        self.assertFalse(summary["uses_windowed_attention"])
        self.assertFalse(summary["uses_positional_embeddings"])
        self.assertEqual(summary["downsample_count"], 3)
        self.assertEqual(summary["input_feature_name"], "second_block_features")
        self.assertEqual(summary["reconstruction_name"], "second_block_features_recon")
        self.assertEqual(summary["parameter_device"], "meta")

    def test_tiny_config_runs_forward(self) -> None:
        config = FuXiIntrinsicConfig(
            feature_channels=16,
            spatial_size=(8, 16),
            d_intrinsic=3,
            depths=(1, 1),
            num_heads=4,
            num_groups=8,
            mlp_hidden_dim=32,
            apply_tanh=True,
            device="cpu",
            dtype=torch.float32,
        )
        model = FuXiIntrinsic(config)
        second_block_features = torch.randn(2, 16, 8, 16)

        outputs = model(second_block_features)

        self.assertEqual(
            set(outputs.keys()),
            {"z_intrinsic", "patch_grid_features_recon", "second_block_features_recon"},
        )
        self.assertEqual(outputs["z_intrinsic"].shape, (2, 3))
        self.assertEqual(outputs["second_block_features_recon"].shape, (2, 16, 8, 16))
        self.assertEqual(outputs["patch_grid_features_recon"].shape, (2, 16, 8, 16))
        self.assertTrue(torch.all(outputs["z_intrinsic"] <= 1.0))
        self.assertTrue(torch.all(outputs["z_intrinsic"] >= -1.0))

    def test_tiny_config_supports_distinct_input_and_hidden_channels(self) -> None:
        config = FuXiIntrinsicConfig(
            input_channels=24,
            feature_channels=16,
            spatial_size=(8, 16),
            d_intrinsic=3,
            depths=(1, 1, 1),
            num_groups=8,
            apply_tanh=False,
            device="cpu",
            dtype=torch.float32,
        )
        model = FuXiIntrinsic(config)
        second_block_features = torch.randn(2, 24, 8, 16)

        outputs = model(second_block_features)

        self.assertEqual(outputs["z_intrinsic"].shape, (2, 3))
        self.assertEqual(outputs["second_block_features_recon"].shape, (2, 24, 8, 16))


class TestFuXiBottleneckCompressor(unittest.TestCase):
    def test_default_recipe_uses_meta_to_avoid_oom(self) -> None:
        model = FuXiBottleneckCompressor()
        summary = model.summary()
        config = model.config

        self.assertEqual(summary["input_channels"], config.resolved_input_channels)
        self.assertEqual(summary["output_channels"], config.resolved_input_channels)
        self.assertEqual(summary["spatial_size"], list(config.spatial_size))
        self.assertEqual(summary["bottleneck_channels"], config.bottleneck_channels)
        self.assertEqual(summary["bottleneck_shape"], list(config.bottleneck_shape))
        self.assertEqual(summary["architecture"], "transformer_grid_bottleneck_autoencoder")
        self.assertEqual(summary["transformer_type"], "torch.nn.TransformerEncoder")
        self.assertTrue(summary["uses_positional_embeddings"])
        self.assertFalse(summary["uses_skip_connections"])
        self.assertEqual(summary["feature_source"], "second_block_features")
        self.assertEqual(summary["parameter_device"], "meta")

    def test_tiny_config_runs_forward(self) -> None:
        config = FuXiBottleneckCompressorConfig(
            input_channels=16,
            spatial_size=(2, 4),
            model_dim=16,
            bottleneck_channels=1,
            num_heads=4,
            encoder_depth=1,
            decoder_depth=1,
            mlp_hidden_dim=32,
            dropout=0.0,
            positional_embedding="learned_2d",
            feature_source="second_block_features",
            device="cpu",
            dtype=torch.float32,
        )
        model = FuXiBottleneckCompressor(config)
        second_block_features = torch.randn(2, 16, 2, 4)

        outputs = model(second_block_features)

        self.assertEqual(
            set(outputs.keys()),
            {
                "z_bottleneck",
                "bottleneck_features",
                "second_block_features_recon",
                "feature_grid_recon",
            },
        )
        self.assertEqual(outputs["z_bottleneck"].shape, (2, 1, 2, 4))
        self.assertEqual(outputs["bottleneck_features"].shape, (2, 1, 2, 4))
        self.assertEqual(outputs["second_block_features_recon"].shape, (2, 16, 2, 4))
        self.assertEqual(outputs["feature_grid_recon"].shape, (2, 16, 2, 4))


class TestLatentDynamics(unittest.TestCase):
    def test_default_recipe_uses_meta_to_avoid_oom(self) -> None:
        model = NeuralLatentDynamics()
        summary = model.summary()

        self.assertEqual(summary["activation"], "relu")
        self.assertEqual(summary["layer_count"], len(summary["hidden_dims"]) + 1)
        self.assertEqual(summary["parameter_device"], "meta")

    def test_tiny_config_runs_forward(self) -> None:
        config = LatentDynamicsConfig(
            latent_dim=3,
            hidden_dims=(8, 16, 8),
            activation="relu",
            device="cpu",
            dtype=torch.float32,
        )
        model = NeuralLatentDynamics(config)
        z = torch.randn(4, 3)

        derivative = model(z)

        self.assertEqual(derivative.shape, (4, 3))
        self.assertEqual(derivative.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
