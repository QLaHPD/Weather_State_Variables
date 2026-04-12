import unittest

import torch

from weather_state_variables.models import (
    FuXiEncoderOutput,
    FuXiIntrinsic,
    FuXiIntrinsicConfig,
    FuXiLowerRes,
    FuXiLowerResConfig,
    FuXiLowerResDecoder,
    FuXiLowerResEncoder,
)


class TestFuXiLowerRes(unittest.TestCase):
    def test_default_model_exposes_encoder_and_decoder(self) -> None:
        model = FuXiLowerRes()

        self.assertIsInstance(model.encoder, FuXiLowerResEncoder)
        self.assertIsInstance(model.decoder, FuXiLowerResDecoder)

    def test_default_recipe_uses_meta_to_avoid_oom(self) -> None:
        model = FuXiLowerRes()
        summary = model.summary()

        self.assertEqual(summary["input_size"], [181, 360])
        self.assertEqual(summary["resized_input_size"], [180, 360])
        self.assertEqual(summary["window_size"], 9)
        self.assertEqual(summary["depths"], [4, 4, 4, 4])
        self.assertEqual(summary["shared_feature_name"], "second_block_features")
        self.assertEqual(summary["second_block_feature_shape"], [768, 23, 45])
        self.assertEqual(summary["forecast_steps"], 2)
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

        encoded = model.encode(x, temb, static_features=static_features)
        forecast = model.decode(encoded)
        next_forecast = model.predict_next(x, temb, static_features=static_features)
        future_forecast = model.predict_future(x, temb, static_features=static_features)

        self.assertIsInstance(encoded, FuXiEncoderOutput)
        self.assertEqual(encoded.second_block_features.shape, (2, 16, 2, 4))
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

        encoded = model.encode(x, temb, static_features=static_features)
        outputs = model(x, temb, static_features=static_features)

        self.assertEqual(encoded.second_block_features.shape, (2, 16, 2, 4))
        self.assertEqual(set(outputs.keys()), {"forecast", "second_block_features"})
        self.assertEqual(outputs["second_block_features"].shape, (2, 16, 2, 4))
        self.assertEqual(outputs["forecast"].shape, (2, 2, 8, 17, 32))


class TestFuXiIntrinsic(unittest.TestCase):
    def test_default_recipe_uses_meta_to_avoid_oom(self) -> None:
        model = FuXiIntrinsic()
        summary = model.summary()

        self.assertEqual(summary["feature_channels"], 768)
        self.assertEqual(summary["spatial_size"], [23, 45])
        self.assertEqual(summary["d_intrinsic"], 16)
        self.assertEqual(summary["parameter_device"], "meta")

    def test_tiny_config_runs_forward(self) -> None:
        config = FuXiIntrinsicConfig(
            feature_channels=16,
            spatial_size=(2, 4),
            d_intrinsic=3,
            hidden_dims=(24, 12),
            apply_tanh=True,
            device="cpu",
            dtype=torch.float32,
        )
        model = FuXiIntrinsic(config)
        second_block_features = torch.randn(2, 16, 2, 4)

        outputs = model(second_block_features)

        self.assertEqual(set(outputs.keys()), {"z_intrinsic", "second_block_features_recon"})
        self.assertEqual(outputs["z_intrinsic"].shape, (2, 3))
        self.assertEqual(outputs["second_block_features_recon"].shape, (2, 16, 2, 4))
        self.assertTrue(torch.all(outputs["z_intrinsic"] <= 1.0))
        self.assertTrue(torch.all(outputs["z_intrinsic"] >= -1.0))


if __name__ == "__main__":
    unittest.main()
