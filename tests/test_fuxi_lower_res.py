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
        self.assertEqual(summary["depths"], [12, 12, 12, 12])
        self.assertEqual(summary["d_high"], 128)
        self.assertEqual(summary["z_high_shape"], [128, 23, 45])
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
            d_high=6,
            device="cpu",
            dtype=torch.float32,
        )
        model = FuXiLowerRes(config)
        x = torch.randn(2, 2, 8, 17, 32)
        temb = torch.randn(2, 12)
        static_features = torch.randn(2, 2, 2, 17, 32)

        outputs = model(x, temb, static_features=static_features)

        self.assertEqual(set(outputs.keys()), {"forecast", "z_high"})
        self.assertEqual(outputs["forecast"].shape, (2, 2, 8, 17, 32))
        self.assertEqual(outputs["z_high"].shape, (2, 6, 2, 4))

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
            d_high=6,
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
        self.assertEqual(encoded.z_high.shape, (2, 6, 2, 4))
        self.assertEqual(forecast.shape, (2, 2, 8, 17, 32))
        self.assertEqual(next_forecast.shape, (2, 8, 17, 32))
        self.assertEqual(future_forecast.shape, (2, 2, 8, 17, 32))


class TestFuXiIntrinsic(unittest.TestCase):
    def test_default_recipe_uses_meta_to_avoid_oom(self) -> None:
        model = FuXiIntrinsic()
        summary = model.summary()

        self.assertEqual(summary["d_high"], 128)
        self.assertEqual(summary["spatial_size"], [23, 45])
        self.assertEqual(summary["d_intrinsic"], 16)
        self.assertEqual(summary["parameter_device"], "meta")

    def test_tiny_config_runs_forward(self) -> None:
        config = FuXiIntrinsicConfig(
            d_high=6,
            spatial_size=(2, 4),
            d_intrinsic=3,
            hidden_dims=(24, 12),
            apply_tanh=True,
            device="cpu",
            dtype=torch.float32,
        )
        model = FuXiIntrinsic(config)
        z_high = torch.randn(2, 6, 2, 4)

        outputs = model(z_high)

        self.assertEqual(set(outputs.keys()), {"z_intrinsic", "z_high_recon"})
        self.assertEqual(outputs["z_intrinsic"].shape, (2, 3))
        self.assertEqual(outputs["z_high_recon"].shape, (2, 6, 2, 4))
        self.assertTrue(torch.all(outputs["z_intrinsic"] <= 1.0))
        self.assertTrue(torch.all(outputs["z_intrinsic"] >= -1.0))


if __name__ == "__main__":
    unittest.main()
