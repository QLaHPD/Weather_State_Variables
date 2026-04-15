from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from ..config import DEFAULT_MODEL_CONFIG_PATH, load_config_section, resolve_torch_dtype
from .fuxi_lower_res import FuXiLowerResConfig, ResBlock


def _to_int_tuple(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def _stride2_same_size(size: int) -> int:
    return (int(size) + 1) // 2


def _stride2_same_shape(spatial_size: tuple[int, int]) -> tuple[int, int]:
    return tuple(_stride2_same_size(value) for value in spatial_size)


def _conv_transpose_output_padding(
    input_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[int, int]:
    padding: list[int] = []
    for input_dim, target_dim in zip(input_size, target_size, strict=True):
        baseline = 2 * int(input_dim) - 1
        output_padding = int(target_dim) - baseline
        if output_padding not in {0, 1}:
            raise ValueError(
                "Expected conv-transpose output_padding to be 0 or 1 when inverting "
                f"stride-2 same-padded convolutions, got {output_padding} for "
                f"input_dim={input_dim}, target_dim={target_dim}."
            )
        padding.append(output_padding)
    return tuple(padding)


class IntrinsicTransformerStage(nn.Module):
    """Full-sequence transformer encoder over a fixed spatial grid."""

    def __init__(
        self,
        *,
        dim: int,
        spatial_size: tuple[int, int],
        depth: int,
        num_heads: int,
        mlp_hidden_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        dd = {"device": device, "dtype": dtype}
        super().__init__()
        self.dim = int(dim)
        self.spatial_size = tuple(int(value) for value in spatial_size)
        self.seq_len = int(self.spatial_size[0] * self.spatial_size[1])

        self.position_embedding = nn.Parameter(torch.zeros(1, self.seq_len, self.dim, **dd))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=int(num_heads),
            dim_feedforward=int(mlp_hidden_dim),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            **dd,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(depth),
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(self.dim, **dd)

        if self.position_embedding.device.type != "meta":
            nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected transformer stage input shaped [B, C, H, W], got {tuple(x.shape)}"
            )
        batch_size, channels, height, width = x.shape
        if channels != self.dim or (height, width) != self.spatial_size:
            raise ValueError(
                "Expected transformer stage input shaped "
                f"[B, {self.dim}, {self.spatial_size[0]}, {self.spatial_size[1]}], "
                f"got {tuple(x.shape)}"
            )

        tokens = x.permute(0, 2, 3, 1).reshape(batch_size, self.seq_len, self.dim)
        tokens = tokens + self.position_embedding.to(dtype=tokens.dtype)
        tokens = self.encoder(tokens)
        tokens = self.output_norm(tokens)
        return tokens.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)


@dataclass(frozen=True)
class FuXiIntrinsicConfig:
    """Hierarchical intrinsic autoencoder over the encoder patch-grid feature map."""

    feature_channels: int = 128
    spatial_size: tuple[int, int] = (45, 90)
    d_intrinsic: int = 16
    depths: tuple[int, int] = (8, 8)
    num_heads: int = 16
    num_groups: int = 32
    mlp_hidden_dim: int = 2048
    apply_tanh: bool = True
    config_path: Path = DEFAULT_MODEL_CONFIG_PATH
    device: str | torch.device | None = "meta"
    dtype: torch.dtype | None = torch.float16

    def __post_init__(self) -> None:
        if self.feature_channels <= 0:
            raise ValueError(f"feature_channels must be positive, got {self.feature_channels}")
        if len(self.spatial_size) != 2:
            raise ValueError(f"Expected spatial_size with 2 dims, got {self.spatial_size}")
        if any(int(value) <= 0 for value in self.spatial_size):
            raise ValueError(f"spatial_size must be positive, got {self.spatial_size}")
        if self.d_intrinsic <= 0:
            raise ValueError(f"d_intrinsic must be positive, got {self.d_intrinsic}")
        if len(self.depths) != 2:
            raise ValueError(f"Expected 2 intrinsic transformer stages, got {self.depths}")
        if any(int(value) <= 0 for value in self.depths):
            raise ValueError(f"Intrinsic stage depths must be positive, got {self.depths}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.feature_channels % self.num_heads != 0:
            raise ValueError(
                "feature_channels must be divisible by num_heads, got "
                f"feature_channels={self.feature_channels}, num_heads={self.num_heads}"
            )
        if self.num_groups <= 0:
            raise ValueError(f"num_groups must be positive, got {self.num_groups}")
        if self.mlp_hidden_dim <= 0:
            raise ValueError(f"mlp_hidden_dim must be positive, got {self.mlp_hidden_dim}")

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
    ) -> "FuXiIntrinsicConfig":
        resolved_config_path, intrinsic_data = load_config_section("intrinsic_model", config_path)
        forecast_config = FuXiLowerResConfig.from_yaml(resolved_config_path)

        default_depths = tuple(int(value) for value in forecast_config.depths[:2])
        return cls(
            feature_channels=int(
                intrinsic_data.get(
                    "feature_channels",
                    intrinsic_data.get("d_high", forecast_config.embed_dim),
                )
            ),
            spatial_size=_to_int_tuple(
                intrinsic_data.get("spatial_size", list(forecast_config.patch_grid))
            ),
            d_intrinsic=int(intrinsic_data["d_intrinsic"]),
            depths=_to_int_tuple(intrinsic_data.get("depths", list(default_depths))),
            num_heads=int(intrinsic_data.get("num_heads", forecast_config.num_heads)),
            num_groups=int(intrinsic_data.get("num_groups", forecast_config.num_groups)),
            mlp_hidden_dim=int(
                intrinsic_data.get("mlp_hidden_dim", forecast_config.mlp_hidden_dim)
            ),
            apply_tanh=bool(intrinsic_data.get("apply_tanh", True)),
            config_path=resolved_config_path,
            device=intrinsic_data.get("device", forecast_config.device),
            dtype=resolve_torch_dtype(
                intrinsic_data.get("dtype", str(forecast_config.dtype).replace("torch.", ""))
            ),
        )

    @property
    def first_downsampled_size(self) -> tuple[int, int]:
        return _stride2_same_shape(self.spatial_size)

    @property
    def bottleneck_spatial_size(self) -> tuple[int, int]:
        return _stride2_same_shape(self.first_downsampled_size)

    @property
    def decoder_stage2_output_padding(self) -> tuple[int, int]:
        return _conv_transpose_output_padding(
            self.bottleneck_spatial_size,
            self.first_downsampled_size,
        )

    @property
    def decoder_stage1_output_padding(self) -> tuple[int, int]:
        return _conv_transpose_output_padding(
            self.first_downsampled_size,
            self.spatial_size,
        )

    @property
    def bottleneck_flat_dim(self) -> int:
        return (
            self.feature_channels
            * self.bottleneck_spatial_size[0]
            * self.bottleneck_spatial_size[1]
        )


class FuXiIntrinsic(nn.Module):
    """Hierarchical intrinsic autoencoder over encoder patch-grid features."""

    def __init__(self, config: FuXiIntrinsicConfig | None = None) -> None:
        super().__init__()
        self.config = config or FuXiIntrinsicConfig.from_yaml()
        dd = {"device": self.config.device, "dtype": self.config.dtype}

        self.downsample1 = nn.Conv2d(
            self.config.feature_channels,
            self.config.feature_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            **dd,
        )
        self.down_resblock1 = ResBlock(
            in_channels=self.config.feature_channels,
            out_channels=self.config.feature_channels,
            num_groups=self.config.num_groups,
            **dd,
        )
        self.encoder_stage1 = IntrinsicTransformerStage(
            dim=self.config.feature_channels,
            spatial_size=self.config.first_downsampled_size,
            depth=self.config.depths[0],
            num_heads=self.config.num_heads,
            mlp_hidden_dim=self.config.mlp_hidden_dim,
            **dd,
        )
        self.downsample2 = nn.Conv2d(
            self.config.feature_channels,
            self.config.feature_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            **dd,
        )
        self.down_resblock2 = ResBlock(
            in_channels=self.config.feature_channels,
            out_channels=self.config.feature_channels,
            num_groups=self.config.num_groups,
            **dd,
        )
        self.encoder_stage2 = IntrinsicTransformerStage(
            dim=self.config.feature_channels,
            spatial_size=self.config.bottleneck_spatial_size,
            depth=self.config.depths[1],
            num_heads=self.config.num_heads,
            mlp_hidden_dim=self.config.mlp_hidden_dim,
            **dd,
        )
        self.to_intrinsic = nn.Linear(self.config.bottleneck_flat_dim, self.config.d_intrinsic, **dd)
        self.from_intrinsic = nn.Linear(self.config.d_intrinsic, self.config.bottleneck_flat_dim, **dd)

        self.decoder_stage2 = IntrinsicTransformerStage(
            dim=self.config.feature_channels,
            spatial_size=self.config.bottleneck_spatial_size,
            depth=self.config.depths[1],
            num_heads=self.config.num_heads,
            mlp_hidden_dim=self.config.mlp_hidden_dim,
            **dd,
        )
        self.upsample2 = nn.ConvTranspose2d(
            self.config.feature_channels,
            self.config.feature_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=self.config.decoder_stage2_output_padding,
            **dd,
        )
        self.up_resblock2 = ResBlock(
            in_channels=self.config.feature_channels,
            out_channels=self.config.feature_channels,
            num_groups=self.config.num_groups,
            **dd,
        )
        self.decoder_stage1 = IntrinsicTransformerStage(
            dim=self.config.feature_channels,
            spatial_size=self.config.first_downsampled_size,
            depth=self.config.depths[0],
            num_heads=self.config.num_heads,
            mlp_hidden_dim=self.config.mlp_hidden_dim,
            **dd,
        )
        self.upsample1 = nn.ConvTranspose2d(
            self.config.feature_channels,
            self.config.feature_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=self.config.decoder_stage1_output_padding,
            **dd,
        )
        self.up_resblock1 = ResBlock(
            in_channels=self.config.feature_channels,
            out_channels=self.config.feature_channels,
            num_groups=self.config.num_groups,
            **dd,
        )
        self.output_proj = nn.Conv2d(
            self.config.feature_channels,
            self.config.feature_channels,
            kernel_size=1,
            **dd,
        )

    def _model_dtype(self) -> torch.dtype:
        return self.to_intrinsic.weight.dtype

    def _validate_feature_grid(self, feature_grid: Tensor) -> None:
        expected_shape = (self.config.feature_channels, *self.config.spatial_size)
        if feature_grid.ndim != 4 or tuple(feature_grid.shape[1:]) != expected_shape:
            raise ValueError(
                "Expected patch-grid features shaped "
                f"[B, {self.config.feature_channels}, {self.config.spatial_size[0]}, {self.config.spatial_size[1]}], "
                f"got {tuple(feature_grid.shape)}"
            )

    def encode(self, feature_grid: Tensor) -> Tensor:
        self._validate_feature_grid(feature_grid)
        h = feature_grid.to(dtype=self._model_dtype())
        h = self.downsample1(h)
        h = self.down_resblock1(h)
        h = self.encoder_stage1(h)

        h = self.downsample2(h)
        h = self.down_resblock2(h)
        h = self.encoder_stage2(h)

        batch_size = h.shape[0]
        flat = h.reshape(batch_size, self.config.bottleneck_flat_dim)
        z_intrinsic = self.to_intrinsic(flat)
        if self.config.apply_tanh:
            z_intrinsic = torch.tanh(z_intrinsic)
        return z_intrinsic

    def decode(self, z_intrinsic: Tensor) -> Tensor:
        if z_intrinsic.ndim != 2 or z_intrinsic.shape[1] != self.config.d_intrinsic:
            raise ValueError(
                f"Expected z_intrinsic shaped [B, {self.config.d_intrinsic}], got {tuple(z_intrinsic.shape)}"
            )

        batch_size = z_intrinsic.shape[0]
        h = self.from_intrinsic(z_intrinsic.to(dtype=self._model_dtype()))
        h = h.reshape(
            batch_size,
            self.config.feature_channels,
            *self.config.bottleneck_spatial_size,
        )
        h = self.decoder_stage2(h)

        h = self.upsample2(h)
        h = self.up_resblock2(h)
        h = self.decoder_stage1(h)

        h = self.upsample1(h)
        h = self.up_resblock1(h)
        return self.output_proj(h)

    def forward(self, feature_grid: Tensor) -> dict[str, Tensor]:
        z_intrinsic = self.encode(feature_grid)
        patch_grid_features_recon = self.decode(z_intrinsic)
        return {
            "z_intrinsic": z_intrinsic,
            "patch_grid_features_recon": patch_grid_features_recon,
            "second_block_features_recon": patch_grid_features_recon,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config.config_path),
            "feature_channels": self.config.feature_channels,
            "spatial_size": list(self.config.spatial_size),
            "first_downsampled_size": list(self.config.first_downsampled_size),
            "bottleneck_spatial_size": list(self.config.bottleneck_spatial_size),
            "d_intrinsic": self.config.d_intrinsic,
            "depths": list(self.config.depths),
            "num_heads": self.config.num_heads,
            "num_groups": self.config.num_groups,
            "mlp_hidden_dim": self.config.mlp_hidden_dim,
            "apply_tanh": self.config.apply_tanh,
            "transformer_type": "standard_encoder",
            "uses_windowed_attention": False,
            "bottleneck_projection": "flatten_linear",
            "bottleneck_flat_dim": self.config.bottleneck_flat_dim,
            "input_feature_name": "patch_grid_features",
            "reconstruction_name": "patch_grid_features_recon",
            "parameter_device": str(self.to_intrinsic.weight.device),
            "parameter_dtype": str(self.to_intrinsic.weight.dtype),
        }


__all__ = ["FuXiIntrinsic", "FuXiIntrinsicConfig"]
