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


class IntrinsicConvStage(nn.Module):
    """Stack of unconditioned convolutional residual blocks at a fixed spatial grid."""

    def __init__(
        self,
        *,
        channels: int,
        depth: int,
        num_groups: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ResBlock(
                    in_channels=channels,
                    out_channels=channels,
                    num_groups=num_groups,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(int(depth))
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


@dataclass(frozen=True)
class FuXiIntrinsicConfig:
    """All-convolution intrinsic autoencoder over the main encoder bottleneck feature map."""

    input_channels: int | None = None
    feature_channels: int = 128
    spatial_size: tuple[int, int] = (23, 45)
    d_intrinsic: int = 16
    depths: tuple[int, ...] = (8, 8, 8)
    num_heads: int = 16
    num_groups: int = 32
    mlp_hidden_dim: int = 2048
    apply_tanh: bool = True
    config_path: Path = DEFAULT_MODEL_CONFIG_PATH
    device: str | torch.device | None = "meta"
    dtype: torch.dtype | None = torch.float16

    def __post_init__(self) -> None:
        if self.input_channels is not None and int(self.input_channels) <= 0:
            raise ValueError(f"input_channels must be positive when set, got {self.input_channels}")
        if self.feature_channels <= 0:
            raise ValueError(f"feature_channels must be positive, got {self.feature_channels}")
        if len(self.spatial_size) != 2:
            raise ValueError(f"Expected spatial_size with 2 dims, got {self.spatial_size}")
        if any(int(value) <= 0 for value in self.spatial_size):
            raise ValueError(f"spatial_size must be positive, got {self.spatial_size}")
        if self.d_intrinsic <= 0:
            raise ValueError(f"d_intrinsic must be positive, got {self.d_intrinsic}")
        if len(self.depths) not in {2, 3}:
            raise ValueError(
                "Expected 2 or 3 intrinsic convolution stage depths, "
                f"got {self.depths}"
            )
        if any(int(value) <= 0 for value in self.depths):
            raise ValueError(f"Intrinsic stage depths must be positive, got {self.depths}")
        if self.num_groups <= 0:
            raise ValueError(f"num_groups must be positive, got {self.num_groups}")

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
    ) -> "FuXiIntrinsicConfig":
        resolved_config_path, intrinsic_data = load_config_section("intrinsic_model", config_path)
        forecast_config = FuXiLowerResConfig.from_yaml(resolved_config_path)

        default_depths = tuple(int(value) for value in forecast_config.depths[:3])
        if not default_depths:
            default_depths = (8, 8, 8)
        if len(default_depths) < 3:
            default_depths = default_depths + (default_depths[-1],) * (3 - len(default_depths))

        return cls(
            input_channels=(
                None
                if intrinsic_data.get("input_channels") in {None, ""}
                else int(intrinsic_data.get("input_channels"))
            ),
            feature_channels=int(
                intrinsic_data.get(
                    "feature_channels",
                    intrinsic_data.get("d_high", forecast_config.embed_dim),
                )
            ),
            spatial_size=_to_int_tuple(
                intrinsic_data.get("spatial_size", list(forecast_config.latent_grid))
            ),
            d_intrinsic=int(intrinsic_data["d_intrinsic"]),
            depths=_to_int_tuple(
                intrinsic_data.get(
                    "resblocks_per_stage",
                    intrinsic_data.get("depths", list(default_depths)),
                )
            ),
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
    def stage_depths(self) -> tuple[int, int, int]:
        if len(self.depths) == 3:
            return tuple(int(value) for value in self.depths)
        first_depth, second_depth = (int(value) for value in self.depths)
        return (first_depth, second_depth, second_depth)

    @property
    def resblocks_per_stage(self) -> tuple[int, int, int]:
        return self.stage_depths

    @property
    def resolved_input_channels(self) -> int:
        return self.feature_channels if self.input_channels is None else int(self.input_channels)

    @property
    def first_downsampled_size(self) -> tuple[int, int]:
        return _stride2_same_shape(self.spatial_size)

    @property
    def second_downsampled_size(self) -> tuple[int, int]:
        return _stride2_same_shape(self.first_downsampled_size)

    @property
    def bottleneck_spatial_size(self) -> tuple[int, int]:
        return _stride2_same_shape(self.second_downsampled_size)

    @property
    def decoder_stage3_output_padding(self) -> tuple[int, int]:
        return _conv_transpose_output_padding(
            self.bottleneck_spatial_size,
            self.second_downsampled_size,
        )

    @property
    def decoder_stage2_output_padding(self) -> tuple[int, int]:
        return _conv_transpose_output_padding(
            self.second_downsampled_size,
            self.first_downsampled_size,
        )

    @property
    def decoder_stage1_output_padding(self) -> tuple[int, int]:
        return _conv_transpose_output_padding(
            self.first_downsampled_size,
            self.spatial_size,
        )

    @property
    def bottleneck_kernel_size(self) -> tuple[int, int]:
        return self.bottleneck_spatial_size


class FuXiIntrinsic(nn.Module):
    """Three-level convolutional intrinsic autoencoder over main encoder bottleneck features."""

    def __init__(self, config: FuXiIntrinsicConfig | None = None) -> None:
        super().__init__()
        self.config = config or FuXiIntrinsicConfig.from_yaml()
        dd = {"device": self.config.device, "dtype": self.config.dtype}
        stage_depths = self.config.stage_depths
        input_channels = self.config.resolved_input_channels

        if input_channels == self.config.feature_channels:
            self.input_proj: nn.Module = nn.Identity()
        else:
            self.input_proj = nn.Conv2d(
                input_channels,
                self.config.feature_channels,
                kernel_size=1,
                **dd,
            )

        self.input_resblock = ResBlock(
            in_channels=self.config.feature_channels,
            out_channels=self.config.feature_channels,
            num_groups=self.config.num_groups,
            **dd,
        )

        self.downsample1 = nn.Conv2d(
            self.config.feature_channels,
            self.config.feature_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            **dd,
        )
        self.encoder_stage1 = IntrinsicConvStage(
            channels=self.config.feature_channels,
            depth=stage_depths[0],
            num_groups=self.config.num_groups,
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
        self.encoder_stage2 = IntrinsicConvStage(
            channels=self.config.feature_channels,
            depth=stage_depths[1],
            num_groups=self.config.num_groups,
            **dd,
        )

        self.downsample3 = nn.Conv2d(
            self.config.feature_channels,
            self.config.feature_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            **dd,
        )
        self.encoder_stage3 = IntrinsicConvStage(
            channels=self.config.feature_channels,
            depth=stage_depths[2],
            num_groups=self.config.num_groups,
            **dd,
        )

        # Collapse the full bottleneck map to a learned 1x1 intrinsic code without flattening.
        self.to_intrinsic = nn.Conv2d(
            self.config.feature_channels,
            self.config.d_intrinsic,
            kernel_size=self.config.bottleneck_kernel_size,
            stride=1,
            padding=0,
            **dd,
        )
        self.from_intrinsic = nn.ConvTranspose2d(
            self.config.d_intrinsic,
            self.config.feature_channels,
            kernel_size=self.config.bottleneck_kernel_size,
            stride=1,
            padding=0,
            **dd,
        )

        self.decoder_stage3 = IntrinsicConvStage(
            channels=self.config.feature_channels,
            depth=stage_depths[2],
            num_groups=self.config.num_groups,
            **dd,
        )
        self.upsample3 = nn.ConvTranspose2d(
            self.config.feature_channels,
            self.config.feature_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=self.config.decoder_stage3_output_padding,
            **dd,
        )

        self.decoder_stage2 = IntrinsicConvStage(
            channels=self.config.feature_channels,
            depth=stage_depths[1],
            num_groups=self.config.num_groups,
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

        self.decoder_stage1 = IntrinsicConvStage(
            channels=self.config.feature_channels,
            depth=stage_depths[0],
            num_groups=self.config.num_groups,
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

        self.output_resblock = ResBlock(
            in_channels=self.config.feature_channels,
            out_channels=self.config.feature_channels,
            num_groups=self.config.num_groups,
            **dd,
        )
        self.output_proj = nn.Conv2d(
            self.config.feature_channels,
            input_channels,
            kernel_size=1,
            **dd,
        )

    def _model_dtype(self) -> torch.dtype:
        return self.to_intrinsic.weight.dtype

    def _validate_feature_grid(self, feature_grid: Tensor) -> None:
        expected_shape = (self.config.resolved_input_channels, *self.config.spatial_size)
        if feature_grid.ndim != 4 or tuple(feature_grid.shape[1:]) != expected_shape:
            raise ValueError(
                "Expected second-block bottleneck features shaped "
                f"[B, {self.config.resolved_input_channels}, {self.config.spatial_size[0]}, {self.config.spatial_size[1]}], "
                f"got {tuple(feature_grid.shape)}"
            )

    def encode(self, feature_grid: Tensor) -> Tensor:
        self._validate_feature_grid(feature_grid)
        h = feature_grid.to(dtype=self._model_dtype())
        h = self.input_proj(h)
        h = self.input_resblock(h)

        h = self.downsample1(h)
        h = self.encoder_stage1(h)

        h = self.downsample2(h)
        h = self.encoder_stage2(h)

        h = self.downsample3(h)
        h = self.encoder_stage3(h)

        batch_size = h.shape[0]
        z_intrinsic = self.to_intrinsic(h).reshape(batch_size, self.config.d_intrinsic)
        if self.config.apply_tanh:
            z_intrinsic = torch.tanh(z_intrinsic)
        return z_intrinsic

    def decode(self, z_intrinsic: Tensor) -> Tensor:
        if z_intrinsic.ndim != 2 or z_intrinsic.shape[1] != self.config.d_intrinsic:
            raise ValueError(
                f"Expected z_intrinsic shaped [B, {self.config.d_intrinsic}], got {tuple(z_intrinsic.shape)}"
            )

        batch_size = z_intrinsic.shape[0]
        h = self.from_intrinsic(
            z_intrinsic.to(dtype=self._model_dtype()).reshape(batch_size, self.config.d_intrinsic, 1, 1)
        )
        h = self.decoder_stage3(h)

        h = self.upsample3(h)
        h = self.decoder_stage2(h)

        h = self.upsample2(h)
        h = self.decoder_stage1(h)

        h = self.upsample1(h)
        h = self.output_resblock(h)
        return self.output_proj(h)

    def forward(self, feature_grid: Tensor) -> dict[str, Tensor]:
        z_intrinsic = self.encode(feature_grid)
        second_block_features_recon = self.decode(z_intrinsic)
        return {
            "z_intrinsic": z_intrinsic,
            "second_block_features_recon": second_block_features_recon,
            "patch_grid_features_recon": second_block_features_recon,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config.config_path),
            "input_channels": self.config.resolved_input_channels,
            "output_channels": self.config.resolved_input_channels,
            "feature_channels": self.config.feature_channels,
            "spatial_size": list(self.config.spatial_size),
            "first_downsampled_size": list(self.config.first_downsampled_size),
            "second_downsampled_size": list(self.config.second_downsampled_size),
            "bottleneck_spatial_size": list(self.config.bottleneck_spatial_size),
            "d_intrinsic": self.config.d_intrinsic,
            "depths": list(self.config.stage_depths),
            "resblocks_per_stage": list(self.config.resblocks_per_stage),
            "num_heads": self.config.num_heads,
            "num_groups": self.config.num_groups,
            "mlp_hidden_dim": self.config.mlp_hidden_dim,
            "apply_tanh": self.config.apply_tanh,
            "architecture": "conv_autoencoder",
            "transformer_type": "none",
            "block_type": "resblock_conv",
            "uses_attention": False,
            "uses_windowed_attention": False,
            "uses_positional_embeddings": False,
            "downsample_count": 3,
            "bottleneck_projection": "global_conv_1x1_code",
            "bottleneck_kernel_size": list(self.config.bottleneck_kernel_size),
            "input_feature_name": "second_block_features",
            "reconstruction_name": "second_block_features_recon",
            "parameter_device": str(self.to_intrinsic.weight.device),
            "parameter_dtype": str(self.to_intrinsic.weight.dtype),
        }


__all__ = ["FuXiIntrinsic", "FuXiIntrinsicConfig"]
