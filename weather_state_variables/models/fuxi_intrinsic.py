from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from ..config import DEFAULT_MODEL_CONFIG_PATH, load_config_section, resolve_torch_dtype
from .fuxi_lower_res import FuXiLowerResConfig


def _to_int_tuple(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


@dataclass(frozen=True)
class FuXiIntrinsicConfig:
    """Config-driven intrinsic latent branch over second-block shared features."""

    feature_channels: int = 128
    spatial_size: tuple[int, int] = (23, 45)
    d_intrinsic: int = 16
    hidden_dims: tuple[int, int] = (2048, 512)
    apply_tanh: bool = True
    config_path: Path = DEFAULT_MODEL_CONFIG_PATH
    device: str | torch.device | None = "meta"
    dtype: torch.dtype | None = torch.float16

    def __post_init__(self) -> None:
        if self.feature_channels <= 0:
            raise ValueError(f"feature_channels must be positive, got {self.feature_channels}")
        if self.d_intrinsic <= 0:
            raise ValueError(f"d_intrinsic must be positive, got {self.d_intrinsic}")
        if len(self.spatial_size) != 2:
            raise ValueError(f"Expected spatial_size with 2 dims, got {self.spatial_size}")
        if len(self.hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer width")

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
    ) -> "FuXiIntrinsicConfig":
        resolved_config_path, intrinsic_data = load_config_section("intrinsic_model", config_path)
        forecast_config = FuXiLowerResConfig.from_yaml(resolved_config_path)

        return cls(
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
            hidden_dims=_to_int_tuple(intrinsic_data.get("hidden_dims", [2048, 512])),
            apply_tanh=bool(intrinsic_data.get("apply_tanh", True)),
            config_path=resolved_config_path,
            device=intrinsic_data.get("device", forecast_config.device),
            dtype=resolve_torch_dtype(
                intrinsic_data.get("dtype", str(forecast_config.dtype).replace("torch.", ""))
            ),
        )

    @property
    def flat_dim(self) -> int:
        return self.feature_channels * self.spatial_size[0] * self.spatial_size[1]


def _build_mlp(
    dims: list[int],
    *,
    final_activation: nn.Module | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> nn.Sequential:
    dd = {"device": device, "dtype": dtype}
    layers: list[nn.Module] = []
    for index in range(len(dims) - 1):
        layers.append(nn.Linear(dims[index], dims[index + 1], **dd))
        is_last = index == len(dims) - 2
        if not is_last:
            layers.append(nn.SiLU())
        elif final_activation is not None:
            layers.append(final_activation)
    return nn.Sequential(*layers)


class FuXiIntrinsic(nn.Module):
    """Separate intrinsic latent branch that maps second-block features to Z_intrinsic and back."""

    def __init__(self, config: FuXiIntrinsicConfig | None = None) -> None:
        super().__init__()
        self.config = config or FuXiIntrinsicConfig.from_yaml()
        dd = {"device": self.config.device, "dtype": self.config.dtype}

        encoder_dims = [self.config.flat_dim, *self.config.hidden_dims, self.config.d_intrinsic]
        decoder_dims = [self.config.d_intrinsic, *reversed(self.config.hidden_dims), self.config.flat_dim]
        final_activation = nn.Tanh() if self.config.apply_tanh else None

        self.encoder = _build_mlp(encoder_dims, final_activation=final_activation, **dd)
        self.decoder = _build_mlp(decoder_dims, **dd)

    def _validate_second_block_features(self, second_block_features: Tensor) -> None:
        expected_shape = (self.config.feature_channels, *self.config.spatial_size)
        if second_block_features.ndim != 4 or tuple(second_block_features.shape[1:]) != expected_shape:
            raise ValueError(
                "Expected second_block_features shaped "
                f"[B, {self.config.feature_channels}, {self.config.spatial_size[0]}, {self.config.spatial_size[1]}], "
                f"got {tuple(second_block_features.shape)}"
            )

    def forward(self, second_block_features: Tensor) -> dict[str, Tensor]:
        self._validate_second_block_features(second_block_features)
        batch_size = second_block_features.shape[0]
        flat = second_block_features.reshape(batch_size, -1)
        z_intrinsic = self.encoder(flat)
        second_block_features_recon = self.decoder(z_intrinsic).reshape(
            batch_size,
            self.config.feature_channels,
            *self.config.spatial_size,
        )
        return {
            "z_intrinsic": z_intrinsic,
            "second_block_features_recon": second_block_features_recon,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config.config_path),
            "feature_channels": self.config.feature_channels,
            "spatial_size": list(self.config.spatial_size),
            "flat_dim": self.config.flat_dim,
            "d_intrinsic": self.config.d_intrinsic,
            "hidden_dims": list(self.config.hidden_dims),
            "apply_tanh": self.config.apply_tanh,
            "parameter_device": str(self.encoder[0].weight.device),
            "parameter_dtype": str(self.encoder[0].weight.dtype),
        }


__all__ = ["FuXiIntrinsic", "FuXiIntrinsicConfig"]
