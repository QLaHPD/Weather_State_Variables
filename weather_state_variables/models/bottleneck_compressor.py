from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from ..config import DEFAULT_MODEL_CONFIG_PATH, load_config_section, resolve_torch_dtype
from .fuxi_lower_res import FuXiLowerResConfig


def _to_2tuple(values: Sequence[int]) -> tuple[int, int]:
    if len(values) != 2:
        raise ValueError(f"Expected 2 spatial values, got {values}")
    return int(values[0]), int(values[1])


def _clone_transformer_encoder_layer(
    *,
    model_dim: int,
    num_heads: int,
    mlp_hidden_dim: int,
    dropout: float,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=model_dim,
        nhead=num_heads,
        dim_feedforward=mlp_hidden_dim,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
        device=device,
        dtype=dtype,
    )


@dataclass(frozen=True)
class FuXiBottleneckCompressorConfig:
    """Transformer autoencoder that compresses each spatial latent token to a small channel count."""

    input_channels: int | None = None
    spatial_size: tuple[int, int] = (23, 45)
    model_dim: int = 256
    bottleneck_channels: int = 1
    num_heads: int = 8
    encoder_depth: int = 2
    decoder_depth: int = 2
    mlp_hidden_dim: int = 1024
    dropout: float = 0.0
    positional_embedding: str = "learned_2d"
    feature_source: str = "second_block_features"
    config_path: Path = DEFAULT_MODEL_CONFIG_PATH
    device: str | torch.device | None = "meta"
    dtype: torch.dtype | None = torch.float32

    def __post_init__(self) -> None:
        if self.input_channels is not None and int(self.input_channels) <= 0:
            raise ValueError(f"input_channels must be positive when set, got {self.input_channels}")
        if len(self.spatial_size) != 2:
            raise ValueError(f"Expected spatial_size with 2 dims, got {self.spatial_size}")
        if any(int(value) <= 0 for value in self.spatial_size):
            raise ValueError(f"spatial_size must be positive, got {self.spatial_size}")
        if self.model_dim <= 0:
            raise ValueError(f"model_dim must be positive, got {self.model_dim}")
        if self.bottleneck_channels <= 0:
            raise ValueError(f"bottleneck_channels must be positive, got {self.bottleneck_channels}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.model_dim % self.num_heads != 0:
            raise ValueError(
                f"model_dim must be divisible by num_heads, got {self.model_dim} and {self.num_heads}"
            )
        if self.encoder_depth <= 0 or self.decoder_depth <= 0:
            raise ValueError(
                f"encoder_depth and decoder_depth must be positive, got {self.encoder_depth}, {self.decoder_depth}"
            )
        if self.mlp_hidden_dim <= 0:
            raise ValueError(f"mlp_hidden_dim must be positive, got {self.mlp_hidden_dim}")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.positional_embedding not in {"learned_2d", "none"}:
            raise ValueError(
                "positional_embedding must be one of {'learned_2d', 'none'}, "
                f"got {self.positional_embedding!r}"
            )
        if self.feature_source not in {"second_block_features", "patch_grid_features"}:
            raise ValueError(
                "feature_source must be either 'second_block_features' or 'patch_grid_features', "
                f"got {self.feature_source!r}"
            )

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
    ) -> "FuXiBottleneckCompressorConfig":
        resolved_config_path, compressor_data = load_config_section(
            "bottleneck_compressor_model",
            config_path,
        )
        forecast_config = FuXiLowerResConfig.from_yaml(resolved_config_path)
        feature_source = str(compressor_data.get("feature_source", "second_block_features"))
        default_spatial_size = (
            forecast_config.patch_grid
            if feature_source == "patch_grid_features"
            else forecast_config.latent_grid
        )

        return cls(
            input_channels=(
                forecast_config.embed_dim
                if compressor_data.get("input_channels") in {None, ""}
                else int(compressor_data.get("input_channels"))
            ),
            spatial_size=_to_2tuple(compressor_data.get("spatial_size", list(default_spatial_size))),
            model_dim=int(compressor_data.get("model_dim", 256)),
            bottleneck_channels=int(compressor_data.get("bottleneck_channels", 1)),
            num_heads=int(compressor_data.get("num_heads", forecast_config.num_heads)),
            encoder_depth=int(compressor_data.get("encoder_depth", 2)),
            decoder_depth=int(compressor_data.get("decoder_depth", 2)),
            mlp_hidden_dim=int(
                compressor_data.get("mlp_hidden_dim", max(int(forecast_config.embed_dim), 1024))
            ),
            dropout=float(compressor_data.get("dropout", 0.0)),
            positional_embedding=str(compressor_data.get("positional_embedding", "learned_2d")),
            feature_source=feature_source,
            config_path=resolved_config_path,
            device=compressor_data.get("device", forecast_config.device),
            dtype=resolve_torch_dtype(
                compressor_data.get("dtype", str(forecast_config.dtype).replace("torch.", ""))
            ),
        )

    @property
    def resolved_input_channels(self) -> int:
        if self.input_channels is None:
            return self.model_dim
        return int(self.input_channels)

    @property
    def sequence_length(self) -> int:
        return int(self.spatial_size[0]) * int(self.spatial_size[1])

    @property
    def bottleneck_shape(self) -> tuple[int, int, int]:
        return (self.bottleneck_channels, *self.spatial_size)


class FuXiBottleneckCompressor(nn.Module):
    """Normal-transformer compressor for the main encoder bottleneck feature grid.

    The only information crossing the autoencoder midpoint is
    [B, bottleneck_channels, H, W]. No encoder/decoder skip connection is used.
    """

    def __init__(self, config: FuXiBottleneckCompressorConfig | None = None) -> None:
        super().__init__()
        self.config = config or FuXiBottleneckCompressorConfig.from_yaml()
        dd = {"device": self.config.device, "dtype": self.config.dtype}
        input_channels = self.config.resolved_input_channels

        self.input_proj = nn.Linear(input_channels, self.config.model_dim, **dd)
        self.encoder = nn.TransformerEncoder(
            _clone_transformer_encoder_layer(
                model_dim=self.config.model_dim,
                num_heads=self.config.num_heads,
                mlp_hidden_dim=self.config.mlp_hidden_dim,
                dropout=self.config.dropout,
                **dd,
            ),
            num_layers=self.config.encoder_depth,
            norm=nn.LayerNorm(self.config.model_dim, **dd),
        )
        self.to_bottleneck = nn.Linear(self.config.model_dim, self.config.bottleneck_channels, **dd)
        self.from_bottleneck = nn.Linear(self.config.bottleneck_channels, self.config.model_dim, **dd)
        self.decoder = nn.TransformerEncoder(
            _clone_transformer_encoder_layer(
                model_dim=self.config.model_dim,
                num_heads=self.config.num_heads,
                mlp_hidden_dim=self.config.mlp_hidden_dim,
                dropout=self.config.dropout,
                **dd,
            ),
            num_layers=self.config.decoder_depth,
            norm=nn.LayerNorm(self.config.model_dim, **dd),
        )
        self.output_proj = nn.Linear(self.config.model_dim, input_channels, **dd)

        if self.config.positional_embedding == "learned_2d":
            self.row_pos = nn.Parameter(torch.empty(self.config.spatial_size[0], self.config.model_dim, **dd))
            self.col_pos = nn.Parameter(torch.empty(self.config.spatial_size[1], self.config.model_dim, **dd))
        else:
            self.register_parameter("row_pos", None)
            self.register_parameter("col_pos", None)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if self.row_pos is None or self.col_pos is None:
            return
        if self.row_pos.is_meta or self.col_pos.is_meta:
            return
        nn.init.trunc_normal_(self.row_pos, std=0.02)
        nn.init.trunc_normal_(self.col_pos, std=0.02)

    def _model_dtype(self) -> torch.dtype:
        return self.input_proj.weight.dtype

    def _validate_feature_grid(self, feature_grid: Tensor) -> None:
        expected_shape = (self.config.resolved_input_channels, *self.config.spatial_size)
        if feature_grid.ndim != 4 or tuple(feature_grid.shape[1:]) != expected_shape:
            raise ValueError(
                "Expected compressor input features shaped "
                f"[B, {self.config.resolved_input_channels}, {self.config.spatial_size[0]}, {self.config.spatial_size[1]}], "
                f"got {tuple(feature_grid.shape)}"
            )

    def _validate_bottleneck_grid(self, z_bottleneck: Tensor) -> None:
        expected_shape = (self.config.bottleneck_channels, *self.config.spatial_size)
        if z_bottleneck.ndim != 4 or tuple(z_bottleneck.shape[1:]) != expected_shape:
            raise ValueError(
                "Expected z_bottleneck shaped "
                f"[B, {self.config.bottleneck_channels}, {self.config.spatial_size[0]}, {self.config.spatial_size[1]}], "
                f"got {tuple(z_bottleneck.shape)}"
            )

    def _grid_to_tokens(self, feature_grid: Tensor) -> Tensor:
        batch_size, channels, height, width = feature_grid.shape
        del channels
        return feature_grid.permute(0, 2, 3, 1).reshape(batch_size, height * width, -1)

    def _tokens_to_grid(self, tokens: Tensor, channels: int) -> Tensor:
        batch_size = tokens.shape[0]
        height, width = self.config.spatial_size
        return tokens.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()

    def _positional_tokens(self) -> Tensor | None:
        if self.row_pos is None or self.col_pos is None:
            return None
        pos = self.row_pos[:, None, :] + self.col_pos[None, :, :]
        return pos.reshape(1, self.config.sequence_length, self.config.model_dim)

    def _add_positional_tokens(self, tokens: Tensor) -> Tensor:
        pos = self._positional_tokens()
        if pos is None:
            return tokens
        return tokens + pos.to(device=tokens.device, dtype=tokens.dtype)

    def encode(self, feature_grid: Tensor) -> Tensor:
        self._validate_feature_grid(feature_grid)
        tokens = self._grid_to_tokens(feature_grid.to(dtype=self._model_dtype()))
        tokens = self.input_proj(tokens)
        tokens = self._add_positional_tokens(tokens)
        tokens = self.encoder(tokens)
        z_tokens = self.to_bottleneck(tokens)
        return self._tokens_to_grid(z_tokens, self.config.bottleneck_channels)

    def decode(self, z_bottleneck: Tensor) -> Tensor:
        self._validate_bottleneck_grid(z_bottleneck)
        z_tokens = self._grid_to_tokens(z_bottleneck.to(dtype=self._model_dtype()))
        tokens = self.from_bottleneck(z_tokens)
        tokens = self._add_positional_tokens(tokens)
        tokens = self.decoder(tokens)
        recon_tokens = self.output_proj(tokens)
        return self._tokens_to_grid(recon_tokens, self.config.resolved_input_channels)

    def forward(self, feature_grid: Tensor) -> dict[str, Tensor]:
        z_bottleneck = self.encode(feature_grid)
        second_block_features_recon = self.decode(z_bottleneck)
        return {
            "z_bottleneck": z_bottleneck,
            "bottleneck_features": z_bottleneck,
            "second_block_features_recon": second_block_features_recon,
            "feature_grid_recon": second_block_features_recon,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config.config_path),
            "input_channels": self.config.resolved_input_channels,
            "output_channels": self.config.resolved_input_channels,
            "spatial_size": list(self.config.spatial_size),
            "sequence_length": self.config.sequence_length,
            "model_dim": self.config.model_dim,
            "bottleneck_channels": self.config.bottleneck_channels,
            "bottleneck_shape": list(self.config.bottleneck_shape),
            "num_heads": self.config.num_heads,
            "encoder_depth": self.config.encoder_depth,
            "decoder_depth": self.config.decoder_depth,
            "mlp_hidden_dim": self.config.mlp_hidden_dim,
            "dropout": self.config.dropout,
            "positional_embedding": self.config.positional_embedding,
            "uses_positional_embeddings": self.config.positional_embedding != "none",
            "feature_source": self.config.feature_source,
            "architecture": "transformer_grid_bottleneck_autoencoder",
            "transformer_type": "torch.nn.TransformerEncoder",
            "bottleneck_projection": "per_token_channel_projection",
            "uses_skip_connections": False,
            "parameter_device": str(self.input_proj.weight.device),
            "parameter_dtype": str(self.input_proj.weight.dtype),
        }


__all__ = ["FuXiBottleneckCompressor", "FuXiBottleneckCompressorConfig"]
