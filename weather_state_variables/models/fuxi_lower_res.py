from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from timm.models.swin_transformer_v2 import SwinTransformerV2Block
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import (
    DEFAULT_MODEL_CONFIG_PATH,
    load_config_section,
    resolve_repo_path,
    resolve_torch_dtype,
)
from .fuxi_short import DEFAULT_MODEL_PATH, summarize_short_onnx_architecture


def _to_2tuple(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, Sequence):
        if len(value) != 2:
            raise ValueError(f"Expected 2 values, got {len(value)}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _resolve_group_count(channels: int, requested_groups: int) -> int:
    groups = min(channels, requested_groups)
    while channels % groups != 0:
        groups -= 1
    return max(groups, 1)


class GeluGatedMlp(nn.Module):
    """FuXi-style gated GELU MLP inferred from the ONNX block shapes."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        dd = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features * 2, **dd)
        self.fc2 = nn.Linear(hidden_features, out_features, **dd)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.gelu(gate) * value)


class FuXiSwinV2Block(SwinTransformerV2Block):
    """Swin V2 block with FuXi's gated GELU MLP and dynamic masks."""

    def __init__(
        self,
        *,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_hidden_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            dynamic_mask=True,
            mlp_ratio=1.0,
            qkv_bias=True,
            proj_drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer="gelu",
            norm_layer=nn.LayerNorm,
            pretrained_window_size=0,
            device=device,
            dtype=dtype,
        )
        self.mlp = GeluGatedMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            device=device,
            dtype=dtype,
        )


class PatchEmbedMergedTime(nn.Module):
    """Resize, append static channels, merge time into channels, then patch embed."""

    def __init__(
        self,
        *,
        time_steps: int,
        in_chans: int,
        aux_chans: int,
        embed_dim: int,
        patch_size: tuple[int, int],
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        dd = {"device": device, "dtype": dtype}
        super().__init__()
        self.time_steps = time_steps
        self.in_chans = in_chans
        self.aux_chans = aux_chans
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            time_steps * (in_chans + aux_chans),
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            **dd,
        )
        self.norm = nn.LayerNorm(embed_dim, **dd)

    def forward(self, x: Tensor, static_features: Tensor) -> Tensor:
        batch, time_steps, channels, height, width = x.shape
        if time_steps != self.time_steps:
            raise ValueError(f"Expected {self.time_steps} time steps, got {time_steps}")
        if channels != self.in_chans:
            raise ValueError(f"Expected {self.in_chans} channels, got {channels}")
        if static_features.shape[:3] != (batch, time_steps, self.aux_chans):
            raise ValueError(
                "Expected static features shaped "
                f"[B, {self.time_steps}, {self.aux_chans}, H, W], got {tuple(static_features.shape)}"
            )

        x = torch.cat([x, static_features], dim=2)
        x = x.reshape(batch, time_steps * (channels + self.aux_chans), height, width)
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return self.norm(x)


class ScaleShiftResBlock(nn.Module):
    """FuXi residual block with time-conditioned scale/shift normalization."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_groups: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        dd = {"device": device, "dtype": dtype}
        super().__init__()
        in_groups = _resolve_group_count(in_channels, num_groups)
        out_groups = _resolve_group_count(out_channels, num_groups)
        self.norm1 = nn.GroupNorm(in_groups, in_channels, **dd)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **dd)

        self.emb_act = nn.SiLU()
        self.emb_proj = nn.Linear(temb_channels, out_channels * 2, **dd)

        self.norm2 = nn.GroupNorm(out_groups, out_channels, **dd)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, **dd)

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, **dd)

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
        residual = self.skip(x)
        h = self.conv1(self.act1(self.norm1(x)))

        scale, shift = self.emb_proj(self.emb_act(temb)).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.act2(h))
        return h + residual


class FuXiTransformerStage(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_hidden_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        shift_size = window_size // 2
        self.blocks = nn.ModuleList(
            [
                FuXiSwinV2Block(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if block_index % 2 == 0 else shift_size,
                    mlp_hidden_dim=mlp_hidden_dim,
                    device=device,
                    dtype=dtype,
                )
                for block_index in range(depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


@dataclass(frozen=True)
class FuXiLowerResConfig:
    """Config-driven lower-resolution forecast model with a midpoint Z_high bottleneck."""

    input_size: tuple[int, int] = (181, 360)
    time_steps: int = 2
    in_chans: int = 70
    aux_chans: int = 5
    out_chans: int = 70
    forecast_steps: int = 2
    temb_dim: int = 12
    patch_size: tuple[int, int] = (4, 4)
    embed_dim: int = 1536
    num_heads: int = 24
    window_size: int = 9
    depths: tuple[int, int, int, int] = (12, 12, 12, 12)
    num_groups: int = 32
    mlp_hidden_dim: int = 4096
    d_high: int = 128
    source_model_path: Path = DEFAULT_MODEL_PATH
    config_path: Path = DEFAULT_MODEL_CONFIG_PATH
    device: str | torch.device | None = "meta"
    dtype: torch.dtype | None = torch.float16

    def __post_init__(self) -> None:
        if len(self.depths) != 4:
            raise ValueError(f"Expected 4 transformer stages, got {len(self.depths)}")
        if self.d_high <= 0:
            raise ValueError(f"d_high must be positive, got {self.d_high}")
        if self.forecast_steps <= 0:
            raise ValueError(f"forecast_steps must be positive, got {self.forecast_steps}")

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
    ) -> "FuXiLowerResConfig":
        resolved_config_path, data = load_config_section("forecast_model", config_path)
        source_model_path = resolve_repo_path(
            data.get("source_model_path", str(DEFAULT_MODEL_PATH)),
            config_path=resolved_config_path,
        )
        recipe = summarize_short_onnx_architecture(source_model_path)

        return cls(
            input_size=tuple(int(value) for value in data.get("input_size", [181, 360])),
            time_steps=int(data.get("time_steps", recipe["time_steps"])),
            in_chans=int(data.get("in_chans", recipe["in_chans"])),
            aux_chans=int(data.get("aux_chans", recipe["aux_chans"])),
            out_chans=int(data.get("out_chans", recipe["out_chans"])),
            forecast_steps=int(data.get("forecast_steps", 2)),
            temb_dim=int(data.get("temb_dim", recipe["temb_dim"])),
            patch_size=tuple(int(value) for value in data.get("patch_size", recipe["patch_size"])),
            embed_dim=int(data.get("embed_dim", recipe["embed_dim"])),
            num_heads=int(data.get("num_heads", recipe["num_heads"])),
            window_size=int(data.get("window_size", recipe["window_size"][0])),
            depths=tuple(int(value) for value in data.get("depths", recipe["depths"])),
            num_groups=int(data.get("num_groups", 32)),
            mlp_hidden_dim=int(data.get("mlp_hidden_dim", recipe["mlp_hidden_dim"])),
            d_high=int(data["d_high"]),
            source_model_path=source_model_path,
            config_path=resolved_config_path,
            device=data.get("device", "meta"),
            dtype=resolve_torch_dtype(data.get("dtype", "float16")),
        )

    @property
    def resized_input_size(self) -> tuple[int, int]:
        patch_height, patch_width = self.patch_size
        return (
            (self.input_size[0] // patch_height) * patch_height,
            (self.input_size[1] // patch_width) * patch_width,
        )

    @property
    def patch_grid(self) -> tuple[int, int]:
        resized_height, resized_width = self.resized_input_size
        patch_height, patch_width = self.patch_size
        return resized_height // patch_height, resized_width // patch_width

    @property
    def latent_grid(self) -> tuple[int, int]:
        patch_height, patch_width = self.patch_grid
        return (patch_height + 1) // 2, (patch_width + 1) // 2


@dataclass(frozen=True)
class FuXiEncoderOutput:
    z_high: Tensor
    temb_emb: Tensor
    skip: Tensor
    output_size: tuple[int, int]


class FuXiLowerResEncoder(nn.Module):
    """Encoder half of the lower-resolution FuXi-style forecast model."""

    def __init__(self, config: FuXiLowerResConfig | None = None) -> None:
        super().__init__()
        self.config = config or FuXiLowerResConfig.from_yaml()
        dd = {"device": self.config.device, "dtype": self.config.dtype}

        self.patch_embed = PatchEmbedMergedTime(
            time_steps=self.config.time_steps,
            in_chans=self.config.in_chans,
            aux_chans=self.config.aux_chans,
            embed_dim=self.config.embed_dim,
            patch_size=self.config.patch_size,
            **dd,
        )
        self.time_embed = nn.Sequential(
            nn.Linear(self.config.temb_dim, self.config.embed_dim, **dd),
            nn.SiLU(),
            nn.Linear(self.config.embed_dim, self.config.embed_dim, **dd),
        )
        self.downsample = nn.Conv2d(
            self.config.embed_dim,
            self.config.embed_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            **dd,
        )
        self.down_resblock = ScaleShiftResBlock(
            in_channels=self.config.embed_dim,
            out_channels=self.config.embed_dim,
            temb_channels=self.config.embed_dim,
            num_groups=self.config.num_groups,
            **dd,
        )
        self.first_pair_layers = nn.ModuleList(
            [
                FuXiTransformerStage(
                    dim=self.config.embed_dim,
                    input_resolution=self.config.latent_grid,
                    depth=depth,
                    num_heads=self.config.num_heads,
                    window_size=self.config.window_size,
                    mlp_hidden_dim=self.config.mlp_hidden_dim,
                    **dd,
                )
                for depth in self.config.depths[:2]
            ]
        )
        self.first_pair_fusion = nn.Linear(self.config.embed_dim * 2, self.config.embed_dim, **dd)
        self.z_high_proj = nn.Conv2d(self.config.embed_dim, self.config.d_high, kernel_size=1, **dd)
        self.register_buffer(
            "default_static_features",
            torch.zeros(1, self.config.aux_chans, *self.config.input_size, **dd),
            persistent=False,
        )

    def _model_dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    def _validate_input(self, x: Tensor, temb: Tensor) -> None:
        if x.ndim != 5:
            raise ValueError(f"Expected x shaped [B, T, C, H, W], got {tuple(x.shape)}")
        if tuple(x.shape[1:]) != (
            self.config.time_steps,
            self.config.in_chans,
            *self.config.input_size,
        ):
            raise ValueError(
                "Expected x shaped "
                f"[B, {self.config.time_steps}, {self.config.in_chans}, "
                f"{self.config.input_size[0]}, {self.config.input_size[1]}], "
                f"got {tuple(x.shape)}"
            )
        if temb.ndim != 2 or tuple(temb.shape[1:]) != (self.config.temb_dim,):
            raise ValueError(
                f"Expected temb shaped [B, {self.config.temb_dim}], got {tuple(temb.shape)}"
            )
        if x.shape[0] != temb.shape[0]:
            raise ValueError(
                f"Expected matching batch sizes, got x batch {x.shape[0]} and temb batch {temb.shape[0]}"
            )

    def _resize_steps(self, x: Tensor, size: tuple[int, int]) -> Tensor:
        batch, steps, channels, height, width = x.shape
        x = x.reshape(batch * steps, channels, height, width).float()
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return x.reshape(batch, steps, channels, *size).to(dtype=self._model_dtype())

    def _resize_map(self, x: Tensor, size: tuple[int, int]) -> Tensor:
        x = x.float()
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return x.to(dtype=self._model_dtype())

    def _prepare_static_features(
        self,
        batch_size: int,
        static_features: Tensor | None,
    ) -> Tensor:
        target_size = self.config.resized_input_size

        if static_features is None:
            static = self.default_static_features
        else:
            static = static_features
            if static.ndim == 3:
                static = static.unsqueeze(0)
            if static.ndim == 5:
                if static.shape[1] == 1 and self.config.time_steps != 1:
                    static = static.expand(-1, self.config.time_steps, -1, -1, -1)
                if static.shape[1] != self.config.time_steps or static.shape[2] != self.config.aux_chans:
                    raise ValueError(
                        "Expected static_features shaped "
                        f"[B, {self.config.time_steps}, {self.config.aux_chans}, H, W], "
                        f"got {tuple(static.shape)}"
                    )
                batch, steps, channels, height, width = static.shape
                static = static.reshape(batch * steps, channels, height, width)
                static = self._resize_map(static, target_size)
                return static.reshape(batch, steps, channels, *target_size)
            if static.ndim != 4:
                raise ValueError(
                    "Expected static_features shaped [B, aux, H, W] or [B, T, aux, H, W], "
                    f"got {tuple(static.shape)}"
                )
            if static.shape[1] != self.config.aux_chans:
                raise ValueError(
                    f"Expected {self.config.aux_chans} static channels, got {static.shape[1]}"
                )

        if static.shape[0] == 1 and batch_size != 1:
            static = static.expand(batch_size, -1, -1, -1)
        if static.shape[0] != batch_size:
            raise ValueError(
                f"Expected static_features batch {batch_size}, got {static.shape[0]}"
            )
        static = self._resize_map(static, target_size)
        return static.unsqueeze(1).expand(-1, self.config.time_steps, -1, -1, -1)

    def forward(
        self,
        x: Tensor,
        temb: Tensor,
        static_features: Tensor | None = None,
    ) -> FuXiEncoderOutput:
        self._validate_input(x, temb)

        original_size = tuple(int(value) for value in x.shape[-2:])
        resized_size = self.config.resized_input_size
        x_resized = self._resize_steps(x, resized_size)
        static = self._prepare_static_features(x.shape[0], static_features)
        temb_emb = self.time_embed(temb.to(dtype=self._model_dtype()))

        h = self.patch_embed(x_resized.to(dtype=self._model_dtype()), static)
        h = h.permute(0, 3, 1, 2)
        h = self.downsample(h)
        h = self.down_resblock(h, temb_emb)
        skip = h

        h = h.permute(0, 2, 3, 1)
        s0 = self.first_pair_layers[0](h)
        s1 = self.first_pair_layers[1](s0)
        f01 = self.first_pair_fusion(torch.cat([s0, s1], dim=-1))
        z_high = self.z_high_proj(f01.permute(0, 3, 1, 2))
        return FuXiEncoderOutput(
            z_high=z_high,
            temb_emb=temb_emb,
            skip=skip,
            output_size=original_size,
        )

    def summary(self) -> dict[str, Any]:
        recipe = summarize_short_onnx_architecture(self.config.source_model_path)
        return {
            "role": "encoder",
            "config_path": str(self.config.config_path),
            "source_model_path": str(self.config.source_model_path),
            "source_window_size": recipe["window_size"],
            "source_depths": recipe["depths"],
            "source_num_heads": recipe["num_heads"],
            "input_size": list(self.config.input_size),
            "resized_input_size": list(self.config.resized_input_size),
            "patch_grid": list(self.config.patch_grid),
            "latent_grid": list(self.config.latent_grid),
            "time_steps": self.config.time_steps,
            "in_chans": self.config.in_chans,
            "aux_chans": self.config.aux_chans,
            "embed_dim": self.config.embed_dim,
            "num_heads": self.config.num_heads,
            "window_size": self.config.window_size,
            "depths": list(self.config.depths[:2]),
            "mlp_hidden_dim": self.config.mlp_hidden_dim,
            "pair_fusion_in_dim": self.config.embed_dim * 2,
            "d_high": self.config.d_high,
            "z_high_shape": [self.config.d_high, *self.config.latent_grid],
            "parameter_device": str(self.patch_embed.proj.weight.device),
            "parameter_dtype": str(self.patch_embed.proj.weight.dtype),
        }


class FuXiLowerResDecoder(nn.Module):
    """Decoder half of the lower-resolution FuXi-style forecast model."""

    def __init__(self, config: FuXiLowerResConfig | None = None) -> None:
        super().__init__()
        self.config = config or FuXiLowerResConfig.from_yaml()
        dd = {"device": self.config.device, "dtype": self.config.dtype}
        self.z_high_expand = nn.Conv2d(self.config.d_high, self.config.embed_dim, kernel_size=1, **dd)
        self.second_pair_layers = nn.ModuleList(
            [
                FuXiTransformerStage(
                    dim=self.config.embed_dim,
                    input_resolution=self.config.latent_grid,
                    depth=depth,
                    num_heads=self.config.num_heads,
                    window_size=self.config.window_size,
                    mlp_hidden_dim=self.config.mlp_hidden_dim,
                    **dd,
                )
                for depth in self.config.depths[2:]
            ]
        )
        self.second_pair_fusion = nn.Linear(self.config.embed_dim * 2, self.config.embed_dim, **dd)
        self.up_resblock = ScaleShiftResBlock(
            in_channels=self.config.embed_dim * 2,
            out_channels=self.config.embed_dim,
            temb_channels=self.config.embed_dim,
            num_groups=self.config.num_groups,
            **dd,
        )
        self.upsample = nn.ConvTranspose2d(
            self.config.embed_dim,
            self.config.embed_dim,
            kernel_size=4,
            stride=2,
            padding=1,
            **dd,
        )
        patch_height, patch_width = self.config.patch_size
        self.head = nn.Linear(
            self.config.embed_dim,
            self.config.forecast_steps * self.config.out_chans * patch_height * patch_width,
            **dd,
        )

    def _model_dtype(self) -> torch.dtype:
        return self.z_high_expand.weight.dtype

    def _resize_future_maps(self, x: Tensor, size: tuple[int, int]) -> Tensor:
        batch, steps, channels, height, width = x.shape
        x = x.reshape(batch * steps, channels, height, width).float()
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return x.reshape(batch, steps, channels, *size).to(dtype=self._model_dtype())

    def _resize_map(self, x: Tensor, size: tuple[int, int]) -> Tensor:
        x = x.float()
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        return x.to(dtype=self._model_dtype())

    def forward(self, encoded: FuXiEncoderOutput) -> Tensor:
        h = self.z_high_expand(encoded.z_high).permute(0, 2, 3, 1)
        s2 = self.second_pair_layers[0](h)
        s3 = self.second_pair_layers[1](s2)
        f23 = self.second_pair_fusion(torch.cat([s2, s3], dim=-1))

        h = f23.permute(0, 3, 1, 2)
        h = torch.cat([h, encoded.skip], dim=1)
        h = self.up_resblock(h, encoded.temb_emb)
        h = self.upsample(h)

        h = h.permute(0, 2, 3, 1)
        h = self.head(h)
        batch, height_bins, width_bins, _ = h.shape
        patch_height, patch_width = self.config.patch_size
        h = h.reshape(
            batch,
            height_bins,
            width_bins,
            self.config.forecast_steps,
            patch_height,
            patch_width,
            self.config.out_chans,
        )
        h = h.permute(0, 3, 6, 1, 4, 2, 5)
        h = h.reshape(
            batch,
            self.config.forecast_steps,
            self.config.out_chans,
            height_bins * patch_height,
            width_bins * patch_width,
        )
        return self._resize_future_maps(h, encoded.output_size)

    def summary(self) -> dict[str, Any]:
        recipe = summarize_short_onnx_architecture(self.config.source_model_path)
        return {
            "role": "decoder",
            "config_path": str(self.config.config_path),
            "source_model_path": str(self.config.source_model_path),
            "source_window_size": recipe["window_size"],
            "source_depths": recipe["depths"],
            "source_num_heads": recipe["num_heads"],
            "latent_grid": list(self.config.latent_grid),
            "embed_dim": self.config.embed_dim,
            "num_heads": self.config.num_heads,
            "window_size": self.config.window_size,
            "depths": list(self.config.depths[2:]),
            "mlp_hidden_dim": self.config.mlp_hidden_dim,
            "pair_fusion_in_dim": self.config.embed_dim * 2,
            "d_high": self.config.d_high,
            "forecast_steps": self.config.forecast_steps,
            "head_out_dim": (
                self.config.forecast_steps
                * self.config.out_chans
                * self.config.patch_size[0]
                * self.config.patch_size[1]
            ),
            "parameter_device": str(self.z_high_expand.weight.device),
            "parameter_dtype": str(self.z_high_expand.weight.dtype),
        }


class FuXiLowerRes(nn.Module):
    """Config-driven lower-resolution forecast model with explicit encoder/decoder halves."""

    def __init__(self, config: FuXiLowerResConfig | None = None) -> None:
        super().__init__()
        self.config = config or FuXiLowerResConfig.from_yaml()
        self.encoder = FuXiLowerResEncoder(self.config)
        self.decoder = FuXiLowerResDecoder(self.config)

    def encode(
        self,
        x: Tensor,
        temb: Tensor,
        static_features: Tensor | None = None,
    ) -> FuXiEncoderOutput:
        return self.encoder(x, temb, static_features=static_features)

    def decode(self, encoded: FuXiEncoderOutput) -> Tensor:
        return self.decoder(encoded)

    def predict_future(self, x: Tensor, temb: Tensor, static_features: Tensor | None = None) -> Tensor:
        encoded = self.encode(x, temb, static_features=static_features)
        return self.decode(encoded)

    def predict_next(self, x: Tensor, temb: Tensor, static_features: Tensor | None = None) -> Tensor:
        return self.predict_future(x, temb, static_features=static_features)[:, 0]

    def forward(self, x: Tensor, temb: Tensor, static_features: Tensor | None = None) -> dict[str, Tensor]:
        encoded = self.encode(x, temb, static_features=static_features)
        forecast = self.decode(encoded)
        return {"forecast": forecast, "z_high": encoded.z_high}

    def summary(self) -> dict[str, Any]:
        recipe = summarize_short_onnx_architecture(self.config.source_model_path)
        return {
            "config_path": str(self.config.config_path),
            "source_model_path": str(self.config.source_model_path),
            "source_window_size": recipe["window_size"],
            "source_depths": recipe["depths"],
            "source_num_heads": recipe["num_heads"],
            "input_size": list(self.config.input_size),
            "resized_input_size": list(self.config.resized_input_size),
            "patch_grid": list(self.config.patch_grid),
            "latent_grid": list(self.config.latent_grid),
            "time_steps": self.config.time_steps,
            "in_chans": self.config.in_chans,
            "aux_chans": self.config.aux_chans,
            "out_chans": self.config.out_chans,
            "forecast_steps": self.config.forecast_steps,
            "embed_dim": self.config.embed_dim,
            "num_heads": self.config.num_heads,
            "window_size": self.config.window_size,
            "depths": list(self.config.depths),
            "mlp_hidden_dim": self.config.mlp_hidden_dim,
            "pair_fusion_in_dim": self.config.embed_dim * 2,
            "d_high": self.config.d_high,
            "z_high_shape": [self.config.d_high, *self.config.latent_grid],
            "head_out_dim": (
                self.config.forecast_steps
                * self.config.out_chans
                * self.config.patch_size[0]
                * self.config.patch_size[1]
            ),
            "parameter_device": str(self.encoder.patch_embed.proj.weight.device),
            "parameter_dtype": str(self.encoder.patch_embed.proj.weight.dtype),
        }


__all__ = [
    "FuXiEncoderOutput",
    "FuXiLowerRes",
    "FuXiLowerResConfig",
    "FuXiLowerResDecoder",
    "FuXiLowerResEncoder",
]
