from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from ..config import DEFAULT_MODEL_CONFIG_PATH, load_config_section, resolve_torch_dtype
from .fuxi_intrinsic import FuXiIntrinsicConfig


def _to_int_tuple(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def _default_hidden_dims(latent_dim: int) -> tuple[int, ...]:
    if int(latent_dim) <= 2:
        return (32, 64, 128, 64, 32)
    return (32, 64, 128, 256, 128, 64, 32)


@dataclass(frozen=True)
class LatentDynamicsConfig:
    """MLP neural vector field over intrinsic latent states."""

    latent_dim: int | None = None
    hidden_dims: tuple[int, ...] = ()
    activation: str = "relu"
    config_path: Path = DEFAULT_MODEL_CONFIG_PATH
    device: str | torch.device | None = "meta"
    dtype: torch.dtype | None = torch.float32

    def __post_init__(self) -> None:
        if self.latent_dim is not None and int(self.latent_dim) <= 0:
            raise ValueError(f"latent_dim must be positive when set, got {self.latent_dim}")
        if any(int(value) <= 0 for value in self.hidden_dims):
            raise ValueError(f"hidden_dims must all be positive, got {self.hidden_dims}")
        if self.activation not in {"relu", "gelu", "tanh", "silu"}:
            raise ValueError(
                "activation must be one of {'relu', 'gelu', 'tanh', 'silu'}, "
                f"got {self.activation!r}"
            )

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
    ) -> "LatentDynamicsConfig":
        resolved_config_path, data = load_config_section("latent_dynamics_model", config_path)
        intrinsic_config = FuXiIntrinsicConfig.from_yaml(resolved_config_path)
        latent_dim = int(
            intrinsic_config.d_intrinsic if data.get("latent_dim") in {None, ""} else data.get("latent_dim")
        )
        hidden_dims = (
            _default_hidden_dims(latent_dim)
            if data.get("hidden_dims") in {None, ""}
            else _to_int_tuple(data.get("hidden_dims"))
        )
        return cls(
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            activation=str(data.get("activation", "relu")),
            config_path=resolved_config_path,
            device=data.get("device", intrinsic_config.device),
            dtype=resolve_torch_dtype(
                data.get("dtype", str(intrinsic_config.dtype).replace("torch.", ""))
            ),
        )

    @property
    def resolved_latent_dim(self) -> int:
        if self.latent_dim is None:
            raise ValueError("latent_dim is not set.")
        return int(self.latent_dim)


class NeuralLatentDynamics(nn.Module):
    """Neural vector field dz/dt = f(z) over intrinsic latent coordinates."""

    def __init__(self, config: LatentDynamicsConfig | None = None) -> None:
        super().__init__()
        self.config = config or LatentDynamicsConfig.from_yaml()
        dd = {"device": self.config.device, "dtype": self.config.dtype}

        layers: list[nn.Module] = []
        input_dim = self.config.resolved_latent_dim
        for hidden_dim in self.config.hidden_dims:
            layers.append(nn.Linear(input_dim, int(hidden_dim), **dd))
            layers.append(self._make_activation())
            input_dim = int(hidden_dim)
        layers.append(nn.Linear(input_dim, self.config.resolved_latent_dim, **dd))
        self.network = nn.Sequential(*layers)

    def _make_activation(self) -> nn.Module:
        activation = self.config.activation
        if activation == "relu":
            return nn.ReLU()
        if activation == "gelu":
            return nn.GELU()
        if activation == "tanh":
            return nn.Tanh()
        if activation == "silu":
            return nn.SiLU()
        raise ValueError(f"Unsupported activation {activation!r}")

    def _model_dtype(self) -> torch.dtype:
        first_linear = next(module for module in self.network if isinstance(module, nn.Linear))
        return first_linear.weight.dtype

    def forward(self, z: Tensor) -> Tensor:
        if z.ndim < 2 or z.shape[-1] != self.config.resolved_latent_dim:
            raise ValueError(
                f"Expected latent states shaped [..., {self.config.resolved_latent_dim}], got {tuple(z.shape)}"
            )
        return self.network(z.to(dtype=self._model_dtype()))

    def summary(self) -> dict[str, Any]:
        first_linear = next(module for module in self.network if isinstance(module, nn.Linear))
        linear_layers = [module for module in self.network if isinstance(module, nn.Linear)]
        return {
            "config_path": str(self.config.config_path),
            "latent_dim": self.config.resolved_latent_dim,
            "hidden_dims": list(self.config.hidden_dims),
            "activation": self.config.activation,
            "layer_count": len(linear_layers),
            "parameter_device": str(first_linear.weight.device),
            "parameter_dtype": str(first_linear.weight.dtype),
        }


__all__ = ["LatentDynamicsConfig", "NeuralLatentDynamics"]
