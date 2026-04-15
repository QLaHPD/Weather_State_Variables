from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Sampler
try:
    from torchinfo import summary as torchinfo_summary
except ImportError:
    torchinfo_summary = None

from ..config import (
    DEFAULT_MODEL_CONFIG_PATH,
    load_config_section,
    resolve_repo_path,
    resolve_torch_dtype,
)
from ..data import (
    ArcoEra5FuXiDataConfig,
    ArcoEra5FuXiDataset,
    ContiguousDistributedSampler,
    build_arco_era5_dataloader,
)
from ..models import (
    FuXiIntrinsic,
    FuXiIntrinsicConfig,
    FuXiLowerRes,
    FuXiLowerResConfig,
    FuXiLowerResEncoder,
)


@dataclass(frozen=True)
class DistributedRuntime:
    enabled: bool
    backend: str | None
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _resolve_distributed_runtime(device_name: str) -> DistributedRuntime:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1

    if enabled:
        if device_name in {"auto", "cuda"} and torch.cuda.is_available():
            device = torch.device("cuda", local_rank)
        else:
            device = _resolve_device(device_name)

        if device.type == "cuda":
            torch.cuda.set_device(device)
            backend = "nccl"
        else:
            backend = "gloo"

        if not dist.is_initialized():
            dist.init_process_group(backend=backend)

        return DistributedRuntime(
            enabled=True,
            backend=backend,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=device,
        )

    return DistributedRuntime(
        enabled=False,
        backend=None,
        rank=0,
        local_rank=0,
        world_size=1,
        device=_resolve_device(device_name),
    )


def _cleanup_distributed_runtime(runtime: DistributedRuntime) -> None:
    if runtime.enabled and dist.is_initialized():
        dist.destroy_process_group()


def _wrap_for_distributed_training(
    model: nn.Module,
    runtime: DistributedRuntime,
    *,
    find_unused_parameters: bool = False,
) -> nn.Module:
    if not runtime.enabled:
        return model

    kwargs: dict[str, Any] = {"find_unused_parameters": find_unused_parameters}
    if runtime.device.type == "cuda":
        kwargs["device_ids"] = [runtime.local_rank]
        kwargs["output_device"] = runtime.local_rank
    return DistributedDataParallel(model, **kwargs)


def _unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def _to_optional_timestamp(value: Any) -> pd.Timestamp | None:
    if value in {None, ""}:
        return None
    return pd.Timestamp(value)


def _to_optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def _to_optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def _to_positive_int(value: Any, *, default: int = 1, field_name: str) -> int:
    if value in {None, ""}:
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive, got {parsed}")
    return parsed


def _to_optional_positive_int(value: Any, *, field_name: str) -> int | None:
    if value in {None, ""}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive when set, got {parsed}")
    return parsed


def _to_plain_data(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_plain_data(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_data(item) for item in value]
    return value


def _device_type_for_amp(device: torch.device) -> str:
    return "cuda" if device.type == "cuda" else "cpu"


def _amp_autocast_context(use_amp: bool, device: torch.device, amp_dtype: torch.dtype | None):
    if not use_amp or amp_dtype is None:
        return nullcontext()
    if device.type not in {"cuda", "cpu"}:
        return nullcontext()
    return torch.autocast(device_type=_device_type_for_amp(device), dtype=amp_dtype)


def _build_grad_scaler(use_amp: bool, device: torch.device):
    enabled = use_amp and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _validate_training_precision_config(
    *,
    section_name: str,
    use_amp: bool,
    model_dtype: torch.dtype,
    amp_dtype: torch.dtype | None,
) -> None:
    if use_amp and model_dtype == torch.float16:
        raise ValueError(
            f"{section_name}.model_dtype=float16 is not supported with AMP training in this pipeline. "
            "Use float32 master weights with AMP instead: set "
            f"{section_name}.model_dtype=float32 and keep {section_name}.amp_dtype=float16."
        )
    if use_amp and amp_dtype is None:
        raise ValueError(f"{section_name}.use_amp=true requires a valid {section_name}.amp_dtype.")


def _build_main_forecast_criterion(
    train_config: MainTrainingConfig,
    data_config: ArcoEra5FuXiDataConfig,
) -> nn.Module:
    loss_name = train_config.forecast_loss.strip().lower()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "charbonnier":
        return LatitudeWeightedCharbonnierLoss(
            data_config.channel_names,
            epsilon=train_config.charbonnier_epsilon,
            upper_air_weight=train_config.upper_air_loss_weight,
            surface_weight=train_config.surface_loss_weight,
            latitude_descending=data_config.latitude_descending,
        )
    raise ValueError(
        f"Unsupported train_main.forecast_loss {train_config.forecast_loss!r}. "
        "Expected 'charbonnier' or 'mse'."
    )


def _move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            moved[key] = value.to(device, non_blocking=non_blocking)
        else:
            moved[key] = value
    return moved


def _tensor_tree_shapes(value: Any) -> Any:
    if isinstance(value, Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {key: _tensor_tree_shapes(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_tensor_tree_shapes(item) for item in value]
    return value


def _print_json_block(title: str, payload: Any) -> None:
    print(title)
    print(json.dumps(_to_plain_data(payload), indent=2, sort_keys=True))


def _print_if_primary(runtime: DistributedRuntime, message: str) -> None:
    if runtime.is_primary:
        print(message)


def _reduced_sum_and_count(
    value_sum: float,
    count: int,
    runtime: DistributedRuntime,
) -> tuple[float, int]:
    if not runtime.enabled:
        return value_sum, count

    tensor = torch.tensor([value_sum, float(count)], device=runtime.device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor[0].item()), int(tensor[1].item())


def _reduced_mean_scalar(value: float, runtime: DistributedRuntime) -> float:
    if not runtime.enabled:
        return value

    tensor = torch.tensor([value], device=runtime.device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= runtime.world_size
    return float(tensor.item())


def _limited_length(loader: DataLoader[dict[str, Any]], max_batches: int | None) -> int:
    total = len(loader)
    if max_batches is None:
        return total
    return min(total, max_batches)


def _print_model_and_summary(
    title: str,
    model: nn.Module,
    *,
    input_data: tuple[Any, ...],
    depth: int,
    print_summary: bool,
) -> None:
    print(f"\n== {title} ==")
    print(model)
    if not print_summary:
        return
    if torchinfo_summary is None:
        print("torchinfo summary unavailable: torchinfo is not installed")
        if hasattr(model, "summary"):
            _print_json_block("fallback_summary", model.summary())
        return
    try:
        info = torchinfo_summary(
            model,
            input_data=input_data,
            depth=depth,
            verbose=0,
            col_names=("input_size", "output_size", "num_params", "trainable"),
        )
    except Exception as exc:
        print(f"torchinfo summary failed: {exc}")
        if hasattr(model, "summary"):
            _print_json_block("fallback_summary", model.summary())
    else:
        print(info)


def _make_main_random_inputs(
    model_config: FuXiLowerResConfig,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    dtype = model_config.dtype or torch.float32
    x = torch.randn(
        batch_size,
        model_config.time_steps,
        model_config.in_chans,
        *model_config.input_size,
        device=device,
        dtype=dtype,
    )
    temb = torch.randn(batch_size, model_config.temb_dim, device=device, dtype=dtype)
    static_features = torch.randn(
        batch_size,
        model_config.aux_chans,
        *model_config.input_size,
        device=device,
        dtype=dtype,
    )
    return x, temb, static_features


def _make_intrinsic_random_inputs(
    intrinsic_config: FuXiIntrinsicConfig,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    dtype = intrinsic_config.dtype or torch.float32
    return torch.randn(
        batch_size,
        intrinsic_config.feature_channels,
        *intrinsic_config.spatial_size,
        device=device,
        dtype=dtype,
    )


def run_main_model_smoke_test(
    model: FuXiLowerRes,
    *,
    batch_size: int,
    print_outputs: bool = True,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    x, temb, static_features = _make_main_random_inputs(model.config, batch_size=batch_size, device=device)
    with torch.no_grad():
        outputs = model(x, temb, static_features=static_features)
    report = {
        "input": {
            "x": {"shape": list(x.shape), "dtype": str(x.dtype)},
            "temb": {"shape": list(temb.shape), "dtype": str(temb.dtype)},
            "static_features": {"shape": list(static_features.shape), "dtype": str(static_features.dtype)},
        },
        "output": _tensor_tree_shapes(outputs),
    }
    if print_outputs:
        _print_json_block("main_smoke_test", report)
    return report


def run_intrinsic_model_smoke_test(
    encoder: FuXiLowerResEncoder,
    intrinsic_model: FuXiIntrinsic,
    *,
    batch_size: int,
    print_outputs: bool = True,
) -> dict[str, Any]:
    device = next(intrinsic_model.parameters()).device
    x, temb, static_features = _make_main_random_inputs(encoder.config, batch_size=batch_size, device=device)
    with torch.no_grad():
        encoded = encoder(
            x,
            temb,
            static_features=static_features,
            return_patch_grid_features=True,
        )
        if encoded.patch_grid_features is None:
            raise RuntimeError("Encoder did not return patch_grid_features for the intrinsic smoke test.")
        outputs = intrinsic_model(encoded.patch_grid_features)
    report = {
        "encoder_input": {
            "x": {"shape": list(x.shape), "dtype": str(x.dtype)},
            "temb": {"shape": list(temb.shape), "dtype": str(temb.dtype)},
            "static_features": {"shape": list(static_features.shape), "dtype": str(static_features.dtype)},
        },
        "patch_grid_features": {
            "shape": list(encoded.patch_grid_features.shape),
            "dtype": str(encoded.patch_grid_features.dtype),
        },
        "intrinsic_output": _tensor_tree_shapes(outputs),
    }
    if print_outputs:
        _print_json_block("intrinsic_smoke_test", report)
    return report


def _require_patch_grid_features(encoded: Any) -> Tensor:
    patch_grid_features = getattr(encoded, "patch_grid_features", None)
    if patch_grid_features is None:
        raise RuntimeError(
            "The encoder did not expose patch_grid_features. "
            "Call the encoder with return_patch_grid_features=True for intrinsic training."
        )
    return patch_grid_features


def _encode_patch_grid_features_for_intrinsic(
    encoder: FuXiLowerResEncoder,
    batch: dict[str, Tensor],
    *,
    detach_features: bool,
    clear_encoder_grads: bool,
) -> Tensor:
    if clear_encoder_grads:
        encoder.zero_grad(set_to_none=True)
    encoded = encoder(
        batch["x"],
        batch["temb"],
        static_features=batch["static_features"],
        return_patch_grid_features=True,
    )
    patch_grid_features = _require_patch_grid_features(encoded)
    return patch_grid_features.detach() if detach_features else patch_grid_features


def _latitude_weight_vector(
    height: int,
    *,
    latitude_descending: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if height <= 1:
        return torch.ones(height, device=device, dtype=dtype)

    start, end = (90.0, -90.0) if latitude_descending else (-90.0, 90.0)
    latitudes = torch.linspace(start, end, height, device=device, dtype=torch.float32)
    weights = torch.cos(torch.deg2rad(latitudes)).clamp_min(0.0)
    mean_weight = weights.mean().clamp_min(torch.finfo(weights.dtype).eps)
    return (weights / mean_weight).to(dtype=dtype)


def _forecast_channel_weight_vector(
    channel_names: list[str] | tuple[str, ...],
    *,
    upper_air_weight: float,
    surface_weight: float,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    surface_channels = {"T2M", "U10", "V10", "MSL", "TP"}
    weights = [
        surface_weight if channel_name in surface_channels else upper_air_weight
        for channel_name in channel_names
    ]
    return torch.tensor(weights, device=device, dtype=dtype or torch.float32)


class LatitudeWeightedCharbonnierLoss(nn.Module):
    def __init__(
        self,
        channel_names: list[str] | tuple[str, ...],
        *,
        epsilon: float = 1.0e-3,
        upper_air_weight: float = 1.0,
        surface_weight: float = 0.1,
        latitude_descending: bool = True,
    ) -> None:
        super().__init__()
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.epsilon = float(epsilon)
        self.latitude_descending = latitude_descending
        self.register_buffer(
            "channel_weights",
            _forecast_channel_weight_vector(
                list(channel_names),
                upper_air_weight=upper_air_weight,
                surface_weight=surface_weight,
            ),
            persistent=False,
        )

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"Expected prediction and target to have the same shape, got "
                f"{tuple(prediction.shape)} and {tuple(target.shape)}"
            )
        if prediction.ndim != 5:
            raise ValueError(
                f"Expected forecast tensors shaped [B, T, C, H, W], got {tuple(prediction.shape)}"
            )
        if prediction.shape[2] != int(self.channel_weights.shape[0]):
            raise ValueError(
                f"Expected {int(self.channel_weights.shape[0])} forecast channels, got {prediction.shape[2]}"
            )

        prediction = prediction.float()
        target = target.float()
        latitude_weights = _latitude_weight_vector(
            int(prediction.shape[-2]),
            latitude_descending=self.latitude_descending,
            device=prediction.device,
            dtype=prediction.dtype,
        ).view(1, 1, 1, -1, 1)
        channel_weights = self.channel_weights.to(device=prediction.device, dtype=prediction.dtype).view(
            1,
            1,
            -1,
            1,
            1,
        )
        weighted_diff = latitude_weights * (prediction - target)
        charbonnier = torch.sqrt(weighted_diff.square() + self.epsilon**2)
        return (channel_weights * charbonnier).mean()


def _forecast_variable_channel_groups(
    data_config: ArcoEra5FuXiDataConfig,
) -> list[tuple[str, list[int]]]:
    groups: list[tuple[str, list[int]]] = []
    channel_index = 0
    level_count = len(data_config.pressure_levels)
    for variable_name in data_config.upper_air_variables:
        groups.append((variable_name, list(range(channel_index, channel_index + level_count))))
        channel_index += level_count
    for variable_name in data_config.surface_variables:
        groups.append((variable_name, [channel_index]))
        channel_index += 1
    return groups


def _reduce_tensor_in_place(tensor: Tensor, runtime: DistributedRuntime) -> Tensor:
    if runtime.enabled:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def _main_validation_denormalized_mae(
    prediction: Tensor,
    target: Tensor,
    *,
    dataset: ArcoEra5FuXiDataset,
    data_config: ArcoEra5FuXiDataConfig,
) -> tuple[Tensor, Tensor]:
    prediction_denorm = dataset.denormalize_dynamic_tensor(prediction)
    target_denorm = dataset.denormalize_dynamic_tensor(target)
    absolute_error = (prediction_denorm - target_denorm).abs()
    latitude_weights = _latitude_weight_vector(
        int(absolute_error.shape[-2]),
        latitude_descending=data_config.latitude_descending,
        device=absolute_error.device,
        dtype=absolute_error.dtype,
    ).view(1, 1, 1, -1, 1)
    weighted_error = absolute_error * latitude_weights

    groups = _forecast_variable_channel_groups(data_config)
    metric_sums = torch.zeros(len(groups) + 1, device=absolute_error.device, dtype=torch.float64)
    metric_counts = torch.zeros(len(groups) + 1, device=absolute_error.device, dtype=torch.float64)

    metric_sums[0] = weighted_error.sum(dtype=torch.float64)
    metric_counts[0] = float(weighted_error.numel())

    for group_index, (_, channel_indices) in enumerate(groups, start=1):
        group_error = weighted_error[:, :, channel_indices]
        metric_sums[group_index] = group_error.sum(dtype=torch.float64)
        metric_counts[group_index] = float(group_error.numel())

    return metric_sums, metric_counts


def _main_validation_loss_terms(
    prediction: Tensor,
    target: Tensor,
    *,
    criterion: nn.Module,
    data_config: ArcoEra5FuXiDataConfig,
) -> tuple[Tensor, Tensor]:
    prediction = prediction.float()
    target = target.float()
    if prediction.shape != target.shape:
        raise ValueError(
            f"Expected prediction and target to have the same shape, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )

    if isinstance(criterion, LatitudeWeightedCharbonnierLoss):
        latitude_weights = _latitude_weight_vector(
            int(prediction.shape[-2]),
            latitude_descending=criterion.latitude_descending,
            device=prediction.device,
            dtype=prediction.dtype,
        ).view(1, 1, 1, -1, 1)
        channel_weights = criterion.channel_weights.to(device=prediction.device, dtype=prediction.dtype).view(
            1,
            1,
            -1,
            1,
            1,
        )
        weighted_diff = latitude_weights * (prediction - target)
        per_element_loss = channel_weights * torch.sqrt(weighted_diff.square() + criterion.epsilon**2)
    elif isinstance(criterion, nn.MSELoss):
        per_element_loss = (prediction - target).square()
    else:
        raise TypeError(
            f"Unsupported main-validation criterion type {type(criterion).__name__}; "
            "expected LatitudeWeightedCharbonnierLoss or nn.MSELoss."
        )

    groups = _forecast_variable_channel_groups(data_config)
    metric_sums = torch.zeros(len(groups) + 1, device=prediction.device, dtype=torch.float64)
    metric_counts = torch.zeros(len(groups) + 1, device=prediction.device, dtype=torch.float64)

    metric_sums[0] = per_element_loss.sum(dtype=torch.float64)
    metric_counts[0] = float(per_element_loss.numel())

    for group_index, (_, channel_indices) in enumerate(groups, start=1):
        group_loss = per_element_loss[:, :, channel_indices]
        metric_sums[group_index] = group_loss.sum(dtype=torch.float64)
        metric_counts[group_index] = float(group_loss.numel())

    return metric_sums, metric_counts


@dataclass(frozen=True)
class ForecastRolloutPlotGroup:
    variable_name: str
    display_name: str
    channel_indices: tuple[int, ...]
    row_labels: tuple[str, ...]
    is_upper_air: bool


def _forecast_variable_display_name(variable_name: str) -> str:
    display_names = {
        "geopotential": "Geopotential",
        "temperature": "Temperature",
        "u_component_of_wind": "U Wind",
        "v_component_of_wind": "V Wind",
        "relative_humidity": "Relative Humidity",
        "2m_temperature": "2m Temperature",
        "10m_u_component_of_wind": "10m U Wind",
        "10m_v_component_of_wind": "10m V Wind",
        "mean_sea_level_pressure": "Mean Sea Level Pressure",
        "total_precipitation": "Total Precipitation",
    }
    return display_names.get(variable_name, variable_name.replace("_", " ").title())


def _forecast_rollout_plot_groups(
    data_config: ArcoEra5FuXiDataConfig,
) -> list[ForecastRolloutPlotGroup]:
    groups: list[ForecastRolloutPlotGroup] = []
    channel_index = 0
    for variable_name in data_config.upper_air_variables:
        channel_indices = tuple(channel_index + offset for offset in range(len(data_config.pressure_levels)))
        groups.append(
            ForecastRolloutPlotGroup(
                variable_name=variable_name,
                display_name=_forecast_variable_display_name(variable_name),
                channel_indices=channel_indices,
                row_labels=tuple(f"{int(level)} hPa" for level in data_config.pressure_levels),
                is_upper_air=True,
            )
        )
        channel_index += len(data_config.pressure_levels)
    for variable_name in data_config.surface_variables:
        groups.append(
            ForecastRolloutPlotGroup(
                variable_name=variable_name,
                display_name=_forecast_variable_display_name(variable_name),
                channel_indices=(channel_index,),
                row_labels=("surface",),
                is_upper_air=False,
            )
        )
        channel_index += 1
    return groups


def _rollout_filename_slug(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in value)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def _format_rollout_hour_label(hours: int) -> str:
    return f"{hours:+d}h"


def _import_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional plotting dependency
        raise ImportError(
            "Rollout plotting requires matplotlib. Install it in the active environment "
            "or run validation without the rollout plot options."
        ) from exc
    return plt


def _default_rollout_anchor_stride_hours(data_config: ArcoEra5FuXiDataConfig) -> int:
    lead = int(data_config.lead_time_hours)
    input_offsets = tuple(int(offset) for offset in data_config.input_time_offsets_hours)
    expected_offsets = tuple(range(-lead * (len(input_offsets) - 1), lead, lead))
    if input_offsets == expected_offsets and data_config.forecast_steps >= len(input_offsets):
        return lead * len(input_offsets)
    return lead


def _resolve_rollout_anchor_stride_hours(
    data_config: ArcoEra5FuXiDataConfig,
    rollout_anchor_stride_hours: int | None,
) -> int:
    resolved = (
        _default_rollout_anchor_stride_hours(data_config)
        if rollout_anchor_stride_hours is None
        else int(rollout_anchor_stride_hours)
    )
    if resolved <= 0:
        raise ValueError(f"rollout_anchor_stride_hours must be positive, got {resolved}")
    return resolved


def _rollout_channel_mae(
    absolute_error: Tensor,
    *,
    data_config: ArcoEra5FuXiDataConfig,
) -> Tensor:
    latitude_weights = _latitude_weight_vector(
        int(absolute_error.shape[-2]),
        latitude_descending=data_config.latitude_descending,
        device=absolute_error.device,
        dtype=absolute_error.dtype,
    ).view(1, 1, -1, 1)
    return (absolute_error * latitude_weights).mean(dim=(-2, -1))


def _save_rollout_step_map_figures(
    *,
    output_dir: Path,
    plot_groups: Sequence[ForecastRolloutPlotGroup],
    sample_index: int,
    pass_index: int,
    target_step_index: int,
    target_hour_from_initial_anchor: int,
    target_timestamp: pd.Timestamp,
    input_hours_from_initial_anchor: Sequence[int],
    input_sources: Sequence[str],
    truth_inputs: Tensor,
    model_inputs: Tensor,
    truth_target: Tensor,
    prediction: Tensor,
    absolute_error: Tensor,
) -> list[str]:
    plt = _import_pyplot()
    saved_paths: list[str] = []
    show_model_inputs = any(source != "real" for source in input_sources)
    input_column_count = len(input_hours_from_initial_anchor)
    total_columns = input_column_count + (input_column_count if show_model_inputs else 0) + 3

    for plot_group in plot_groups:
        row_count = len(plot_group.channel_indices)
        figure, axes = plt.subplots(
            row_count,
            total_columns,
            figsize=(2.6 * total_columns, 2.0 * row_count + 0.8),
            squeeze=False,
        )
        figure.suptitle(
            f"Sample {sample_index:05d} | {plot_group.display_name} | rollout pass {pass_index + 1} | "
            f"target {_format_rollout_hour_label(target_hour_from_initial_anchor)} | "
            f"{target_timestamp.strftime('%Y-%m-%d %H:%M')}",
            fontsize=12,
        )

        title_columns: list[str] = [
            f"real in {_format_rollout_hour_label(hours)}"
            for hours in input_hours_from_initial_anchor
        ]
        if show_model_inputs:
            title_columns.extend(
                f"model in {_format_rollout_hour_label(hours)} ({source})"
                for hours, source in zip(input_hours_from_initial_anchor, input_sources, strict=False)
            )
        title_columns.extend(
            [
                f"real out {_format_rollout_hour_label(target_hour_from_initial_anchor)}",
                f"pred {_format_rollout_hour_label(target_hour_from_initial_anchor)}",
                f"|err| {_format_rollout_hour_label(target_hour_from_initial_anchor)}",
            ]
        )

        for row_index, channel_index in enumerate(plot_group.channel_indices):
            value_panels = [truth_inputs[input_index, channel_index] for input_index in range(input_column_count)]
            if show_model_inputs:
                value_panels.extend(model_inputs[input_index, channel_index] for input_index in range(input_column_count))
            value_panels.extend([truth_target[channel_index], prediction[channel_index]])
            value_min = min(float(panel.min().item()) for panel in value_panels)
            value_max = max(float(panel.max().item()) for panel in value_panels)
            if np.isclose(value_min, value_max):
                value_max = value_min + 1.0
            error_max = float(absolute_error[channel_index].max().item())
            if error_max <= 0.0:
                error_max = 1.0

            display_panels = [panel.detach().cpu().numpy() for panel in value_panels]
            error_panel = absolute_error[channel_index].detach().cpu().numpy()
            all_panels = display_panels + [error_panel]

            for column_index, panel in enumerate(all_panels):
                axis = axes[row_index, column_index]
                if column_index < len(display_panels):
                    axis.imshow(panel, cmap="viridis", vmin=value_min, vmax=value_max)
                else:
                    axis.imshow(panel, cmap="magma", vmin=0.0, vmax=error_max)
                axis.set_xticks([])
                axis.set_yticks([])
                if row_index == 0:
                    axis.set_title(title_columns[column_index], fontsize=9)
                if column_index == 0:
                    axis.set_ylabel(plot_group.row_labels[row_index], fontsize=9)

        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        figure_path = (
            output_dir
            / (
                f"sample_{sample_index:05d}_pass_{pass_index + 1:03d}_"
                f"target_{_format_rollout_hour_label(target_hour_from_initial_anchor).replace('+', 'p').replace('-', 'm')}_"
                f"{_rollout_filename_slug(plot_group.variable_name)}.png"
            )
        )
        figure.savefig(figure_path, dpi=120)
        plt.close(figure)
        saved_paths.append(str(figure_path))

    return saved_paths


def _save_rollout_error_graphs(
    *,
    output_dir: Path,
    plot_groups: Sequence[ForecastRolloutPlotGroup],
    horizon_labels: Sequence[str],
    channel_mae_history: Tensor,
) -> list[str]:
    plt = _import_pyplot()
    x_positions = np.arange(len(horizon_labels), dtype=np.int64)
    saved_paths: list[str] = []

    for plot_group in plot_groups:
        figure, axis = plt.subplots(figsize=(max(9.0, 0.75 * len(horizon_labels) + 5.0), 5.2))
        for row_label, channel_index in zip(plot_group.row_labels, plot_group.channel_indices, strict=False):
            axis.plot(
                x_positions,
                channel_mae_history[:, channel_index].detach().cpu().numpy(),
                marker="o",
                linewidth=1.5,
                label=row_label,
            )
        axis.set_title(f"Rollout Denormalized Absolute Error | {plot_group.display_name}")
        axis.set_ylabel("Latitude-weighted MAE")
        axis.set_xlabel("Predicted rollout time")
        axis.set_xticks(x_positions)
        axis.set_xticklabels(horizon_labels, rotation=45, ha="right")
        axis.grid(True, alpha=0.3)
        if len(plot_group.channel_indices) > 1:
            axis.legend(ncol=2, fontsize=8)
        figure.tight_layout()
        figure_path = output_dir / f"rollout_error_{_rollout_filename_slug(plot_group.variable_name)}.png"
        figure.savefig(figure_path, dpi=120)
        plt.close(figure)
        saved_paths.append(str(figure_path))

    return saved_paths


def _save_main_rollout_plots(
    model: nn.Module,
    dataset: ArcoEra5FuXiDataset,
    *,
    data_config: ArcoEra5FuXiDataConfig,
    runtime: DistributedRuntime,
    use_amp: bool,
    amp_dtype: torch.dtype | None,
    output_dir: Path,
    rollout_samples: int,
    rollout_passes: int,
    rollout_anchor_stride_hours: int | None,
) -> dict[str, Any]:
    if rollout_samples <= 0:
        raise ValueError(f"rollout_samples must be positive, got {rollout_samples}")
    if rollout_passes <= 0:
        raise ValueError(f"rollout_passes must be positive, got {rollout_passes}")

    from ..models.fuxi_short import build_fuxi_time_embeddings

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_groups = _forecast_rollout_plot_groups(data_config)
    resolved_stride_hours = _resolve_rollout_anchor_stride_hours(data_config, rollout_anchor_stride_hours)
    anchor_stride_steps = dataset._step_count(resolved_stride_hours)
    input_offset_steps = [dataset._step_count(hours) for hours in data_config.input_time_offsets_hours]
    target_offset_steps = [
        dataset._step_count(data_config.lead_time_hours * step)
        for step in range(1, data_config.forecast_steps + 1)
    ]
    dataset_step_hours = dataset._dataset_frequency_hours()
    time_values = dataset._load_time_values()
    valid_anchor_indices = dataset._build_valid_anchor_indices()
    static_features = dataset._build_static_features().to(runtime.device)

    sample_reports: list[dict[str, Any]] = []
    total_time_steps = len(time_values)

    for sample_index in range(min(int(rollout_samples), len(valid_anchor_indices))):
        sample_output_dir = output_dir / f"sample_{sample_index:05d}"
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        initial_anchor_index = int(valid_anchor_indices[sample_index])
        seed_input_indices = [initial_anchor_index + offset for offset in input_offset_steps]
        dataset._ensure_dynamic_chunk(seed_input_indices)
        seed_inputs = torch.from_numpy(dataset._read_dynamic_tensor_from_ring(seed_input_indices)).float()
        frame_store = {
            int(time_index): seed_inputs[position].clone()
            for position, time_index in enumerate(seed_input_indices)
        }
        frame_sources = {int(time_index): "real" for time_index in seed_input_indices}

        horizon_labels: list[str] = []
        channel_mae_history: list[Tensor] = []
        saved_figure_paths: list[str] = []

        for pass_index in range(int(rollout_passes)):
            current_anchor_index = initial_anchor_index + pass_index * anchor_stride_steps
            model_input_indices = [current_anchor_index + offset for offset in input_offset_steps]
            target_indices = [current_anchor_index + offset for offset in target_offset_steps]
            if any(index < 0 or index >= total_time_steps for index in (*model_input_indices, *target_indices)):
                break

            missing_indices = [index for index in model_input_indices if index not in frame_store]
            if missing_indices:
                raise ValueError(
                    "Autoregressive rollout is missing model-input frames at dataset indices "
                    f"{missing_indices}. Try a smaller rollout_anchor_stride_hours value."
                )

            model_inputs_norm = torch.stack([frame_store[index] for index in model_input_indices], dim=0)
            input_sources = [frame_sources[index] for index in model_input_indices]

            dataset._ensure_dynamic_chunk((*model_input_indices, *target_indices))
            truth_inputs_norm = torch.from_numpy(dataset._read_dynamic_tensor_from_ring(model_input_indices)).float()
            truth_targets_norm = torch.from_numpy(dataset._read_dynamic_tensor_from_ring(target_indices)).float()

            anchor_time = pd.Timestamp(time_values[current_anchor_index])
            temb = torch.from_numpy(
                build_fuxi_time_embeddings(anchor_time, total_steps=1, freq_hours=data_config.lead_time_hours)[0, 0]
            ).to(runtime.device)

            with torch.no_grad():
                with _amp_autocast_context(use_amp, runtime.device, amp_dtype):
                    outputs = model(
                        model_inputs_norm.unsqueeze(0).to(runtime.device),
                        temb.unsqueeze(0),
                        static_features=static_features,
                    )
            prediction_norm = outputs["forecast"][0].detach().cpu().float()
            for target_index, predicted_frame in zip(target_indices, prediction_norm, strict=False):
                frame_store[int(target_index)] = predicted_frame.clone()
                frame_sources[int(target_index)] = "pred"

            truth_inputs_denorm = dataset.denormalize_dynamic_tensor(truth_inputs_norm)
            model_inputs_denorm = dataset.denormalize_dynamic_tensor(model_inputs_norm)
            truth_targets_denorm = dataset.denormalize_dynamic_tensor(truth_targets_norm)
            prediction_denorm = dataset.denormalize_dynamic_tensor(prediction_norm)
            absolute_error = (prediction_denorm - truth_targets_denorm).abs()
            channel_mae = _rollout_channel_mae(absolute_error, data_config=data_config)

            input_hours_from_initial_anchor = [
                int((time_index - initial_anchor_index) * dataset_step_hours)
                for time_index in model_input_indices
            ]

            for target_step_index, target_index in enumerate(target_indices):
                target_hour_from_initial_anchor = int((target_index - initial_anchor_index) * dataset_step_hours)
                target_timestamp = pd.Timestamp(time_values[target_index])
                horizon_labels.append(
                    f"{_format_rollout_hour_label(target_hour_from_initial_anchor)}\n"
                    f"{target_timestamp.strftime('%m-%d %H:%M')}"
                )
                channel_mae_history.append(channel_mae[target_step_index].clone())
                saved_figure_paths.extend(
                    _save_rollout_step_map_figures(
                        output_dir=sample_output_dir,
                        plot_groups=plot_groups,
                        sample_index=sample_index,
                        pass_index=pass_index,
                        target_step_index=target_step_index,
                        target_hour_from_initial_anchor=target_hour_from_initial_anchor,
                        target_timestamp=target_timestamp,
                        input_hours_from_initial_anchor=input_hours_from_initial_anchor,
                        input_sources=input_sources,
                        truth_inputs=truth_inputs_denorm,
                        model_inputs=model_inputs_denorm,
                        truth_target=truth_targets_denorm[target_step_index],
                        prediction=prediction_denorm[target_step_index],
                        absolute_error=absolute_error[target_step_index],
                    )
                )

        graph_paths: list[str] = []
        if channel_mae_history:
            graph_paths = _save_rollout_error_graphs(
                output_dir=sample_output_dir,
                plot_groups=plot_groups,
                horizon_labels=horizon_labels,
                channel_mae_history=torch.stack(channel_mae_history, dim=0),
            )

        sample_reports.append(
            {
                "sample_index": sample_index,
                "sample_output_dir": str(sample_output_dir),
                "initial_anchor_time": str(pd.Timestamp(time_values[initial_anchor_index])),
                "rollout_predictions": len(horizon_labels),
                "saved_map_figures": len(saved_figure_paths),
                "saved_error_graphs": len(graph_paths),
            }
        )

    return {
        "output_dir": str(output_dir),
        "rollout_samples": len(sample_reports),
        "rollout_passes": int(rollout_passes),
        "rollout_anchor_stride_hours": int(resolved_stride_hours),
        "samples": sample_reports,
    }


@dataclass(frozen=True)
class MainTrainingConfig:
    batch_size: int = 1
    num_workers: int = 0
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_epochs: int = 1
    device: str = "auto"
    model_dtype: str = "float32"
    use_amp: bool = False
    amp_dtype: str = "float16"
    forecast_loss: str = "charbonnier"
    charbonnier_epsilon: float = 1.0e-3
    upper_air_loss_weight: float = 1.0
    surface_loss_weight: float = 0.1
    gradient_clip_norm: float | None = None
    output_dir: Path = Path("runs/main")
    checkpoint_name: str = "main_last.pt"
    best_checkpoint_name: str = "main_best.pt"
    resume_checkpoint_path: Path | None = None
    save_epoch_checkpoint: bool = True
    save_best_checkpoint: bool = True
    save_every_train_batches: int | None = None
    save_every_optimizer_steps: int | None = None
    train_start_time: pd.Timestamp | None = None
    train_end_time: pd.Timestamp | None = None
    val_start_time: pd.Timestamp | None = None
    val_end_time: pd.Timestamp | None = None
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    log_every: int = 10
    print_model_summary: bool = True
    summary_depth: int = 3
    random_smoke_batch_size: int = 1
    config_path: Path = DEFAULT_MODEL_CONFIG_PATH

    @classmethod
    def from_yaml(cls, config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH) -> "MainTrainingConfig":
        resolved_config_path, data = load_config_section("train_main", config_path)
        return cls(
            batch_size=int(data.get("batch_size", 1)),
            num_workers=int(data.get("num_workers", 0)),
            gradient_accumulation_steps=_to_positive_int(
                data.get("gradient_accumulation_steps"),
                default=1,
                field_name="train_main.gradient_accumulation_steps",
            ),
            learning_rate=float(data.get("learning_rate", 1e-4)),
            weight_decay=float(data.get("weight_decay", 0.0)),
            max_epochs=int(data.get("max_epochs", 1)),
            device=str(data.get("device", "auto")),
            model_dtype=str(data.get("model_dtype", "float32")),
            use_amp=bool(data.get("use_amp", False)),
            amp_dtype=str(data.get("amp_dtype", "float16")),
            forecast_loss=str(data.get("forecast_loss", "charbonnier")),
            charbonnier_epsilon=float(data.get("charbonnier_epsilon", 1.0e-3)),
            upper_air_loss_weight=float(data.get("upper_air_loss_weight", 1.0)),
            surface_loss_weight=float(data.get("surface_loss_weight", 0.1)),
            gradient_clip_norm=_to_optional_float(data.get("gradient_clip_norm")),
            output_dir=resolve_repo_path(data.get("output_dir", "runs/main"), config_path=resolved_config_path),
            checkpoint_name=str(data.get("checkpoint_name", "main_last.pt")),
            best_checkpoint_name=str(data.get("best_checkpoint_name", "main_best.pt")),
            resume_checkpoint_path=(
                None
                if data.get("resume_checkpoint_path") in {None, ""}
                else resolve_repo_path(data.get("resume_checkpoint_path"), config_path=resolved_config_path)
            ),
            save_epoch_checkpoint=bool(data.get("save_epoch_checkpoint", True)),
            save_best_checkpoint=bool(data.get("save_best_checkpoint", True)),
            save_every_train_batches=_to_optional_positive_int(
                data.get("save_every_train_batches"),
                field_name="train_main.save_every_train_batches",
            ),
            save_every_optimizer_steps=_to_optional_positive_int(
                data.get("save_every_optimizer_steps"),
                field_name="train_main.save_every_optimizer_steps",
            ),
            train_start_time=_to_optional_timestamp(data.get("train_start_time")),
            train_end_time=_to_optional_timestamp(data.get("train_end_time")),
            val_start_time=_to_optional_timestamp(data.get("val_start_time")),
            val_end_time=_to_optional_timestamp(data.get("val_end_time")),
            max_train_batches=_to_optional_int(data.get("max_train_batches")),
            max_val_batches=_to_optional_int(data.get("max_val_batches")),
            log_every=int(data.get("log_every", 10)),
            print_model_summary=bool(data.get("print_model_summary", True)),
            summary_depth=int(data.get("summary_depth", 3)),
            random_smoke_batch_size=int(data.get("random_smoke_batch_size", 1)),
            config_path=resolved_config_path,
        )


@dataclass(frozen=True)
class IntrinsicTrainingConfig:
    batch_size: int = 1
    num_workers: int = 0
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_epochs: int = 1
    device: str = "auto"
    model_dtype: str = "float32"
    use_amp: bool = False
    amp_dtype: str = "float16"
    gradient_clip_norm: float | None = None
    output_dir: Path = Path("runs/intrinsic")
    checkpoint_name: str = "intrinsic_last.pt"
    best_checkpoint_name: str = "intrinsic_best.pt"
    resume_checkpoint_path: Path | None = None
    save_epoch_checkpoint: bool = True
    save_best_checkpoint: bool = True
    save_every_train_batches: int | None = None
    save_every_optimizer_steps: int | None = None
    main_checkpoint_path: Path | None = None
    detach_second_block_features: bool = False
    train_start_time: pd.Timestamp | None = None
    train_end_time: pd.Timestamp | None = None
    val_start_time: pd.Timestamp | None = None
    val_end_time: pd.Timestamp | None = None
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    log_every: int = 10
    print_model_summary: bool = True
    summary_depth: int = 3
    random_smoke_batch_size: int = 1
    config_path: Path = DEFAULT_MODEL_CONFIG_PATH

    @classmethod
    def from_yaml(cls, config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH) -> "IntrinsicTrainingConfig":
        resolved_config_path, data = load_config_section("train_intrinsic", config_path)
        checkpoint_value = data.get("main_checkpoint_path")
        checkpoint_path = None
        if checkpoint_value not in {None, ""}:
            checkpoint_path = resolve_repo_path(checkpoint_value, config_path=resolved_config_path)
        return cls(
            batch_size=int(data.get("batch_size", 1)),
            num_workers=int(data.get("num_workers", 0)),
            gradient_accumulation_steps=_to_positive_int(
                data.get("gradient_accumulation_steps"),
                default=1,
                field_name="train_intrinsic.gradient_accumulation_steps",
            ),
            learning_rate=float(data.get("learning_rate", 1e-4)),
            weight_decay=float(data.get("weight_decay", 0.0)),
            max_epochs=int(data.get("max_epochs", 1)),
            device=str(data.get("device", "auto")),
            model_dtype=str(data.get("model_dtype", "float32")),
            use_amp=bool(data.get("use_amp", False)),
            amp_dtype=str(data.get("amp_dtype", "float16")),
            gradient_clip_norm=_to_optional_float(data.get("gradient_clip_norm")),
            output_dir=resolve_repo_path(data.get("output_dir", "runs/intrinsic"), config_path=resolved_config_path),
            checkpoint_name=str(data.get("checkpoint_name", "intrinsic_last.pt")),
            best_checkpoint_name=str(data.get("best_checkpoint_name", "intrinsic_best.pt")),
            resume_checkpoint_path=(
                None
                if data.get("resume_checkpoint_path") in {None, ""}
                else resolve_repo_path(data.get("resume_checkpoint_path"), config_path=resolved_config_path)
            ),
            save_epoch_checkpoint=bool(data.get("save_epoch_checkpoint", True)),
            save_best_checkpoint=bool(data.get("save_best_checkpoint", True)),
            save_every_train_batches=_to_optional_positive_int(
                data.get("save_every_train_batches"),
                field_name="train_intrinsic.save_every_train_batches",
            ),
            save_every_optimizer_steps=_to_optional_positive_int(
                data.get("save_every_optimizer_steps"),
                field_name="train_intrinsic.save_every_optimizer_steps",
            ),
            main_checkpoint_path=checkpoint_path,
            detach_second_block_features=bool(
                data.get(
                    "detach_second_block_features",
                    data.get("detach_z_high", False),
                )
            ),
            train_start_time=_to_optional_timestamp(data.get("train_start_time")),
            train_end_time=_to_optional_timestamp(data.get("train_end_time")),
            val_start_time=_to_optional_timestamp(data.get("val_start_time")),
            val_end_time=_to_optional_timestamp(data.get("val_end_time")),
            max_train_batches=_to_optional_int(data.get("max_train_batches")),
            max_val_batches=_to_optional_int(data.get("max_val_batches")),
            log_every=int(data.get("log_every", 10)),
            print_model_summary=bool(data.get("print_model_summary", True)),
            summary_depth=int(data.get("summary_depth", 3)),
            random_smoke_batch_size=int(data.get("random_smoke_batch_size", 1)),
            config_path=resolved_config_path,
        )


def _build_main_training_objects(
    config_path: str | Path,
) -> tuple[
    MainTrainingConfig,
    FuXiLowerResConfig,
    ArcoEra5FuXiDataConfig,
    DistributedRuntime,
    torch.dtype,
    torch.dtype | None,
]:
    train_config = MainTrainingConfig.from_yaml(config_path)
    runtime = _resolve_distributed_runtime(train_config.device)
    model_dtype = resolve_torch_dtype(train_config.model_dtype) or torch.float32
    amp_dtype = resolve_torch_dtype(train_config.amp_dtype)
    _validate_training_precision_config(
        section_name="train_main",
        use_amp=train_config.use_amp,
        model_dtype=model_dtype,
        amp_dtype=amp_dtype,
    )

    model_config = replace(
        FuXiLowerResConfig.from_yaml(config_path),
        device=runtime.device,
        dtype=model_dtype,
    )
    data_config = replace(
        ArcoEra5FuXiDataConfig.from_yaml(config_path),
        forecast_steps=model_config.forecast_steps,
    )
    return train_config, model_config, data_config, runtime, model_dtype, amp_dtype


def _build_intrinsic_training_objects(
    config_path: str | Path,
) -> tuple[
    IntrinsicTrainingConfig,
    FuXiLowerResConfig,
    FuXiIntrinsicConfig,
    ArcoEra5FuXiDataConfig,
    DistributedRuntime,
    torch.dtype,
    torch.dtype | None,
]:
    train_config = IntrinsicTrainingConfig.from_yaml(config_path)
    runtime = _resolve_distributed_runtime(train_config.device)
    model_dtype = resolve_torch_dtype(train_config.model_dtype) or torch.float32
    amp_dtype = resolve_torch_dtype(train_config.amp_dtype)
    _validate_training_precision_config(
        section_name="train_intrinsic",
        use_amp=train_config.use_amp,
        model_dtype=model_dtype,
        amp_dtype=amp_dtype,
    )

    encoder_config = replace(
        FuXiLowerResConfig.from_yaml(config_path),
        device=runtime.device,
        dtype=model_dtype,
    )
    intrinsic_config = replace(
        FuXiIntrinsicConfig.from_yaml(config_path),
        device=runtime.device,
        dtype=model_dtype,
        feature_channels=encoder_config.embed_dim,
        spatial_size=encoder_config.patch_grid,
    )
    data_config = replace(
        ArcoEra5FuXiDataConfig.from_yaml(config_path),
        forecast_steps=encoder_config.forecast_steps,
    )
    return train_config, encoder_config, intrinsic_config, data_config, runtime, model_dtype, amp_dtype


def _build_split_dataloaders(
    data_config: ArcoEra5FuXiDataConfig,
    *,
    batch_size: int,
    num_workers: int,
    runtime: DistributedRuntime,
    train_start_time: pd.Timestamp | None,
    train_end_time: pd.Timestamp | None,
    val_start_time: pd.Timestamp | None,
    val_end_time: pd.Timestamp | None,
    pin_memory: bool,
) -> tuple[
    DataLoader[dict[str, Any]],
    DataLoader[dict[str, Any]] | None,
    Sampler[Any] | None,
    Sampler[Any] | None,
]:
    train_loader, train_sampler = _build_eval_dataloader(
        data_config,
        batch_size=batch_size,
        num_workers=num_workers,
        runtime=runtime,
        start_time=train_start_time,
        end_time=train_end_time,
        pin_memory=pin_memory,
    )

    if val_start_time is None and val_end_time is None:
        return train_loader, None, train_sampler, None

    val_loader, val_sampler = _build_eval_dataloader(
        data_config,
        batch_size=batch_size,
        num_workers=num_workers,
        runtime=runtime,
        start_time=val_start_time,
        end_time=val_end_time,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, train_sampler, val_sampler


def _build_eval_dataloader(
    data_config: ArcoEra5FuXiDataConfig,
    *,
    batch_size: int,
    num_workers: int,
    runtime: DistributedRuntime,
    start_time: pd.Timestamp | None,
    end_time: pd.Timestamp | None,
    pin_memory: bool,
) -> tuple[DataLoader[dict[str, Any]], Sampler[Any] | None]:
    dataset = ArcoEra5FuXiDataset(
        replace(
            data_config,
            start_time=start_time,
            end_time=end_time,
        )
    )
    if data_config.apply_normalization:
        if runtime.is_primary:
            dataset.ensure_normalization_stats()
        if runtime.enabled:
            dist.barrier()
    sampler: Sampler[Any] | None = None
    if runtime.enabled:
        sampler = ContiguousDistributedSampler(
            dataset,
            num_replicas=runtime.world_size,
            rank=runtime.rank,
        )
    loader = build_arco_era5_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=sampler,
    )
    return loader, sampler


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _step_checkpoint_path(output_dir: Path, checkpoint_name: str, optimizer_step: int) -> Path:
    template = Path(checkpoint_name)
    suffix = template.suffix or ".pt"
    return output_dir / f"{template.stem}_step_{optimizer_step:08d}{suffix}"


def _load_encoder_checkpoint(encoder: FuXiLowerResEncoder, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "encoder_state_dict" in checkpoint:
        state_dict = checkpoint["encoder_state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = {
            key[len("encoder.") :]: value
            for key, value in checkpoint["model_state_dict"].items()
            if key.startswith("encoder.")
        }
    elif isinstance(checkpoint, dict) and all(isinstance(value, Tensor) for value in checkpoint.values()):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")
    encoder.load_state_dict(state_dict, strict=True)


def _load_main_forecast_checkpoint(model: FuXiLowerRes, checkpoint_path: Path) -> dict[str, Any] | None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict: dict[str, Tensor]
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and {
        "encoder_state_dict",
        "decoder_state_dict",
    }.issubset(checkpoint):
        state_dict = {
            **{f"encoder.{key}": value for key, value in checkpoint["encoder_state_dict"].items()},
            **{f"decoder.{key}": value for key, value in checkpoint["decoder_state_dict"].items()},
        }
    elif isinstance(checkpoint, dict) and all(isinstance(value, Tensor) for value in checkpoint.values()):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")
    model.load_state_dict(state_dict, strict=True)
    return checkpoint if isinstance(checkpoint, dict) else None


def _load_intrinsic_checkpoint(model: FuXiIntrinsic, checkpoint_path: Path) -> dict[str, Any] | None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict: dict[str, Tensor]
    if isinstance(checkpoint, dict) and "intrinsic_state_dict" in checkpoint:
        state_dict = checkpoint["intrinsic_state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and all(isinstance(value, Tensor) for value in checkpoint.values()):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported intrinsic checkpoint format at {checkpoint_path}")
    model.load_state_dict(state_dict, strict=True)
    return checkpoint if isinstance(checkpoint, dict) else None


def _iter_limited(loader: Iterable[dict[str, Any]], max_batches: int | None) -> Iterable[tuple[int, dict[str, Any]]]:
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        yield batch_index, batch


def _record_batch_stream(batch: dict[str, Any], stream: torch.cuda.Stream) -> None:
    for value in batch.values():
        if isinstance(value, Tensor):
            value.record_stream(stream)


def _iter_prefetched_batches(
    loader: DataLoader[dict[str, Any]],
    *,
    runtime: DistributedRuntime,
    max_batches: int | None,
) -> Iterable[tuple[int, dict[str, Any]]]:
    if runtime.device.type != "cuda":
        yield from _iter_limited(
            (_move_batch_to_device(batch, runtime.device) for batch in loader),
            max_batches,
        )
        return

    iterator = iter(loader)
    transfer_stream = torch.cuda.Stream(device=runtime.device)
    next_batch: dict[str, Any] | None = None

    def preload_next() -> None:
        nonlocal next_batch
        try:
            cpu_batch = next(iterator)
        except StopIteration:
            next_batch = None
            return

        with torch.cuda.stream(transfer_stream):
            next_batch = _move_batch_to_device(cpu_batch, runtime.device, non_blocking=True)

    preload_next()
    batch_index = 0
    while next_batch is not None:
        if max_batches is not None and batch_index >= max_batches:
            break
        current_stream = torch.cuda.current_stream(device=runtime.device)
        current_stream.wait_stream(transfer_stream)
        batch = next_batch
        _record_batch_stream(batch, current_stream)
        preload_next()
        yield batch_index, batch
        batch_index += 1


@dataclass(frozen=True)
class TrainingResumeState:
    checkpoint_path: Path
    checkpoint_epoch: int
    start_epoch: int
    resume_epoch: int | None
    resume_batch_index: int
    replay_start_batch_index: int
    optimizer_steps: int
    global_batch_steps: int
    best_val_loss: float
    history: list[dict[str, float]]


def _checkpoint_history(checkpoint: dict[str, Any] | None) -> list[dict[str, float]]:
    if checkpoint is None:
        return []
    history = checkpoint.get("history")
    if not isinstance(history, list):
        return []
    return [entry for entry in history if isinstance(entry, dict)]


def _best_val_loss_from_history(history: list[dict[str, float]]) -> float:
    best = float("inf")
    for entry in history:
        value = entry.get("val_loss")
        if value is not None:
            best = min(best, float(value))
    return best


def _last_completed_optimizer_batch_count(
    total_train_batches: int,
    processed_batch_count: int,
    accumulation_steps: int,
) -> int:
    last_completed = 0
    for batch_index in range(processed_batch_count):
        if _should_optimizer_step(total_train_batches, batch_index, accumulation_steps):
            last_completed = batch_index + 1
    return last_completed


def _checkpoint_global_batch_steps(
    checkpoint: dict[str, Any] | None,
    *,
    history: list[dict[str, float]],
    checkpoint_epoch: int,
    processed_batch_count: int | None,
) -> int:
    if checkpoint is not None and checkpoint.get("global_batch_step") is not None:
        return int(checkpoint["global_batch_step"])

    history_global = 0
    if history:
        history_global = int(history[-1].get("global_batch_steps", 0.0))
    if processed_batch_count is not None and checkpoint_epoch > len(history):
        return history_global + int(processed_batch_count)
    return history_global


def _build_resume_state(
    checkpoint: dict[str, Any],
    *,
    checkpoint_path: Path,
    total_train_batches: int,
    accumulation_steps: int,
) -> TrainingResumeState:
    history = _checkpoint_history(checkpoint)
    checkpoint_epoch = int(checkpoint.get("epoch", 0))
    processed_batch_count_raw = checkpoint.get("batch_index_within_epoch")
    processed_batch_count = None if processed_batch_count_raw in {None, ""} else int(processed_batch_count_raw)
    if processed_batch_count is not None and processed_batch_count < 0:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has invalid batch_index_within_epoch={processed_batch_count}"
        )
    if processed_batch_count is not None and total_train_batches > 0 and processed_batch_count > total_train_batches:
        raise ValueError(
            f"Checkpoint {checkpoint_path} was saved after batch {processed_batch_count}, "
            f"but the current training run only has {total_train_batches} batches per epoch. "
            "Keep batch_size/max_train_batches consistent when resuming."
        )

    optimizer_steps = int(
        checkpoint.get(
            "optimizer_step",
            history[-1].get("optimizer_steps", 0.0) if history else 0.0,
        )
    )
    global_batch_steps = _checkpoint_global_batch_steps(
        checkpoint,
        history=history,
        checkpoint_epoch=checkpoint_epoch,
        processed_batch_count=processed_batch_count,
    )
    history_best_val_loss = _best_val_loss_from_history(history)
    stored_best_val_loss_raw = checkpoint.get("best_val_loss")
    if stored_best_val_loss_raw is None:
        best_val_loss = float(history_best_val_loss)
    else:
        best_val_loss = float(stored_best_val_loss_raw)
        if history_best_val_loss < float("inf"):
            best_val_loss = min(best_val_loss, float(history_best_val_loss))

    if processed_batch_count is None or total_train_batches <= 0:
        start_epoch = max(checkpoint_epoch + 1, 1)
        return TrainingResumeState(
            checkpoint_path=checkpoint_path,
            checkpoint_epoch=checkpoint_epoch,
            start_epoch=start_epoch,
            resume_epoch=None,
            resume_batch_index=0,
            replay_start_batch_index=0,
            optimizer_steps=optimizer_steps,
            global_batch_steps=global_batch_steps,
            best_val_loss=best_val_loss,
            history=history,
        )

    if processed_batch_count >= total_train_batches:
        start_epoch = max(checkpoint_epoch + 1, 1)
        return TrainingResumeState(
            checkpoint_path=checkpoint_path,
            checkpoint_epoch=checkpoint_epoch,
            start_epoch=start_epoch,
            resume_epoch=None,
            resume_batch_index=0,
            replay_start_batch_index=0,
            optimizer_steps=optimizer_steps,
            global_batch_steps=global_batch_steps,
            best_val_loss=best_val_loss,
            history=history,
        )

    replay_start_batch_index = _last_completed_optimizer_batch_count(
        total_train_batches,
        processed_batch_count,
        accumulation_steps,
    )
    return TrainingResumeState(
        checkpoint_path=checkpoint_path,
        checkpoint_epoch=max(checkpoint_epoch, 1),
        start_epoch=max(checkpoint_epoch, 1),
        resume_epoch=max(checkpoint_epoch, 1),
        resume_batch_index=processed_batch_count,
        replay_start_batch_index=replay_start_batch_index,
        optimizer_steps=optimizer_steps,
        global_batch_steps=global_batch_steps,
        best_val_loss=best_val_loss,
        history=history,
    )


def _validate_resume_compatibility(
    checkpoint: dict[str, Any] | None,
    *,
    checkpoint_path: Path,
    section_name: str,
    current_batch_size: int,
    current_accumulation_steps: int,
    resume_state: TrainingResumeState | None,
) -> list[str]:
    warnings: list[str] = []
    if checkpoint is None:
        return warnings
    saved_train_config = checkpoint.get("train_config")
    if not isinstance(saved_train_config, dict):
        return warnings
    is_mid_epoch_resume = resume_state is not None and resume_state.resume_epoch is not None
    saved_batch_size = saved_train_config.get("batch_size")
    if saved_batch_size is not None and int(saved_batch_size) != int(current_batch_size):
        message = (
            f"{section_name} resume checkpoint {checkpoint_path} was created with batch_size={saved_batch_size}, "
            f"but the current config uses batch_size={current_batch_size}."
        )
        if is_mid_epoch_resume:
            raise ValueError(
                f"{message} Keep batch_size consistent when resuming from a mid-epoch checkpoint."
            )
        warnings.append(
            f"{message} Resuming from an epoch-complete checkpoint is allowed; training will continue "
            "with the new batch size from the next epoch onward."
        )
    saved_accumulation = saved_train_config.get("gradient_accumulation_steps")
    if saved_accumulation is not None and int(saved_accumulation) != int(current_accumulation_steps):
        message = (
            f"{section_name} resume checkpoint {checkpoint_path} was created with "
            f"gradient_accumulation_steps={saved_accumulation}, but the current config uses "
            f"gradient_accumulation_steps={current_accumulation_steps}."
        )
        if is_mid_epoch_resume:
            raise ValueError(
                f"{message} Keep gradient accumulation consistent when resuming from a mid-epoch checkpoint."
            )
        warnings.append(
            f"{message} Resuming from an epoch-complete checkpoint is allowed; the effective batch size "
            "changes from the next epoch onward."
        )
    return warnings


def _apply_optimizer_hyperparameter_overrides(
    optimizer: Optimizer,
    *,
    learning_rate: float,
    weight_decay: float,
) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = float(learning_rate)
        param_group["weight_decay"] = float(weight_decay)
        if "initial_lr" in param_group:
            param_group["initial_lr"] = float(learning_rate)


def _resume_optimizer_override_messages(
    checkpoint: dict[str, Any] | None,
    *,
    learning_rate: float,
    weight_decay: float,
    section_name: str,
) -> list[str]:
    messages: list[str] = []
    if checkpoint is None:
        return messages
    saved_train_config = checkpoint.get("train_config")
    if not isinstance(saved_train_config, dict):
        return messages

    saved_learning_rate = saved_train_config.get("learning_rate")
    if saved_learning_rate is not None and float(saved_learning_rate) != float(learning_rate):
        messages.append(
            f"{section_name} resume is overriding checkpoint learning_rate={saved_learning_rate} "
            f"with config learning_rate={learning_rate}."
        )

    saved_weight_decay = saved_train_config.get("weight_decay")
    if saved_weight_decay is not None and float(saved_weight_decay) != float(weight_decay):
        messages.append(
            f"{section_name} resume is overriding checkpoint weight_decay={saved_weight_decay} "
            f"with config weight_decay={weight_decay}."
        )
    return messages


def _evaluate_main_forecast_model(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    *,
    criterion: nn.Module,
    data_config: ArcoEra5FuXiDataConfig,
    runtime: DistributedRuntime,
    use_amp: bool,
    amp_dtype: torch.dtype | None,
    max_batches: int | None,
) -> dict[str, Any]:
    model.eval()
    val_running_loss = 0.0
    val_steps = 0
    variable_group_names = [name for name, _ in _forecast_variable_channel_groups(data_config)]
    loss_metric_sums = torch.zeros(
        len(variable_group_names) + 1,
        device=runtime.device,
        dtype=torch.float64,
    )
    loss_metric_counts = torch.zeros_like(loss_metric_sums)
    denorm_metric_sums = torch.zeros_like(loss_metric_sums)
    denorm_metric_counts = torch.zeros_like(loss_metric_sums)
    dataset_for_metrics = loader.dataset
    if not hasattr(dataset_for_metrics, "denormalize_dynamic_tensor"):
        raise TypeError(
            "Expected the evaluation loader dataset to expose denormalize_dynamic_tensor(...) "
            "for denormalized forecast metrics."
        )

    with torch.no_grad():
        for _batch_index, batch in _iter_prefetched_batches(
            loader,
            runtime=runtime,
            max_batches=max_batches,
        ):
            with _amp_autocast_context(use_amp, runtime.device, amp_dtype):
                outputs = model(
                    batch["x"],
                    batch["temb"],
                    static_features=batch["static_features"],
                )
                loss = criterion(outputs["forecast"], batch["target"])
            val_running_loss += float(loss.item())
            val_steps += 1

            batch_loss_sums, batch_loss_counts = _main_validation_loss_terms(
                outputs["forecast"],
                batch["target"],
                criterion=criterion,
                data_config=data_config,
            )
            batch_denorm_sums, batch_denorm_counts = _main_validation_denormalized_mae(
                outputs["forecast"],
                batch["target"],
                dataset=dataset_for_metrics,
                data_config=data_config,
            )
            loss_metric_sums += batch_loss_sums
            loss_metric_counts += batch_loss_counts
            denorm_metric_sums += batch_denorm_sums
            denorm_metric_counts += batch_denorm_counts

    val_loss_sum, global_val_steps = _reduced_sum_and_count(val_running_loss, val_steps, runtime)
    _reduce_tensor_in_place(loss_metric_sums, runtime)
    _reduce_tensor_in_place(loss_metric_counts, runtime)
    _reduce_tensor_in_place(denorm_metric_sums, runtime)
    _reduce_tensor_in_place(denorm_metric_counts, runtime)

    loss_means = loss_metric_sums / loss_metric_counts.clamp_min(1.0)
    denorm_means = denorm_metric_sums / denorm_metric_counts.clamp_min(1.0)
    return {
        "loss": val_loss_sum / max(global_val_steps, 1),
        "variable_losses": {
            variable_name: float(loss_means[group_index].item())
            for group_index, variable_name in enumerate(variable_group_names, start=1)
        },
        "denorm_mae": float(denorm_means[0].item()),
        "variable_denorm_mae": {
            variable_name: float(denorm_means[group_index].item())
            for group_index, variable_name in enumerate(variable_group_names, start=1)
        },
        "batches": global_val_steps,
    }


def _accumulation_divisor(total_batches: int, batch_index: int, accumulation_steps: int) -> int:
    if total_batches <= 0:
        return accumulation_steps

    remainder = total_batches % accumulation_steps
    if remainder == 0:
        return accumulation_steps

    first_final_group_index = total_batches - remainder
    if batch_index >= first_final_group_index:
        return remainder
    return accumulation_steps


def _should_optimizer_step(total_batches: int, batch_index: int, accumulation_steps: int) -> bool:
    batch_number = batch_index + 1
    return batch_number % accumulation_steps == 0 or batch_number == total_batches


def train_main_model(
    config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
    *,
    smoke_only: bool = False,
    resume_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    train_config, model_config, data_config, runtime, _model_dtype, amp_dtype = _build_main_training_objects(config_path)
    try:
        model = FuXiLowerRes(model_config).to(runtime.device)
        resolved_resume_checkpoint_path = (
            train_config.resume_checkpoint_path
            if resume_checkpoint_path is None
            else resolve_repo_path(resume_checkpoint_path, config_path=train_config.config_path)
        )
        resume_checkpoint: dict[str, Any] | None = None
        smoke_report: dict[str, Any] | None = None

        if runtime.is_primary:
            smoke_inputs = _make_main_random_inputs(
                model_config,
                batch_size=train_config.random_smoke_batch_size,
                device=runtime.device,
            )
            _print_model_and_summary(
                "Forecast Model",
                model,
                input_data=smoke_inputs,
                depth=train_config.summary_depth,
                print_summary=train_config.print_model_summary,
            )
            smoke_report = run_main_model_smoke_test(
                model,
                batch_size=train_config.random_smoke_batch_size,
                print_outputs=True,
            )
        if smoke_only:
            return {"smoke_only": True, "smoke_report": smoke_report}
        if resolved_resume_checkpoint_path is not None:
            if not resolved_resume_checkpoint_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found at {resolved_resume_checkpoint_path}")
            resume_checkpoint = _load_main_forecast_checkpoint(model, resolved_resume_checkpoint_path)

        model = _wrap_for_distributed_training(model, runtime)
        pin_memory = runtime.device.type == "cuda"
        train_loader, val_loader, train_sampler, val_sampler = _build_split_dataloaders(
            data_config,
            batch_size=train_config.batch_size,
            num_workers=train_config.num_workers,
            runtime=runtime,
            train_start_time=train_config.train_start_time,
            train_end_time=train_config.train_end_time,
            val_start_time=train_config.val_start_time,
            val_end_time=train_config.val_end_time,
            pin_memory=pin_memory,
        )

        optimizer = AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
        scaler = _build_grad_scaler(train_config.use_amp, runtime.device)
        criterion = _build_main_forecast_criterion(train_config, data_config).to(runtime.device)

        output_dir = train_config.output_dir
        if runtime.is_primary:
            output_dir.mkdir(parents=True, exist_ok=True)

        effective_batch_size = (
            train_config.batch_size * runtime.world_size * train_config.gradient_accumulation_steps
        )
        _print_if_primary(
            runtime,
            (
                f"[main] device={runtime.device} world_size={runtime.world_size} "
                f"gradient_accumulation_steps={train_config.gradient_accumulation_steps} "
                f"effective_batch_size={effective_batch_size}"
            ),
        )

        total_train_batches = _limited_length(train_loader, train_config.max_train_batches)
        total_val_batches = 0 if val_loader is None else _limited_length(val_loader, train_config.max_val_batches)
        resume_state = (
            None
            if resume_checkpoint is None
            else _build_resume_state(
                resume_checkpoint,
                checkpoint_path=resolved_resume_checkpoint_path,
                total_train_batches=total_train_batches,
                accumulation_steps=train_config.gradient_accumulation_steps,
            )
        )
        if resume_checkpoint is not None:
            resume_warnings = _validate_resume_compatibility(
                resume_checkpoint,
                checkpoint_path=resolved_resume_checkpoint_path,
                section_name="train_main",
                current_batch_size=train_config.batch_size,
                current_accumulation_steps=train_config.gradient_accumulation_steps,
                resume_state=resume_state,
            )
            if "optimizer_state_dict" not in resume_checkpoint:
                raise ValueError(
                    f"Resume checkpoint {resolved_resume_checkpoint_path} does not contain optimizer_state_dict."
                )
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            _apply_optimizer_hyperparameter_overrides(
                optimizer,
                learning_rate=train_config.learning_rate,
                weight_decay=train_config.weight_decay,
            )
            if "scaler_state_dict" in resume_checkpoint and scaler.is_enabled():
                scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
            for message in resume_warnings + _resume_optimizer_override_messages(
                resume_checkpoint,
                learning_rate=train_config.learning_rate,
                weight_decay=train_config.weight_decay,
                section_name="train_main",
            ):
                _print_if_primary(runtime, f"[main][resume] {message}")
        history: list[dict[str, float]] = [] if resume_state is None else list(resume_state.history)
        best_val_loss = float("inf") if resume_state is None else resume_state.best_val_loss
        optimizer_steps = 0 if resume_state is None else resume_state.optimizer_steps
        global_batch_steps = 0 if resume_state is None else resume_state.global_batch_steps
        start_epoch = 1 if resume_state is None else resume_state.start_epoch
        if resume_state is not None:
            _print_if_primary(
                runtime,
                f"[main] resuming from {resume_state.checkpoint_path} "
                f"checkpoint_epoch={resume_state.checkpoint_epoch} start_epoch={resume_state.start_epoch} "
                f"resume_batch={resume_state.resume_batch_index} optimizer_steps={resume_state.optimizer_steps} "
                f"global_batch_steps={resume_state.global_batch_steps}",
            )
            if start_epoch > train_config.max_epochs:
                raise ValueError(
                    f"Resume checkpoint {resume_state.checkpoint_path} would continue at epoch {start_epoch}, "
                    f"but train_main.max_epochs={train_config.max_epochs}. Increase max_epochs to continue training."
                )

        for epoch in range(start_epoch, train_config.max_epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)

            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            train_steps = 0
            resume_batch_index = (
                0
                if resume_state is None or resume_state.resume_epoch != epoch
                else resume_state.resume_batch_index
            )
            replay_start_batch_index = (
                0
                if resume_state is None or resume_state.resume_epoch != epoch
                else resume_state.replay_start_batch_index
            )
            if resume_batch_index > 0:
                replay_from = replay_start_batch_index + 1
                replay_to = resume_batch_index
                if replay_start_batch_index < resume_batch_index:
                    _print_if_primary(
                        runtime,
                        f"[main][epoch {epoch}] rebuilding accumulated gradients from batches "
                        f"{replay_from}-{replay_to} before resuming at batch {resume_batch_index + 1}",
                    )
                else:
                    _print_if_primary(
                        runtime,
                        f"[main][epoch {epoch}] skipping directly to batch {resume_batch_index + 1}",
                    )

            for batch_index, batch in _iter_prefetched_batches(
                train_loader,
                runtime=runtime,
                max_batches=train_config.max_train_batches,
            ):
                if batch_index < resume_batch_index:
                    if batch_index < replay_start_batch_index:
                        continue
                    should_step = _should_optimizer_step(
                        total_train_batches,
                        batch_index,
                        train_config.gradient_accumulation_steps,
                    )
                    if should_step:
                        raise RuntimeError(
                            "Resume replay encountered an optimizer-step batch. "
                            "This indicates the checkpoint resume window was computed incorrectly."
                        )
                    loss_divisor = float(
                        _accumulation_divisor(
                            total_train_batches,
                            batch_index,
                            train_config.gradient_accumulation_steps,
                        )
                    )
                    sync_context = nullcontext()
                    if runtime.enabled and isinstance(model, DistributedDataParallel):
                        sync_context = model.no_sync()
                    with sync_context:
                        with _amp_autocast_context(train_config.use_amp, runtime.device, amp_dtype):
                            outputs = model(
                                batch["x"],
                                batch["temb"],
                                static_features=batch["static_features"],
                            )
                            scaled_loss = criterion(outputs["forecast"], batch["target"]) / loss_divisor
                        if scaler.is_enabled():
                            scaler.scale(scaled_loss).backward()
                        else:
                            scaled_loss.backward()
                    continue

                should_step = _should_optimizer_step(
                    total_train_batches,
                    batch_index,
                    train_config.gradient_accumulation_steps,
                )
                global_batch_steps += 1
                loss_divisor = float(
                    _accumulation_divisor(
                        total_train_batches,
                        batch_index,
                        train_config.gradient_accumulation_steps,
                    )
                )
                sync_context = nullcontext()
                if runtime.enabled and isinstance(model, DistributedDataParallel) and not should_step:
                    sync_context = model.no_sync()

                with sync_context:
                    with _amp_autocast_context(train_config.use_amp, runtime.device, amp_dtype):
                        outputs = model(
                            batch["x"],
                            batch["temb"],
                            static_features=batch["static_features"],
                        )
                        loss = criterion(outputs["forecast"], batch["target"])
                        scaled_loss = loss / loss_divisor

                    if scaler.is_enabled():
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                if should_step:
                    if train_config.gradient_clip_norm is not None:
                        if scaler.is_enabled():
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            _unwrap_model(model).parameters(),
                            train_config.gradient_clip_norm,
                        )

                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1

                    if (
                        runtime.is_primary
                        and train_config.save_every_optimizer_steps is not None
                        and int(optimizer_steps) % train_config.save_every_optimizer_steps == 0
                    ):
                        base_model = _unwrap_model(model)
                        step_checkpoint_payload = {
                            "epoch": epoch,
                            "optimizer_step": int(optimizer_steps),
                            "batch_index_within_epoch": batch_index + 1,
                            "model_state_dict": base_model.state_dict(),
                            "encoder_state_dict": base_model.encoder.state_dict(),
                            "decoder_state_dict": base_model.decoder.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scaler_state_dict": scaler.state_dict(),
                            "history": history,
                            "best_val_loss": best_val_loss,
                            "train_config": _to_plain_data(asdict(train_config)),
                            "model_config": _to_plain_data(asdict(model_config)),
                            "data_config": _to_plain_data(asdict(data_config)),
                            "distributed_runtime": {
                                "enabled": runtime.enabled,
                                "backend": runtime.backend,
                                "world_size": runtime.world_size,
                            },
                            "effective_batch_size": effective_batch_size,
                            "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
                        }
                        _save_checkpoint(
                            _step_checkpoint_path(output_dir, train_config.checkpoint_name, int(optimizer_steps)),
                            step_checkpoint_payload,
                        )

                if (
                    runtime.is_primary
                    and train_config.save_every_train_batches is not None
                    and int(global_batch_steps) % train_config.save_every_train_batches == 0
                ):
                    base_model = _unwrap_model(model)
                    batch_checkpoint_payload = {
                        "epoch": epoch,
                        "global_batch_step": int(global_batch_steps),
                        "optimizer_step": int(optimizer_steps),
                        "batch_index_within_epoch": batch_index + 1,
                        "model_state_dict": base_model.state_dict(),
                        "encoder_state_dict": base_model.encoder.state_dict(),
                        "decoder_state_dict": base_model.decoder.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "history": history,
                        "best_val_loss": best_val_loss,
                        "train_config": _to_plain_data(asdict(train_config)),
                        "model_config": _to_plain_data(asdict(model_config)),
                        "data_config": _to_plain_data(asdict(data_config)),
                        "distributed_runtime": {
                            "enabled": runtime.enabled,
                            "backend": runtime.backend,
                            "world_size": runtime.world_size,
                        },
                        "effective_batch_size": effective_batch_size,
                        "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
                    }
                    _save_checkpoint(
                        _step_checkpoint_path(output_dir, train_config.checkpoint_name, int(global_batch_steps)),
                        batch_checkpoint_payload,
                    )

                running_loss += float(loss.item())
                train_steps += 1
                if (batch_index + 1) % max(train_config.log_every, 1) == 0:
                    display_loss = _reduced_mean_scalar(float(loss.item()), runtime)
                    _print_if_primary(
                        runtime,
                        f"[main][epoch {epoch}] batch {batch_index + 1} loss={display_loss:.6f}",
                    )

            train_loss_sum, global_train_steps = _reduced_sum_and_count(running_loss, train_steps, runtime)
            train_loss = train_loss_sum / max(global_train_steps, 1)
            val_loss: float | None = None
            val_denorm_mae: float | None = None
            val_variable_losses: dict[str, float] | None = None
            val_variable_mae: dict[str, float] | None = None

            if val_loader is not None:
                validation_result = _evaluate_main_forecast_model(
                    model,
                    val_loader,
                    criterion=criterion,
                    data_config=data_config,
                    runtime=runtime,
                    use_amp=train_config.use_amp,
                    amp_dtype=amp_dtype,
                    max_batches=train_config.max_val_batches,
                )
                val_loss = validation_result["loss"]
                val_denorm_mae = validation_result["denorm_mae"]
                val_variable_losses = validation_result["variable_losses"]
                val_variable_mae = validation_result["variable_denorm_mae"]

            epoch_record = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "optimizer_steps": float(optimizer_steps),
                "global_batch_steps": float(global_batch_steps),
            }
            if val_loss is not None:
                epoch_record["val_loss"] = val_loss
            if val_denorm_mae is not None:
                epoch_record["val_denorm_mae"] = val_denorm_mae
            history.append(epoch_record)
            _print_if_primary(
                runtime,
                f"[main][epoch {epoch}] train_loss={train_loss:.6f}"
                + (f" val_loss={val_loss:.6f}" if val_loss is not None else ""),
            )
            if runtime.is_primary and val_denorm_mae is not None and val_variable_mae is not None:
                if val_variable_losses is not None:
                    _print_json_block(
                        f"main_validation_loss_epoch_{epoch}",
                        {
                            "overall": val_loss,
                            **val_variable_losses,
                        },
                    )
                _print_json_block(
                    f"main_validation_denorm_mae_epoch_{epoch}",
                    {
                        "overall": val_denorm_mae,
                        **val_variable_mae,
                    },
                )

            if runtime.is_primary and train_config.save_epoch_checkpoint:
                base_model = _unwrap_model(model)
                checkpoint_best_val_loss = (
                    min(best_val_loss, float(val_loss)) if val_loss is not None else best_val_loss
                )
                checkpoint_payload = {
                    "epoch": epoch,
                    "global_batch_step": int(global_batch_steps),
                    "optimizer_step": int(optimizer_steps),
                    "model_state_dict": base_model.state_dict(),
                    "encoder_state_dict": base_model.encoder.state_dict(),
                    "decoder_state_dict": base_model.decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "history": history,
                    "best_val_loss": checkpoint_best_val_loss,
                    "train_config": _to_plain_data(asdict(train_config)),
                    "model_config": _to_plain_data(asdict(model_config)),
                    "data_config": _to_plain_data(asdict(data_config)),
                    "distributed_runtime": {
                        "enabled": runtime.enabled,
                        "backend": runtime.backend,
                        "world_size": runtime.world_size,
                    },
                    "effective_batch_size": effective_batch_size,
                    "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
                }
                _save_checkpoint(output_dir / train_config.checkpoint_name, checkpoint_payload)

                if train_config.save_best_checkpoint and val_loss is not None and val_loss < best_val_loss:
                    best_val_loss = checkpoint_best_val_loss
                    _save_checkpoint(output_dir / train_config.best_checkpoint_name, checkpoint_payload)

        result = {
            "smoke_report": smoke_report,
            "history": history,
            "checkpoint_path": str(output_dir / train_config.checkpoint_name),
            "best_checkpoint_path": str(output_dir / train_config.best_checkpoint_name),
            "resumed_from_checkpoint": None if resume_state is None else str(resume_state.checkpoint_path),
            "distributed": runtime.enabled,
            "world_size": runtime.world_size,
            "effective_batch_size": effective_batch_size,
            "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
            "optimizer_steps": int(history[-1]["optimizer_steps"]) if history else 0,
            "global_batch_steps": int(history[-1]["global_batch_steps"]) if history else 0,
            "train_batches": total_train_batches,
            "val_batches": total_val_batches,
        }
        if runtime.is_primary:
            _print_json_block("main_training_result", result)
        return result
    finally:
        _cleanup_distributed_runtime(runtime)


def validate_main_model(
    config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
    *,
    checkpoint_path: str | Path | None = None,
    split: str = "val",
    batch_size: int | None = None,
    num_workers: int | None = None,
    max_batches: int | None = None,
    start_time: str | pd.Timestamp | None = None,
    end_time: str | pd.Timestamp | None = None,
    print_model_summary: bool = False,
    save_rollout_plots: bool = False,
    rollout_output_dir: str | Path | None = None,
    rollout_samples: int = 1,
    rollout_passes: int = 3,
    rollout_anchor_stride_hours: int | None = None,
) -> dict[str, Any]:
    train_config, model_config, data_config, runtime, _model_dtype, amp_dtype = _build_main_training_objects(config_path)
    try:
        normalized_split = split.strip().lower()
        if normalized_split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        if checkpoint_path is None:
            resolved_checkpoint_path = train_config.output_dir / train_config.checkpoint_name
        else:
            resolved_checkpoint_path = resolve_repo_path(checkpoint_path, config_path=train_config.config_path)
        if not resolved_checkpoint_path.exists():
            raise FileNotFoundError(f"Main-model checkpoint not found at {resolved_checkpoint_path}")

        resolved_batch_size = int(train_config.batch_size if batch_size is None else batch_size)
        resolved_num_workers = int(train_config.num_workers if num_workers is None else num_workers)
        resolved_max_batches = (
            (train_config.max_val_batches if normalized_split == "val" else train_config.max_train_batches)
            if max_batches is None
            else int(max_batches)
        )
        resolved_start_time = _to_optional_timestamp(
            start_time
            if start_time is not None
            else (train_config.val_start_time if normalized_split == "val" else train_config.train_start_time)
        )
        resolved_end_time = _to_optional_timestamp(
            end_time
            if end_time is not None
            else (train_config.val_end_time if normalized_split == "val" else train_config.train_end_time)
        )
        if resolved_start_time is None and resolved_end_time is None:
            raise ValueError(
                f"No {normalized_split} evaluation window is configured. "
                "Set train_main/train_start_time and train_end_time or pass --start-time/--end-time."
            )

        model = FuXiLowerRes(model_config).to(runtime.device)
        if print_model_summary and runtime.is_primary:
            summary_inputs = _make_main_random_inputs(
                model_config,
                batch_size=train_config.random_smoke_batch_size,
                device=runtime.device,
            )
            _print_model_and_summary(
                "Forecast Model",
                model,
                input_data=summary_inputs,
                depth=train_config.summary_depth,
                print_summary=True,
            )
        checkpoint_metadata = _load_main_forecast_checkpoint(model, resolved_checkpoint_path)

        pin_memory = runtime.device.type == "cuda"
        eval_loader, eval_sampler = _build_eval_dataloader(
            data_config,
            batch_size=resolved_batch_size,
            num_workers=resolved_num_workers,
            runtime=runtime,
            start_time=resolved_start_time,
            end_time=resolved_end_time,
            pin_memory=pin_memory,
        )
        if eval_sampler is not None:
            eval_sampler.set_epoch(0)

        criterion = _build_main_forecast_criterion(train_config, data_config).to(runtime.device)
        evaluation = _evaluate_main_forecast_model(
            model,
            eval_loader,
            criterion=criterion,
            data_config=data_config,
            runtime=runtime,
            use_amp=train_config.use_amp,
            amp_dtype=amp_dtype,
            max_batches=resolved_max_batches,
        )
        rollout_report: dict[str, Any] | None = None
        if save_rollout_plots:
            if not runtime.is_primary:
                rollout_report = {
                    "output_dir": None,
                    "rollout_samples": 0,
                    "rollout_passes": int(rollout_passes),
                    "rollout_anchor_stride_hours": _resolve_rollout_anchor_stride_hours(
                        data_config,
                        rollout_anchor_stride_hours,
                    ),
                    "samples": [],
                }
            else:
                resolved_rollout_output_dir = resolve_repo_path(
                    rollout_output_dir or (train_config.output_dir / "validation_rollouts"),
                    config_path=train_config.config_path,
                )
                dataset_for_rollout = eval_loader.dataset
                if not isinstance(dataset_for_rollout, ArcoEra5FuXiDataset):
                    raise TypeError(
                        "Expected the validation loader to wrap ArcoEra5FuXiDataset for rollout plotting."
                    )
                rollout_report = _save_main_rollout_plots(
                    model,
                    dataset_for_rollout,
                    data_config=data_config,
                    runtime=runtime,
                    use_amp=train_config.use_amp,
                    amp_dtype=amp_dtype,
                    output_dir=resolved_rollout_output_dir,
                    rollout_samples=rollout_samples,
                    rollout_passes=rollout_passes,
                    rollout_anchor_stride_hours=rollout_anchor_stride_hours,
                )

        result = {
            "checkpoint_path": str(resolved_checkpoint_path),
            "checkpoint_epoch": None if checkpoint_metadata is None else checkpoint_metadata.get("epoch"),
            "checkpoint_optimizer_step": (
                None if checkpoint_metadata is None else checkpoint_metadata.get("optimizer_step")
            ),
            "checkpoint_global_batch_step": (
                None if checkpoint_metadata is None else checkpoint_metadata.get("global_batch_step")
            ),
            "split": normalized_split,
            "start_time": resolved_start_time,
            "end_time": resolved_end_time,
            "batch_size": resolved_batch_size,
            "num_workers": resolved_num_workers,
            "max_batches": resolved_max_batches,
            "evaluated_batches": evaluation["batches"],
            "loss": evaluation["loss"],
            "variable_losses": evaluation["variable_losses"],
            "denorm_mae": evaluation["denorm_mae"],
            "variable_denorm_mae": evaluation["variable_denorm_mae"],
            "distributed": runtime.enabled,
            "world_size": runtime.world_size,
        }
        if rollout_report is not None:
            result["rollout_plots"] = rollout_report
        if runtime.is_primary:
            _print_if_primary(
                runtime,
                f"[main][validate] split={normalized_split} loss={evaluation['loss']:.6f} "
                f"denorm_mae={evaluation['denorm_mae']:.6f}",
            )
            _print_json_block(
                "main_validation_loss",
                {
                    "overall": evaluation["loss"],
                    **evaluation["variable_losses"],
                },
            )
            _print_json_block(
                "main_validation_denorm_mae",
                {
                    "overall": evaluation["denorm_mae"],
                    **evaluation["variable_denorm_mae"],
                },
            )
            if rollout_report is not None:
                _print_json_block("main_validation_rollout_plots", rollout_report)
            _print_json_block("main_validation_result", result)
        return result
    finally:
        _cleanup_distributed_runtime(runtime)


def train_intrinsic_model(
    config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
    *,
    smoke_only: bool = False,
    resume_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    (
        train_config,
        encoder_config,
        intrinsic_config,
        data_config,
        runtime,
        _model_dtype,
        amp_dtype,
    ) = _build_intrinsic_training_objects(config_path)
    try:
        encoder = FuXiLowerResEncoder(encoder_config).to(runtime.device)
        if train_config.main_checkpoint_path is not None:
            _load_encoder_checkpoint(encoder, train_config.main_checkpoint_path)
        elif not smoke_only:
            raise FileNotFoundError(
                "train_intrinsic.main_checkpoint_path is required for intrinsic training after the main model."
            )
        encoder.eval()

        intrinsic_model = FuXiIntrinsic(intrinsic_config).to(runtime.device)
        resolved_resume_checkpoint_path = (
            train_config.resume_checkpoint_path
            if resume_checkpoint_path is None
            else resolve_repo_path(resume_checkpoint_path, config_path=train_config.config_path)
        )
        resume_checkpoint: dict[str, Any] | None = None
        smoke_report: dict[str, Any] | None = None

        if runtime.is_primary:
            smoke_inputs = _make_main_random_inputs(
                encoder_config,
                batch_size=train_config.random_smoke_batch_size,
                device=runtime.device,
            )
            _print_model_and_summary(
                "Frozen Forecast Encoder",
                encoder,
                input_data=smoke_inputs,
                depth=train_config.summary_depth,
                print_summary=train_config.print_model_summary,
            )
            intrinsic_smoke_input = (
                _make_intrinsic_random_inputs(
                    intrinsic_config,
                    batch_size=train_config.random_smoke_batch_size,
                    device=runtime.device,
                ),
            )
            _print_model_and_summary(
                "Intrinsic Model",
                intrinsic_model,
                input_data=intrinsic_smoke_input,
                depth=train_config.summary_depth,
                print_summary=train_config.print_model_summary,
            )
            smoke_report = run_intrinsic_model_smoke_test(
                encoder,
                intrinsic_model,
                batch_size=train_config.random_smoke_batch_size,
                print_outputs=True,
            )
        if smoke_only:
            return {"smoke_only": True, "smoke_report": smoke_report}
        if resolved_resume_checkpoint_path is not None:
            if not resolved_resume_checkpoint_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found at {resolved_resume_checkpoint_path}")
            resume_checkpoint = _load_intrinsic_checkpoint(intrinsic_model, resolved_resume_checkpoint_path)

        intrinsic_model = _wrap_for_distributed_training(intrinsic_model, runtime)
        pin_memory = runtime.device.type == "cuda"
        train_loader, val_loader, train_sampler, val_sampler = _build_split_dataloaders(
            data_config,
            batch_size=train_config.batch_size,
            num_workers=train_config.num_workers,
            runtime=runtime,
            train_start_time=train_config.train_start_time,
            train_end_time=train_config.train_end_time,
            val_start_time=train_config.val_start_time,
            val_end_time=train_config.val_end_time,
            pin_memory=pin_memory,
        )

        optimizer = AdamW(
            intrinsic_model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
        scaler = _build_grad_scaler(train_config.use_amp, runtime.device)
        criterion = nn.MSELoss()

        output_dir = train_config.output_dir
        if runtime.is_primary:
            output_dir.mkdir(parents=True, exist_ok=True)

        effective_batch_size = (
            train_config.batch_size * runtime.world_size * train_config.gradient_accumulation_steps
        )
        _print_if_primary(
            runtime,
            (
                f"[intrinsic] device={runtime.device} world_size={runtime.world_size} "
                f"gradient_accumulation_steps={train_config.gradient_accumulation_steps} "
                f"effective_batch_size={effective_batch_size}"
            ),
        )

        total_train_batches = _limited_length(train_loader, train_config.max_train_batches)
        total_val_batches = 0 if val_loader is None else _limited_length(val_loader, train_config.max_val_batches)
        resume_state = (
            None
            if resume_checkpoint is None
            else _build_resume_state(
                resume_checkpoint,
                checkpoint_path=resolved_resume_checkpoint_path,
                total_train_batches=total_train_batches,
                accumulation_steps=train_config.gradient_accumulation_steps,
            )
        )
        if resume_checkpoint is not None:
            resume_warnings = _validate_resume_compatibility(
                resume_checkpoint,
                checkpoint_path=resolved_resume_checkpoint_path,
                section_name="train_intrinsic",
                current_batch_size=train_config.batch_size,
                current_accumulation_steps=train_config.gradient_accumulation_steps,
                resume_state=resume_state,
            )
            if "optimizer_state_dict" not in resume_checkpoint:
                raise ValueError(
                    f"Resume checkpoint {resolved_resume_checkpoint_path} does not contain optimizer_state_dict."
                )
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            _apply_optimizer_hyperparameter_overrides(
                optimizer,
                learning_rate=train_config.learning_rate,
                weight_decay=train_config.weight_decay,
            )
            if "scaler_state_dict" in resume_checkpoint and scaler.is_enabled():
                scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
            for message in resume_warnings + _resume_optimizer_override_messages(
                resume_checkpoint,
                learning_rate=train_config.learning_rate,
                weight_decay=train_config.weight_decay,
                section_name="train_intrinsic",
            ):
                _print_if_primary(runtime, f"[intrinsic][resume] {message}")
        history: list[dict[str, float]] = [] if resume_state is None else list(resume_state.history)
        best_val_loss = float("inf") if resume_state is None else resume_state.best_val_loss
        optimizer_steps = 0 if resume_state is None else resume_state.optimizer_steps
        global_batch_steps = 0 if resume_state is None else resume_state.global_batch_steps
        start_epoch = 1 if resume_state is None else resume_state.start_epoch
        if resume_state is not None:
            _print_if_primary(
                runtime,
                f"[intrinsic] resuming from {resume_state.checkpoint_path} "
                f"checkpoint_epoch={resume_state.checkpoint_epoch} start_epoch={resume_state.start_epoch} "
                f"resume_batch={resume_state.resume_batch_index} optimizer_steps={resume_state.optimizer_steps} "
                f"global_batch_steps={resume_state.global_batch_steps}",
            )
            if start_epoch > train_config.max_epochs:
                raise ValueError(
                    f"Resume checkpoint {resume_state.checkpoint_path} would continue at epoch {start_epoch}, "
                    f"but train_intrinsic.max_epochs={train_config.max_epochs}. Increase max_epochs to continue training."
                )

        for epoch in range(start_epoch, train_config.max_epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)

            intrinsic_model.train()
            optimizer.zero_grad(set_to_none=True)
            encoder.zero_grad(set_to_none=True)
            running_loss = 0.0
            train_steps = 0
            resume_batch_index = (
                0
                if resume_state is None or resume_state.resume_epoch != epoch
                else resume_state.resume_batch_index
            )
            replay_start_batch_index = (
                0
                if resume_state is None or resume_state.resume_epoch != epoch
                else resume_state.replay_start_batch_index
            )
            if resume_batch_index > 0:
                replay_from = replay_start_batch_index + 1
                replay_to = resume_batch_index
                if replay_start_batch_index < resume_batch_index:
                    _print_if_primary(
                        runtime,
                        f"[intrinsic][epoch {epoch}] rebuilding accumulated gradients from batches "
                        f"{replay_from}-{replay_to} before resuming at batch {resume_batch_index + 1}",
                    )
                else:
                    _print_if_primary(
                        runtime,
                        f"[intrinsic][epoch {epoch}] skipping directly to batch {resume_batch_index + 1}",
                    )

            for batch_index, batch in _iter_prefetched_batches(
                train_loader,
                runtime=runtime,
                max_batches=train_config.max_train_batches,
            ):
                if batch_index < resume_batch_index:
                    if batch_index < replay_start_batch_index:
                        continue
                    should_step = _should_optimizer_step(
                        total_train_batches,
                        batch_index,
                        train_config.gradient_accumulation_steps,
                    )
                    if should_step:
                        raise RuntimeError(
                            "Resume replay encountered an optimizer-step batch. "
                            "This indicates the checkpoint resume window was computed incorrectly."
                        )
                    loss_divisor = float(
                        _accumulation_divisor(
                            total_train_batches,
                            batch_index,
                            train_config.gradient_accumulation_steps,
                        )
                    )
                    sync_context = nullcontext()
                    if runtime.enabled and isinstance(intrinsic_model, DistributedDataParallel):
                        sync_context = intrinsic_model.no_sync()
                    patch_grid_features = _encode_patch_grid_features_for_intrinsic(
                        encoder,
                        batch,
                        detach_features=train_config.detach_second_block_features,
                        clear_encoder_grads=True,
                    )
                    with sync_context:
                        with _amp_autocast_context(train_config.use_amp, runtime.device, amp_dtype):
                            outputs = intrinsic_model(patch_grid_features)
                            scaled_loss = criterion(
                                outputs["patch_grid_features_recon"],
                                patch_grid_features,
                            ) / loss_divisor
                        if scaler.is_enabled():
                            scaler.scale(scaled_loss).backward()
                        else:
                            scaled_loss.backward()
                    continue

                should_step = _should_optimizer_step(
                    total_train_batches,
                    batch_index,
                    train_config.gradient_accumulation_steps,
                )
                global_batch_steps += 1
                loss_divisor = float(
                    _accumulation_divisor(
                        total_train_batches,
                        batch_index,
                        train_config.gradient_accumulation_steps,
                    )
                )
                sync_context = nullcontext()
                if runtime.enabled and isinstance(intrinsic_model, DistributedDataParallel) and not should_step:
                    sync_context = intrinsic_model.no_sync()

                patch_grid_features = _encode_patch_grid_features_for_intrinsic(
                    encoder,
                    batch,
                    detach_features=train_config.detach_second_block_features,
                    clear_encoder_grads=True,
                )

                with sync_context:
                    with _amp_autocast_context(train_config.use_amp, runtime.device, amp_dtype):
                        outputs = intrinsic_model(patch_grid_features)
                        loss = criterion(
                            outputs["patch_grid_features_recon"],
                            patch_grid_features,
                        )
                        scaled_loss = loss / loss_divisor

                    if scaler.is_enabled():
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                if should_step:
                    if train_config.gradient_clip_norm is not None:
                        if scaler.is_enabled():
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            _unwrap_model(intrinsic_model).parameters(),
                            train_config.gradient_clip_norm,
                        )

                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1

                    if (
                        runtime.is_primary
                        and train_config.save_every_optimizer_steps is not None
                        and int(optimizer_steps) % train_config.save_every_optimizer_steps == 0
                    ):
                        step_checkpoint_payload = {
                            "epoch": epoch,
                            "optimizer_step": int(optimizer_steps),
                            "batch_index_within_epoch": batch_index + 1,
                            "intrinsic_state_dict": _unwrap_model(intrinsic_model).state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scaler_state_dict": scaler.state_dict(),
                            "history": history,
                            "best_val_loss": best_val_loss,
                            "train_config": _to_plain_data(asdict(train_config)),
                            "encoder_config": _to_plain_data(asdict(encoder_config)),
                            "intrinsic_config": _to_plain_data(asdict(intrinsic_config)),
                            "data_config": _to_plain_data(asdict(data_config)),
                            "main_checkpoint_path": str(train_config.main_checkpoint_path) if train_config.main_checkpoint_path else None,
                            "distributed_runtime": {
                                "enabled": runtime.enabled,
                                "backend": runtime.backend,
                                "world_size": runtime.world_size,
                            },
                            "effective_batch_size": effective_batch_size,
                            "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
                        }
                        _save_checkpoint(
                            _step_checkpoint_path(output_dir, train_config.checkpoint_name, int(optimizer_steps)),
                            step_checkpoint_payload,
                        )

                if (
                    runtime.is_primary
                    and train_config.save_every_train_batches is not None
                    and int(global_batch_steps) % train_config.save_every_train_batches == 0
                ):
                    batch_checkpoint_payload = {
                        "epoch": epoch,
                        "global_batch_step": int(global_batch_steps),
                        "optimizer_step": int(optimizer_steps),
                        "batch_index_within_epoch": batch_index + 1,
                        "intrinsic_state_dict": _unwrap_model(intrinsic_model).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "history": history,
                        "best_val_loss": best_val_loss,
                        "train_config": _to_plain_data(asdict(train_config)),
                        "encoder_config": _to_plain_data(asdict(encoder_config)),
                        "intrinsic_config": _to_plain_data(asdict(intrinsic_config)),
                        "data_config": _to_plain_data(asdict(data_config)),
                        "main_checkpoint_path": str(train_config.main_checkpoint_path) if train_config.main_checkpoint_path else None,
                        "distributed_runtime": {
                            "enabled": runtime.enabled,
                            "backend": runtime.backend,
                            "world_size": runtime.world_size,
                        },
                        "effective_batch_size": effective_batch_size,
                        "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
                    }
                    _save_checkpoint(
                        _step_checkpoint_path(output_dir, train_config.checkpoint_name, int(global_batch_steps)),
                        batch_checkpoint_payload,
                    )

                running_loss += float(loss.item())
                train_steps += 1
                if (batch_index + 1) % max(train_config.log_every, 1) == 0:
                    display_loss = _reduced_mean_scalar(float(loss.item()), runtime)
                    _print_if_primary(
                        runtime,
                        f"[intrinsic][epoch {epoch}] batch {batch_index + 1} loss={display_loss:.6f}",
                    )

            train_loss_sum, global_train_steps = _reduced_sum_and_count(running_loss, train_steps, runtime)
            train_loss = train_loss_sum / max(global_train_steps, 1)
            val_loss: float | None = None

            if val_loader is not None:
                encoder.zero_grad(set_to_none=True)
                intrinsic_model.eval()
                val_running_loss = 0.0
                val_steps = 0
                with torch.no_grad():
                    for _batch_index, batch in _iter_prefetched_batches(
                        val_loader,
                        runtime=runtime,
                        max_batches=train_config.max_val_batches,
                    ):
                        patch_grid_features = _encode_patch_grid_features_for_intrinsic(
                            encoder,
                            batch,
                            detach_features=train_config.detach_second_block_features,
                            clear_encoder_grads=False,
                        )
                        with _amp_autocast_context(train_config.use_amp, runtime.device, amp_dtype):
                            outputs = intrinsic_model(patch_grid_features)
                            loss = criterion(
                                outputs["patch_grid_features_recon"],
                                patch_grid_features,
                            )
                        val_running_loss += float(loss.item())
                        val_steps += 1

                val_loss_sum, global_val_steps = _reduced_sum_and_count(val_running_loss, val_steps, runtime)
                val_loss = val_loss_sum / max(global_val_steps, 1)

            epoch_record = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "optimizer_steps": float(optimizer_steps),
                "global_batch_steps": float(global_batch_steps),
            }
            if val_loss is not None:
                epoch_record["val_loss"] = val_loss
            history.append(epoch_record)
            _print_if_primary(
                runtime,
                f"[intrinsic][epoch {epoch}] train_loss={train_loss:.6f}"
                + (f" val_loss={val_loss:.6f}" if val_loss is not None else ""),
            )

            if runtime.is_primary and train_config.save_epoch_checkpoint:
                checkpoint_best_val_loss = (
                    min(best_val_loss, float(val_loss)) if val_loss is not None else best_val_loss
                )
                checkpoint_payload = {
                    "epoch": epoch,
                    "global_batch_step": int(global_batch_steps),
                    "optimizer_step": int(optimizer_steps),
                    "intrinsic_state_dict": _unwrap_model(intrinsic_model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "history": history,
                    "best_val_loss": checkpoint_best_val_loss,
                    "train_config": _to_plain_data(asdict(train_config)),
                    "encoder_config": _to_plain_data(asdict(encoder_config)),
                    "intrinsic_config": _to_plain_data(asdict(intrinsic_config)),
                    "data_config": _to_plain_data(asdict(data_config)),
                    "main_checkpoint_path": str(train_config.main_checkpoint_path) if train_config.main_checkpoint_path else None,
                    "distributed_runtime": {
                        "enabled": runtime.enabled,
                        "backend": runtime.backend,
                        "world_size": runtime.world_size,
                    },
                    "effective_batch_size": effective_batch_size,
                    "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
                }
                _save_checkpoint(output_dir / train_config.checkpoint_name, checkpoint_payload)

                if train_config.save_best_checkpoint and val_loss is not None and val_loss < best_val_loss:
                    best_val_loss = checkpoint_best_val_loss
                    _save_checkpoint(output_dir / train_config.best_checkpoint_name, checkpoint_payload)

        result = {
            "smoke_report": smoke_report,
            "history": history,
            "checkpoint_path": str(output_dir / train_config.checkpoint_name),
            "best_checkpoint_path": str(output_dir / train_config.best_checkpoint_name),
            "resumed_from_checkpoint": None if resume_state is None else str(resume_state.checkpoint_path),
            "distributed": runtime.enabled,
            "world_size": runtime.world_size,
            "effective_batch_size": effective_batch_size,
            "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
            "optimizer_steps": int(history[-1]["optimizer_steps"]) if history else 0,
            "global_batch_steps": int(history[-1]["global_batch_steps"]) if history else 0,
            "train_batches": total_train_batches,
            "val_batches": total_val_batches,
        }
        if runtime.is_primary:
            _print_json_block("intrinsic_training_result", result)
        return result
    finally:
        _cleanup_distributed_runtime(runtime)


__all__ = [
    "IntrinsicTrainingConfig",
    "LatitudeWeightedCharbonnierLoss",
    "MainTrainingConfig",
    "run_intrinsic_model_smoke_test",
    "run_main_model_smoke_test",
    "train_intrinsic_model",
    "train_main_model",
]
