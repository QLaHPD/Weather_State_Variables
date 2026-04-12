from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
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


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            moved[key] = value.to(device)
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
        encoded = encoder(x, temb, static_features=static_features)
        outputs = intrinsic_model(encoded.second_block_features)
    report = {
        "encoder_input": {
            "x": {"shape": list(x.shape), "dtype": str(x.dtype)},
            "temb": {"shape": list(temb.shape), "dtype": str(temb.dtype)},
            "static_features": {"shape": list(static_features.shape), "dtype": str(static_features.dtype)},
        },
        "second_block_features": {
            "shape": list(encoded.second_block_features.shape),
            "dtype": str(encoded.second_block_features.dtype),
        },
        "intrinsic_output": _tensor_tree_shapes(outputs),
    }
    if print_outputs:
        _print_json_block("intrinsic_smoke_test", report)
    return report


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
    main_checkpoint_path: Path | None = None
    detach_second_block_features: bool = True
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
            main_checkpoint_path=checkpoint_path,
            detach_second_block_features=bool(
                data.get(
                    "detach_second_block_features",
                    data.get("detach_z_high", True),
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
        spatial_size=encoder_config.latent_grid,
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
    train_dataset = ArcoEra5FuXiDataset(
        replace(
            data_config,
            start_time=train_start_time,
            end_time=train_end_time,
        )
    )
    train_sampler: Sampler[Any] | None = None
    if runtime.enabled:
        train_sampler = ContiguousDistributedSampler(
            train_dataset,
            num_replicas=runtime.world_size,
            rank=runtime.rank,
        )
    train_loader = build_arco_era5_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=train_sampler,
    )

    if val_start_time is None and val_end_time is None:
        return train_loader, None, train_sampler, None

    val_dataset = ArcoEra5FuXiDataset(
        replace(
            data_config,
            start_time=val_start_time,
            end_time=val_end_time,
        )
    )
    val_sampler: Sampler[Any] | None = None
    if runtime.enabled:
        val_sampler = ContiguousDistributedSampler(
            val_dataset,
            num_replicas=runtime.world_size,
            rank=runtime.rank,
        )
    val_loader = build_arco_era5_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=val_sampler,
    )
    return train_loader, val_loader, train_sampler, val_sampler


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


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


def _iter_limited(loader: Iterable[dict[str, Any]], max_batches: int | None) -> Iterable[tuple[int, dict[str, Any]]]:
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        yield batch_index, batch


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
) -> dict[str, Any]:
    train_config, model_config, data_config, runtime, _model_dtype, amp_dtype = _build_main_training_objects(config_path)
    try:
        model = FuXiLowerRes(model_config).to(runtime.device)
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

        best_val_loss = float("inf")
        history: list[dict[str, float]] = []
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

        for epoch in range(1, train_config.max_epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)

            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            train_steps = 0

            for batch_index, batch in _iter_limited(train_loader, train_config.max_train_batches):
                batch = _move_batch_to_device(batch, runtime.device)
                should_step = _should_optimizer_step(
                    total_train_batches,
                    batch_index,
                    train_config.gradient_accumulation_steps,
                )
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

            if val_loader is not None:
                model.eval()
                val_running_loss = 0.0
                val_steps = 0
                with torch.no_grad():
                    for _batch_index, batch in _iter_limited(val_loader, train_config.max_val_batches):
                        batch = _move_batch_to_device(batch, runtime.device)
                        with _amp_autocast_context(train_config.use_amp, runtime.device, amp_dtype):
                            outputs = model(
                                batch["x"],
                                batch["temb"],
                                static_features=batch["static_features"],
                            )
                            loss = criterion(outputs["forecast"], batch["target"])
                        val_running_loss += float(loss.item())
                        val_steps += 1

                val_loss_sum, global_val_steps = _reduced_sum_and_count(val_running_loss, val_steps, runtime)
                val_loss = val_loss_sum / max(global_val_steps, 1)

            epoch_record = {"epoch": float(epoch), "train_loss": train_loss}
            if val_loss is not None:
                epoch_record["val_loss"] = val_loss
            history.append(epoch_record)
            _print_if_primary(
                runtime,
                f"[main][epoch {epoch}] train_loss={train_loss:.6f}"
                + (f" val_loss={val_loss:.6f}" if val_loss is not None else ""),
            )

            if runtime.is_primary:
                base_model = _unwrap_model(model)
                checkpoint_payload = {
                    "epoch": epoch,
                    "model_state_dict": base_model.state_dict(),
                    "encoder_state_dict": base_model.encoder.state_dict(),
                    "decoder_state_dict": base_model.decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
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

                if val_loss is not None and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    _save_checkpoint(output_dir / train_config.best_checkpoint_name, checkpoint_payload)

        result = {
            "smoke_report": smoke_report,
            "history": history,
            "checkpoint_path": str(output_dir / train_config.checkpoint_name),
            "best_checkpoint_path": str(output_dir / train_config.best_checkpoint_name),
            "distributed": runtime.enabled,
            "world_size": runtime.world_size,
            "effective_batch_size": effective_batch_size,
            "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
            "train_batches": total_train_batches,
            "val_batches": total_val_batches,
        }
        if runtime.is_primary:
            _print_json_block("main_training_result", result)
        return result
    finally:
        _cleanup_distributed_runtime(runtime)


def train_intrinsic_model(
    config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
    *,
    smoke_only: bool = False,
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
        encoder.requires_grad_(False)

        intrinsic_model = FuXiIntrinsic(intrinsic_config).to(runtime.device)
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

        best_val_loss = float("inf")
        history: list[dict[str, float]] = []
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

        for epoch in range(1, train_config.max_epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)

            intrinsic_model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            train_steps = 0

            for batch_index, batch in _iter_limited(train_loader, train_config.max_train_batches):
                batch = _move_batch_to_device(batch, runtime.device)
                should_step = _should_optimizer_step(
                    total_train_batches,
                    batch_index,
                    train_config.gradient_accumulation_steps,
                )
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

                with torch.no_grad():
                    encoded = encoder(batch["x"], batch["temb"], static_features=batch["static_features"])
                    second_block_features = (
                        encoded.second_block_features.detach()
                        if train_config.detach_second_block_features
                        else encoded.second_block_features
                    )

                with sync_context:
                    with _amp_autocast_context(train_config.use_amp, runtime.device, amp_dtype):
                        outputs = intrinsic_model(second_block_features)
                        loss = criterion(
                            outputs["second_block_features_recon"],
                            second_block_features,
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
                intrinsic_model.eval()
                val_running_loss = 0.0
                val_steps = 0
                with torch.no_grad():
                    for _batch_index, batch in _iter_limited(val_loader, train_config.max_val_batches):
                        batch = _move_batch_to_device(batch, runtime.device)
                        encoded = encoder(batch["x"], batch["temb"], static_features=batch["static_features"])
                        second_block_features = (
                            encoded.second_block_features.detach()
                            if train_config.detach_second_block_features
                            else encoded.second_block_features
                        )
                        with _amp_autocast_context(train_config.use_amp, runtime.device, amp_dtype):
                            outputs = intrinsic_model(second_block_features)
                            loss = criterion(
                                outputs["second_block_features_recon"],
                                second_block_features,
                            )
                        val_running_loss += float(loss.item())
                        val_steps += 1

                val_loss_sum, global_val_steps = _reduced_sum_and_count(val_running_loss, val_steps, runtime)
                val_loss = val_loss_sum / max(global_val_steps, 1)

            epoch_record = {"epoch": float(epoch), "train_loss": train_loss}
            if val_loss is not None:
                epoch_record["val_loss"] = val_loss
            history.append(epoch_record)
            _print_if_primary(
                runtime,
                f"[intrinsic][epoch {epoch}] train_loss={train_loss:.6f}"
                + (f" val_loss={val_loss:.6f}" if val_loss is not None else ""),
            )

            if runtime.is_primary:
                checkpoint_payload = {
                    "epoch": epoch,
                    "intrinsic_state_dict": _unwrap_model(intrinsic_model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
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

                if val_loss is not None and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    _save_checkpoint(output_dir / train_config.best_checkpoint_name, checkpoint_payload)

        result = {
            "smoke_report": smoke_report,
            "history": history,
            "checkpoint_path": str(output_dir / train_config.checkpoint_name),
            "best_checkpoint_path": str(output_dir / train_config.best_checkpoint_name),
            "distributed": runtime.enabled,
            "world_size": runtime.world_size,
            "effective_batch_size": effective_batch_size,
            "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
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
