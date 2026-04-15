from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import DEFAULT_MODEL_CONFIG_PATH
from .data import ArcoEra5FuXiDataConfig, ArcoEra5FuXiDataset
from .models import FuXiLowerRes, FuXiLowerResConfig
from .training.pipeline import MainTrainingConfig


CHINCHILLA_TOKENS_PER_PARAMETER = 20.0
_SCALING_CLOSE_LOWER_RATIO = 0.5
_SCALING_CLOSE_UPPER_RATIO = 2.0


@dataclass(frozen=True)
class ScalingLawReport:
    config_path: str
    model_device: str
    model_dtype: str
    parameter_count: int
    trainable_parameter_count: int
    parameter_size_bytes: int
    parameter_size_mib: float
    parameter_size_gib: float
    input_size: tuple[int, int]
    resized_input_size: tuple[int, int]
    patch_size: tuple[int, int]
    patch_grid: tuple[int, int]
    tokens_per_sample: int
    train_samples: int
    train_unique_tokens: int
    train_unique_tokens_billions: float
    train_window_start: str | None
    train_window_end: str | None
    single_process_batches_per_epoch: int
    full_split_each_epoch_single_process: bool
    scheduled_samples_per_epoch_single_process: int
    max_epochs: int
    scheduled_train_tokens_single_process: int
    scheduled_train_tokens_single_process_billions: float
    chinchilla_tokens_per_parameter: float
    chinchilla_target_tokens: int
    chinchilla_target_tokens_billions: float
    unique_train_ratio_to_chinchilla: float
    unique_train_verdict: str
    epochs_of_full_split_to_reach_chinchilla: float
    scheduled_ratio_to_chinchilla_single_process: float
    scheduled_verdict_single_process: str
    forward_ran: bool
    forecast_shape: tuple[int, ...] | None
    second_block_features_shape: tuple[int, ...] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if not trainable_only or parameter.requires_grad
        )
    )


def parameter_size_bytes(model: nn.Module, *, trainable_only: bool = False) -> int:
    return int(
        sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
            if not trainable_only or parameter.requires_grad
        )
    )


def tokens_per_sample_from_model_config(model_config: FuXiLowerResConfig) -> int:
    return int(model_config.patch_grid[0] * model_config.patch_grid[1])


def chinchilla_target_tokens(
    parameter_count: int,
    *,
    tokens_per_parameter: float = CHINCHILLA_TOKENS_PER_PARAMETER,
) -> int:
    if parameter_count < 0:
        raise ValueError(f"parameter_count must be non-negative, got {parameter_count}")
    if tokens_per_parameter <= 0:
        raise ValueError(f"tokens_per_parameter must be positive, got {tokens_per_parameter}")
    return int(round(float(parameter_count) * float(tokens_per_parameter)))


def classify_scaling_ratio(ratio: float) -> str:
    if ratio < _SCALING_CLOSE_LOWER_RATIO:
        return "below Chinchilla heuristic"
    if ratio > _SCALING_CLOSE_UPPER_RATIO:
        return "above Chinchilla heuristic"
    return "near Chinchilla heuristic"


def single_process_samples_per_epoch(
    train_samples: int,
    *,
    batch_size: int,
    max_train_batches: int | None,
) -> tuple[int, int, bool]:
    if train_samples < 0:
        raise ValueError(f"train_samples must be non-negative, got {train_samples}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    loader_batches = 0 if train_samples == 0 else int(math.ceil(train_samples / batch_size))
    if max_train_batches is None:
        return train_samples, loader_batches, True

    full_split = int(max_train_batches) >= loader_batches
    if full_split:
        return train_samples, loader_batches, True
    return min(train_samples, int(max_train_batches) * int(batch_size)), loader_batches, False


def _cpu_model_config(config_path: str | Path) -> FuXiLowerResConfig:
    config = FuXiLowerResConfig.from_yaml(config_path)
    return replace(config, device=torch.device("cpu"), dtype=torch.float32)


def _make_cpu_forward_inputs(model_config: FuXiLowerResConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_height, input_width = model_config.input_size
    x = torch.zeros(
        1,
        model_config.time_steps,
        model_config.in_chans,
        input_height,
        input_width,
        device="cpu",
        dtype=torch.float32,
    )
    temb = torch.zeros(1, model_config.temb_dim, device="cpu", dtype=torch.float32)
    static = torch.zeros(
        1,
        model_config.aux_chans,
        input_height,
        input_width,
        device="cpu",
        dtype=torch.float32,
    )
    return x, temb, static


def build_main_model_scaling_report(
    config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
    *,
    run_forward: bool = True,
) -> ScalingLawReport:
    resolved_config_path = Path(config_path).resolve()
    model_config = _cpu_model_config(resolved_config_path)
    train_config = MainTrainingConfig.from_yaml(resolved_config_path)
    data_config = replace(
        ArcoEra5FuXiDataConfig.from_yaml(resolved_config_path),
        forecast_steps=model_config.forecast_steps,
        start_time=train_config.train_start_time,
        end_time=train_config.train_end_time,
    )

    model = FuXiLowerRes(model_config).to(device="cpu")
    model.eval()

    total_parameters = count_parameters(model)
    trainable_parameters = count_parameters(model, trainable_only=True)
    total_parameter_bytes = parameter_size_bytes(model)
    tokens_per_sample = tokens_per_sample_from_model_config(model_config)

    dataset = ArcoEra5FuXiDataset(data_config)
    train_samples = len(dataset)
    train_unique_tokens = int(train_samples) * int(tokens_per_sample)

    scheduled_samples_per_epoch, single_process_batches_per_epoch, full_split_each_epoch = (
        single_process_samples_per_epoch(
            train_samples,
            batch_size=train_config.batch_size,
            max_train_batches=train_config.max_train_batches,
        )
    )
    scheduled_train_tokens = int(scheduled_samples_per_epoch) * int(tokens_per_sample) * int(train_config.max_epochs)

    chinchilla_target = chinchilla_target_tokens(total_parameters)
    unique_ratio = 0.0 if chinchilla_target == 0 else float(train_unique_tokens) / float(chinchilla_target)
    scheduled_ratio = 0.0 if chinchilla_target == 0 else float(scheduled_train_tokens) / float(chinchilla_target)
    epochs_to_target = float("inf") if train_unique_tokens == 0 else float(chinchilla_target) / float(train_unique_tokens)

    forecast_shape: tuple[int, ...] | None = None
    second_block_features_shape: tuple[int, ...] | None = None
    if run_forward:
        x, temb, static = _make_cpu_forward_inputs(model_config)
        with torch.inference_mode():
            outputs = model(x, temb, static_features=static)
        forecast_shape = tuple(int(value) for value in outputs["forecast"].shape)
        second_block_features_shape = tuple(int(value) for value in outputs["second_block_features"].shape)

    return ScalingLawReport(
        config_path=str(resolved_config_path),
        model_device=str(next(model.parameters()).device),
        model_dtype=str(next(model.parameters()).dtype),
        parameter_count=total_parameters,
        trainable_parameter_count=trainable_parameters,
        parameter_size_bytes=total_parameter_bytes,
        parameter_size_mib=float(total_parameter_bytes) / (1024.0**2),
        parameter_size_gib=float(total_parameter_bytes) / (1024.0**3),
        input_size=tuple(int(value) for value in model_config.input_size),
        resized_input_size=tuple(int(value) for value in model_config.resized_input_size),
        patch_size=tuple(int(value) for value in model_config.patch_size),
        patch_grid=tuple(int(value) for value in model_config.patch_grid),
        tokens_per_sample=int(tokens_per_sample),
        train_samples=int(train_samples),
        train_unique_tokens=int(train_unique_tokens),
        train_unique_tokens_billions=float(train_unique_tokens) / 1.0e9,
        train_window_start=None if train_config.train_start_time is None else str(train_config.train_start_time),
        train_window_end=None if train_config.train_end_time is None else str(train_config.train_end_time),
        single_process_batches_per_epoch=int(single_process_batches_per_epoch),
        full_split_each_epoch_single_process=bool(full_split_each_epoch),
        scheduled_samples_per_epoch_single_process=int(scheduled_samples_per_epoch),
        max_epochs=int(train_config.max_epochs),
        scheduled_train_tokens_single_process=int(scheduled_train_tokens),
        scheduled_train_tokens_single_process_billions=float(scheduled_train_tokens) / 1.0e9,
        chinchilla_tokens_per_parameter=float(CHINCHILLA_TOKENS_PER_PARAMETER),
        chinchilla_target_tokens=int(chinchilla_target),
        chinchilla_target_tokens_billions=float(chinchilla_target) / 1.0e9,
        unique_train_ratio_to_chinchilla=float(unique_ratio),
        unique_train_verdict=classify_scaling_ratio(unique_ratio),
        epochs_of_full_split_to_reach_chinchilla=float(epochs_to_target),
        scheduled_ratio_to_chinchilla_single_process=float(scheduled_ratio),
        scheduled_verdict_single_process=classify_scaling_ratio(scheduled_ratio),
        forward_ran=bool(run_forward),
        forecast_shape=forecast_shape,
        second_block_features_shape=second_block_features_shape,
    )


__all__ = [
    "CHINCHILLA_TOKENS_PER_PARAMETER",
    "ScalingLawReport",
    "build_main_model_scaling_report",
    "chinchilla_target_tokens",
    "classify_scaling_ratio",
    "count_parameters",
    "parameter_size_bytes",
    "single_process_samples_per_epoch",
    "tokens_per_sample_from_model_config",
]
