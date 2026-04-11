from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchinfo import summary as torchinfo_summary

from ..config import (
    DEFAULT_MODEL_CONFIG_PATH,
    load_config_section,
    resolve_repo_path,
    resolve_torch_dtype,
)
from ..data import ArcoEra5FuXiDataConfig, ArcoEra5FuXiDataset, build_arco_era5_dataloader
from ..models import (
    FuXiIntrinsic,
    FuXiIntrinsicConfig,
    FuXiLowerRes,
    FuXiLowerResConfig,
    FuXiLowerResEncoder,
)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


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
        intrinsic_config.d_high,
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
        outputs = intrinsic_model(encoded.z_high)
    report = {
        "encoder_input": {
            "x": {"shape": list(x.shape), "dtype": str(x.dtype)},
            "temb": {"shape": list(temb.shape), "dtype": str(temb.dtype)},
            "static_features": {"shape": list(static_features.shape), "dtype": str(static_features.dtype)},
        },
        "z_high": {"shape": list(encoded.z_high.shape), "dtype": str(encoded.z_high.dtype)},
        "intrinsic_output": _tensor_tree_shapes(outputs),
    }
    if print_outputs:
        _print_json_block("intrinsic_smoke_test", report)
    return report


@dataclass(frozen=True)
class MainTrainingConfig:
    batch_size: int = 1
    num_workers: int = 0
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_epochs: int = 1
    device: str = "auto"
    model_dtype: str = "float32"
    use_amp: bool = False
    amp_dtype: str = "float16"
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
            learning_rate=float(data.get("learning_rate", 1e-4)),
            weight_decay=float(data.get("weight_decay", 0.0)),
            max_epochs=int(data.get("max_epochs", 1)),
            device=str(data.get("device", "auto")),
            model_dtype=str(data.get("model_dtype", "float32")),
            use_amp=bool(data.get("use_amp", False)),
            amp_dtype=str(data.get("amp_dtype", "float16")),
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
    detach_z_high: bool = True
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
            detach_z_high=bool(data.get("detach_z_high", True)),
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
) -> tuple[MainTrainingConfig, FuXiLowerResConfig, ArcoEra5FuXiDataConfig, torch.device, torch.dtype, torch.dtype | None]:
    train_config = MainTrainingConfig.from_yaml(config_path)
    device = _resolve_device(train_config.device)
    model_dtype = resolve_torch_dtype(train_config.model_dtype) or torch.float32
    amp_dtype = resolve_torch_dtype(train_config.amp_dtype)

    model_config = replace(
        FuXiLowerResConfig.from_yaml(config_path),
        device=device,
        dtype=model_dtype,
    )
    data_config = replace(
        ArcoEra5FuXiDataConfig.from_yaml(config_path),
        forecast_steps=model_config.forecast_steps,
    )
    return train_config, model_config, data_config, device, model_dtype, amp_dtype


def _build_intrinsic_training_objects(
    config_path: str | Path,
) -> tuple[
    IntrinsicTrainingConfig,
    FuXiLowerResConfig,
    FuXiIntrinsicConfig,
    ArcoEra5FuXiDataConfig,
    torch.device,
    torch.dtype,
    torch.dtype | None,
]:
    train_config = IntrinsicTrainingConfig.from_yaml(config_path)
    device = _resolve_device(train_config.device)
    model_dtype = resolve_torch_dtype(train_config.model_dtype) or torch.float32
    amp_dtype = resolve_torch_dtype(train_config.amp_dtype)

    encoder_config = replace(
        FuXiLowerResConfig.from_yaml(config_path),
        device=device,
        dtype=model_dtype,
    )
    intrinsic_config = replace(
        FuXiIntrinsicConfig.from_yaml(config_path),
        device=device,
        dtype=model_dtype,
        d_high=encoder_config.d_high,
        spatial_size=encoder_config.latent_grid,
    )
    data_config = replace(
        ArcoEra5FuXiDataConfig.from_yaml(config_path),
        forecast_steps=encoder_config.forecast_steps,
    )
    return train_config, encoder_config, intrinsic_config, data_config, device, model_dtype, amp_dtype


def _build_split_dataloaders(
    data_config: ArcoEra5FuXiDataConfig,
    *,
    batch_size: int,
    num_workers: int,
    train_start_time: pd.Timestamp | None,
    train_end_time: pd.Timestamp | None,
    val_start_time: pd.Timestamp | None,
    val_end_time: pd.Timestamp | None,
    pin_memory: bool,
) -> tuple[DataLoader[dict[str, Any]], DataLoader[dict[str, Any]] | None]:
    train_dataset = ArcoEra5FuXiDataset(
        replace(
            data_config,
            start_time=train_start_time,
            end_time=train_end_time,
        )
    )
    train_loader = build_arco_era5_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if val_start_time is None and val_end_time is None:
        return train_loader, None

    val_dataset = ArcoEra5FuXiDataset(
        replace(
            data_config,
            start_time=val_start_time,
            end_time=val_end_time,
        )
    )
    val_loader = build_arco_era5_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


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


def train_main_model(
    config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
    *,
    smoke_only: bool = False,
) -> dict[str, Any]:
    train_config, model_config, data_config, device, _model_dtype, amp_dtype = _build_main_training_objects(config_path)
    model = FuXiLowerRes(model_config).to(device)

    smoke_inputs = _make_main_random_inputs(
        model_config,
        batch_size=train_config.random_smoke_batch_size,
        device=device,
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

    pin_memory = device.type == "cuda"
    train_loader, val_loader = _build_split_dataloaders(
        data_config,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        train_start_time=train_config.train_start_time,
        train_end_time=train_config.train_end_time,
        val_start_time=train_config.val_start_time,
        val_end_time=train_config.val_end_time,
        pin_memory=pin_memory,
    )

    optimizer = AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    history: list[dict[str, float]] = []
    output_dir = train_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, train_config.max_epochs + 1):
        model.train()
        running_loss = 0.0
        train_steps = 0

        for batch_index, batch in _iter_limited(train_loader, train_config.max_train_batches):
            batch = _move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _amp_autocast_context(train_config.use_amp, device, amp_dtype):
                outputs = model(batch["x"], batch["temb"], static_features=batch["static_features"])
                loss = criterion(outputs["forecast"], batch["target"])
            loss.backward()
            if train_config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clip_norm)
            optimizer.step()

            running_loss += float(loss.item())
            train_steps += 1
            if (batch_index + 1) % max(train_config.log_every, 1) == 0:
                print(f"[main][epoch {epoch}] batch {batch_index + 1} loss={loss.item():.6f}")

        train_loss = running_loss / max(train_steps, 1)
        val_loss: float | None = None

        if val_loader is not None:
            model.eval()
            val_running_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for _batch_index, batch in _iter_limited(val_loader, train_config.max_val_batches):
                    batch = _move_batch_to_device(batch, device)
                    with _amp_autocast_context(train_config.use_amp, device, amp_dtype):
                        outputs = model(batch["x"], batch["temb"], static_features=batch["static_features"])
                        loss = criterion(outputs["forecast"], batch["target"])
                    val_running_loss += float(loss.item())
                    val_steps += 1
            val_loss = val_running_loss / max(val_steps, 1)

        epoch_record = {"epoch": float(epoch), "train_loss": train_loss}
        if val_loss is not None:
            epoch_record["val_loss"] = val_loss
        history.append(epoch_record)
        print(f"[main][epoch {epoch}] train_loss={train_loss:.6f}" + (f" val_loss={val_loss:.6f}" if val_loss is not None else ""))

        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "decoder_state_dict": model.decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "train_config": _to_plain_data(asdict(train_config)),
            "model_config": _to_plain_data(asdict(model_config)),
            "data_config": _to_plain_data(asdict(data_config)),
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
    }
    _print_json_block("main_training_result", result)
    return result


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
        device,
        _model_dtype,
        amp_dtype,
    ) = _build_intrinsic_training_objects(config_path)

    encoder = FuXiLowerResEncoder(encoder_config).to(device)
    if train_config.main_checkpoint_path is not None:
        _load_encoder_checkpoint(encoder, train_config.main_checkpoint_path)
    elif not smoke_only:
        raise FileNotFoundError(
            "train_intrinsic.main_checkpoint_path is required for intrinsic training after the main model."
        )
    encoder.eval()
    encoder.requires_grad_(False)

    intrinsic_model = FuXiIntrinsic(intrinsic_config).to(device)

    smoke_inputs = _make_main_random_inputs(
        encoder_config,
        batch_size=train_config.random_smoke_batch_size,
        device=device,
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
            device=device,
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

    pin_memory = device.type == "cuda"
    train_loader, val_loader = _build_split_dataloaders(
        data_config,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
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
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    history: list[dict[str, float]] = []
    output_dir = train_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, train_config.max_epochs + 1):
        intrinsic_model.train()
        running_loss = 0.0
        train_steps = 0

        for batch_index, batch in _iter_limited(train_loader, train_config.max_train_batches):
            batch = _move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                encoded = encoder(batch["x"], batch["temb"], static_features=batch["static_features"])
                z_high = encoded.z_high.detach() if train_config.detach_z_high else encoded.z_high
            with _amp_autocast_context(train_config.use_amp, device, amp_dtype):
                outputs = intrinsic_model(z_high)
                loss = criterion(outputs["z_high_recon"], z_high)
            loss.backward()
            if train_config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(intrinsic_model.parameters(), train_config.gradient_clip_norm)
            optimizer.step()

            running_loss += float(loss.item())
            train_steps += 1
            if (batch_index + 1) % max(train_config.log_every, 1) == 0:
                print(f"[intrinsic][epoch {epoch}] batch {batch_index + 1} loss={loss.item():.6f}")

        train_loss = running_loss / max(train_steps, 1)
        val_loss: float | None = None

        if val_loader is not None:
            intrinsic_model.eval()
            val_running_loss = 0.0
            val_steps = 0
            with torch.no_grad():
                for _batch_index, batch in _iter_limited(val_loader, train_config.max_val_batches):
                    batch = _move_batch_to_device(batch, device)
                    encoded = encoder(batch["x"], batch["temb"], static_features=batch["static_features"])
                    z_high = encoded.z_high.detach() if train_config.detach_z_high else encoded.z_high
                    with _amp_autocast_context(train_config.use_amp, device, amp_dtype):
                        outputs = intrinsic_model(z_high)
                        loss = criterion(outputs["z_high_recon"], z_high)
                    val_running_loss += float(loss.item())
                    val_steps += 1
            val_loss = val_running_loss / max(val_steps, 1)

        epoch_record = {"epoch": float(epoch), "train_loss": train_loss}
        if val_loss is not None:
            epoch_record["val_loss"] = val_loss
        history.append(epoch_record)
        print(
            f"[intrinsic][epoch {epoch}] train_loss={train_loss:.6f}"
            + (f" val_loss={val_loss:.6f}" if val_loss is not None else "")
        )

        checkpoint_payload = {
            "epoch": epoch,
            "intrinsic_state_dict": intrinsic_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "train_config": _to_plain_data(asdict(train_config)),
            "encoder_config": _to_plain_data(asdict(encoder_config)),
            "intrinsic_config": _to_plain_data(asdict(intrinsic_config)),
            "data_config": _to_plain_data(asdict(data_config)),
            "main_checkpoint_path": str(train_config.main_checkpoint_path) if train_config.main_checkpoint_path else None,
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
    }
    _print_json_block("intrinsic_training_result", result)
    return result


__all__ = [
    "IntrinsicTrainingConfig",
    "MainTrainingConfig",
    "run_intrinsic_model_smoke_test",
    "run_main_model_smoke_test",
    "train_intrinsic_model",
    "train_main_model",
]
