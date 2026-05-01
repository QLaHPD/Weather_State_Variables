from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
import csv
from dataclasses import dataclass, replace
import gc
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch import Tensor
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from .config import DEFAULT_MODEL_CONFIG_PATH, load_yaml_config, resolve_repo_path
from .data import (
    ArcoEra5FuXiDataConfig,
    ArcoEra5NormalizationStats,
    FUXI_PRESSURE_LEVELS,
    build_fuxi_derived_static_maps,
    build_fuxi_channel_names,
    open_arco_era5_dataset,
    prepare_arco_spatial_dataarray,
    specific_humidity_to_relative_humidity,
)
from .models import FuXiLowerRes, FuXiLowerResConfig
from .models.fuxi_lower_res import FuXiEncoderOutput
from .models.fuxi_short import build_fuxi_time_embeddings
from .training.pipeline import _load_main_forecast_checkpoint


DEFAULT_EARTH2STUDIO_MODEL_CACHE = Path("/mnt/raid0/weather_state_variables/models")
DEFAULT_BENCHMARK_OUTPUT_DIR = Path("runs/benchmark_compare")
_HIGHRES_SIZE = (721, 1440)
_LOWRES_SIZE = (181, 360)

_COMMON_LEVEL_PREFIXES = ("z", "t", "u", "v")
_COMMON_SURFACE_VARIABLES = ("t2m", "u10m", "v10m", "msl")

_CANONICAL_UPPER_AIR_SOURCES = {
    "z": "geopotential",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "r": "relative_humidity",
    "q": "specific_humidity",
    "w": "vertical_velocity",
}
_CANONICAL_SURFACE_SOURCES = {
    "t2m": "2m_temperature",
    "u10m": "10m_u_component_of_wind",
    "v10m": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
}
_OUR_SURFACE_CANONICALS = {
    "T2M": "t2m",
    "U10": "u10m",
    "V10": "v10m",
    "MSL": "msl",
    "TP": "tp",
}


class _NullProgressBar:
    def update(self, _increment: int = 1) -> None:
        return

    def set_postfix_str(self, _value: str, refresh: bool = True) -> None:
        return

    def set_description_str(self, _value: str, refresh: bool = True) -> None:
        return

    def refresh(self) -> None:
        return

    def close(self) -> None:
        return


def _create_progress_bar(
    *,
    total: int,
    description: str,
    unit: str = "step",
) -> Any:
    if tqdm is None:
        return _NullProgressBar()
    return tqdm(
        total=int(total),
        desc=str(description),
        unit=str(unit),
        dynamic_ncols=True,
        leave=True,
        file=sys.stdout,
        mininterval=0.25,
        disable=False,
    )


def _run_loading_stage(
    description: str,
    loader: Callable[[], Any],
) -> Any:
    progress_bar = _create_progress_bar(total=1, description=description, unit="stage")
    progress_bar.refresh()
    try:
        result = loader()
        progress_bar.update(1)
        return result
    finally:
        progress_bar.close()


def _progress_description(model_name: str, *, native_step_hours: int, horizon_hours: int) -> str:
    day_count = horizon_hours / 24.0
    if float(day_count).is_integer():
        horizon_label = f"{int(day_count)}d"
    else:
        horizon_label = f"{day_count:g}d"
    return f"{model_name} ({native_step_hours}h x {horizon_label})"


def _progress_postfix(
    *,
    init_index: int,
    init_count: int,
    init_time: pd.Timestamp,
) -> str:
    return f"init {init_index}/{init_count} @ {pd.Timestamp(init_time).strftime('%Y-%m-%d %H:%M')}"


def _rollout_step_count(max_lead_hours: int, native_step_hours: int) -> int:
    if max_lead_hours <= 0:
        return 0
    if native_step_hours <= 0:
        raise ValueError(f"native_step_hours must be positive, got {native_step_hours}")
    return int(max_lead_hours // native_step_hours)


@dataclass(frozen=True)
class ConservativeNestedLatLonRemapper:
    src_lat: np.ndarray
    src_lon: np.ndarray
    dst_lat: np.ndarray
    dst_lon: np.ndarray
    lat_weights: np.ndarray
    lon_weights: np.ndarray

    @classmethod
    def from_grids(
        cls,
        *,
        src_lat: Sequence[float],
        src_lon: Sequence[float],
        dst_lat: Sequence[float],
        dst_lon: Sequence[float],
    ) -> "ConservativeNestedLatLonRemapper":
        src_lat_array = np.asarray(src_lat, dtype=np.float64)
        src_lon_array = np.asarray(src_lon, dtype=np.float64)
        dst_lat_array = np.asarray(dst_lat, dtype=np.float64)
        dst_lon_array = np.asarray(dst_lon, dtype=np.float64)
        return cls(
            src_lat=src_lat_array,
            src_lon=src_lon_array,
            dst_lat=dst_lat_array,
            dst_lon=dst_lon_array,
            lat_weights=_latitude_overlap_weights(src_lat_array, dst_lat_array),
            lon_weights=_longitude_overlap_weights(src_lon_array, dst_lon_array),
        )

    def remap(self, values: np.ndarray) -> np.ndarray:
        if values.ndim < 2:
            raise ValueError(
                f"Expected values with latitude/longitude trailing dimensions, got {values.shape}"
            )
        if tuple(values.shape[-2:]) != (len(self.src_lat), len(self.src_lon)):
            raise ValueError(
                "Expected trailing grid "
                f"{(len(self.src_lat), len(self.src_lon))}, got {tuple(values.shape[-2:])}"
            )
        remapped = np.einsum(
            "ab,...bc,cd->...ad",
            self.lat_weights,
            values.astype(np.float64, copy=False),
            self.lon_weights.T,
            optimize=True,
        )
        return remapped.astype(np.float32, copy=False)


@dataclass
class MetricAccumulator:
    lead_hours: int
    model_name: str
    resolution_group: str
    variable_name: str | None = None
    absolute_error_sum: float = 0.0
    squared_error_sum: float = 0.0
    count: int = 0

    def update(
        self,
        prediction: np.ndarray,
        truth: np.ndarray,
        *,
        latitude_weights: np.ndarray,
    ) -> None:
        if prediction.shape != truth.shape:
            raise ValueError(
                f"Prediction/truth shape mismatch: {prediction.shape} vs {truth.shape}"
            )
        if prediction.ndim != 3:
            raise ValueError(
                f"Expected [C, H, W] tensors for metric accumulation, got {prediction.shape}"
            )
        if latitude_weights.shape != (prediction.shape[-2],):
            raise ValueError(
                "Latitude-weight vector must match the latitude dimension. Got "
                f"{latitude_weights.shape} for {prediction.shape}"
            )

        error = prediction.astype(np.float64, copy=False) - truth.astype(np.float64, copy=False)
        weights = latitude_weights[None, :, None]
        self.absolute_error_sum += float(np.abs(error * weights).sum())
        self.squared_error_sum += float(np.square(error * weights).sum())
        self.count += int(error.size)

    def summary(self) -> dict[str, Any]:
        if self.count <= 0:
            mae = float("nan")
            rmse = float("nan")
        else:
            mae = self.absolute_error_sum / float(self.count)
            rmse = math.sqrt(self.squared_error_sum / float(self.count))
        return {
            "lead_hours": int(self.lead_hours),
            "model_name": self.model_name,
            "resolution_group": self.resolution_group,
            "variable_name": self.variable_name,
            "mae": float(mae),
            "rmse": float(rmse),
            "count": int(self.count),
        }


class ChannelNormalizer:
    def __init__(self, stats: ArcoEra5NormalizationStats) -> None:
        self.stats = stats
        self.dynamic_channel_names = tuple(str(name).lower() for name in stats.dynamic_channel_names)
        self.static_channel_names = tuple(str(name) for name in stats.static_channel_names)
        self.dynamic_name_to_index = {
            name: index for index, name in enumerate(self.dynamic_channel_names)
        }
        self.static_name_to_index = {
            name: index for index, name in enumerate(self.static_channel_names)
        }
        self.dynamic_mean = np.asarray(stats.dynamic_mean, dtype=np.float32)
        self.dynamic_std = np.asarray(stats.dynamic_std, dtype=np.float32)
        self.static_mean = np.asarray(stats.static_mean, dtype=np.float32)
        self.static_std = np.asarray(stats.static_std, dtype=np.float32)
        self.dynamic_name_aliases = {
            "u10m": "u10",
            "v10m": "v10",
            "tp06": "tp",
            "tp1h": "tp",
        }

    @classmethod
    def from_json(cls, stats_path: str | Path) -> "ChannelNormalizer":
        payload = json.loads(Path(stats_path).read_text(encoding="utf-8"))
        return cls(ArcoEra5NormalizationStats.from_dict(payload))

    @staticmethod
    def _apply_transform(values: np.ndarray, kind: str) -> np.ndarray:
        if kind in {"zscore", "identity"}:
            return values
        if kind == "log1p_mm_zscore":
            np.maximum(values, 0.0, out=values)
            values *= 1000.0
            np.log1p(values, out=values)
            return values
        raise ValueError(f"Unsupported normalization transform kind: {kind}")

    @staticmethod
    def _invert_transform(values: np.ndarray, kind: str) -> np.ndarray:
        if kind in {"zscore", "identity"}:
            return values
        if kind == "log1p_mm_zscore":
            np.expm1(values, out=values)
            values /= 1000.0
            return values
        raise ValueError(f"Unsupported normalization transform kind: {kind}")

    def _dynamic_stats_index(self, variable_name: str) -> int:
        normalized_name = str(variable_name).lower()
        stats_name = self.dynamic_name_aliases.get(normalized_name, normalized_name)
        try:
            return self.dynamic_name_to_index[stats_name]
        except KeyError as exc:
            raise KeyError(
                "Normalization stats do not contain a channel matching "
                f"{variable_name!r} (resolved lookup {stats_name!r}). "
                f"Available channels include: {list(self.dynamic_channel_names[:10])}"
            ) from exc

    def normalize_dynamic(
        self,
        values: np.ndarray,
        *,
        canonical_variable_names: Sequence[str],
    ) -> np.ndarray:
        if values.ndim != 4:
            raise ValueError(f"Expected [T, C, H, W], got {values.shape}")
        if values.shape[1] != len(canonical_variable_names):
            raise ValueError(
                "Canonical variable name count does not match channel dimension. "
                f"Got {len(canonical_variable_names)} names for {values.shape}"
            )
        normalized = values.astype(np.float32, copy=True)
        for channel_index, variable_name in enumerate(canonical_variable_names):
            stats_index = self._dynamic_stats_index(str(variable_name))
            kind = self.stats.dynamic_transform_kinds[stats_index]
            normalized[:, channel_index] = self._apply_transform(normalized[:, channel_index], kind)
            if kind != "identity":
                normalized[:, channel_index] = (
                    normalized[:, channel_index] - self.dynamic_mean[stats_index]
                ) / self.dynamic_std[stats_index]
        return normalized

    def denormalize_dynamic(
        self,
        values: np.ndarray,
        *,
        canonical_variable_names: Sequence[str],
    ) -> np.ndarray:
        if values.ndim not in {3, 4}:
            raise ValueError(f"Expected [C, H, W] or [T, C, H, W], got {values.shape}")
        squeeze_time = values.ndim == 3
        restored = values.astype(np.float32, copy=True)
        if squeeze_time:
            restored = restored[None]
        for channel_index, variable_name in enumerate(canonical_variable_names):
            stats_index = self._dynamic_stats_index(str(variable_name))
            kind = self.stats.dynamic_transform_kinds[stats_index]
            if kind != "identity":
                restored[:, channel_index] = (
                    restored[:, channel_index] * self.dynamic_std[stats_index]
                    + self.dynamic_mean[stats_index]
                )
            restored[:, channel_index] = self._invert_transform(restored[:, channel_index], kind)
        return restored[0] if squeeze_time else restored

    def normalize_static(
        self,
        values: np.ndarray,
        *,
        static_variable_names: Sequence[str],
    ) -> np.ndarray:
        if values.ndim != 3:
            raise ValueError(f"Expected [C, H, W] static tensor, got {values.shape}")
        if values.shape[0] != len(static_variable_names):
            raise ValueError(
                "Static variable name count does not match channel dimension. "
                f"Got {len(static_variable_names)} names for {values.shape}"
            )
        normalized = values.astype(np.float32, copy=True)
        for channel_index, variable_name in enumerate(static_variable_names):
            stats_index = self.static_name_to_index[str(variable_name)]
            kind = self.stats.static_transform_kinds[stats_index]
            normalized[channel_index] = self._apply_transform(normalized[channel_index], kind)
            if kind != "identity":
                normalized[channel_index] = (
                    normalized[channel_index] - self.static_mean[stats_index]
                ) / self.static_std[stats_index]
        return normalized


class _SingleDeviceMainModelRunner:
    def __init__(
        self,
        model: FuXiLowerRes,
        *,
        device: torch.device,
        amp_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.amp_dtype = amp_dtype

    @property
    def device_assignment(self) -> tuple[torch.device, ...]:
        return (self.device,)

    def predict_next(
        self,
        x: Tensor,
        temb: Tensor,
        *,
        static_features: Tensor | None = None,
    ) -> Tensor:
        model_inputs = x.to(device=self.device, dtype=torch.float32, non_blocking=True)
        model_temb = temb.to(device=self.device, dtype=torch.float32, non_blocking=True)
        model_static = (
            None
            if static_features is None
            else static_features.to(device=self.device, dtype=torch.float32, non_blocking=True)
        )
        with torch.inference_mode():
            with _autocast_context(self.device, self.amp_dtype):
                prediction = self.model.predict_next(
                    model_inputs,
                    model_temb,
                    static_features=model_static,
                )
        return prediction.float()


class _ShardedMainModelRunner:
    def __init__(
        self,
        model: FuXiLowerRes,
        *,
        devices: Sequence[torch.device],
        amp_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if len(devices) <= 1:
            raise ValueError("Sharded main-model runner needs at least two devices.")
        self.model = model
        self.devices = tuple(torch.device(device) for device in devices)
        self.amp_dtype = amp_dtype
        self.chunk_devices = self._assign_chunk_devices(self.devices)
        self._move_model_shards()

    @property
    def device_assignment(self) -> tuple[torch.device, ...]:
        return self.devices

    @staticmethod
    def _assign_chunk_devices(devices: Sequence[torch.device]) -> dict[str, torch.device]:
        chunk_names = (
            "enc_embed",
            "enc_down",
            "enc_stage0",
            "enc_stage1",
            "dec_stage0",
            "dec_stage1",
            "dec_rest",
        )
        assignments: dict[str, torch.device] = {}
        chunk_indices = np.array_split(np.arange(len(chunk_names)), len(devices))
        for device, indices in zip(devices, chunk_indices, strict=True):
            for index in indices.tolist():
                assignments[chunk_names[int(index)]] = torch.device(device)
        return assignments

    def _move_model_shards(self) -> None:
        encoder = self.model.encoder
        decoder = self.model.decoder

        embed_device = self.chunk_devices["enc_embed"]
        encoder.patch_embed.to(embed_device)
        encoder.time_embed.to(embed_device)
        encoder.default_static_features = encoder.default_static_features.to(embed_device)

        encoder.downsample.to(self.chunk_devices["enc_down"])
        encoder.down_resblock.to(self.chunk_devices["enc_down"])
        encoder.first_pair_layers[0].to(self.chunk_devices["enc_stage0"])
        encoder.first_pair_layers[1].to(self.chunk_devices["enc_stage1"])

        decoder.second_pair_layers[0].to(self.chunk_devices["dec_stage0"])
        decoder.second_pair_layers[1].to(self.chunk_devices["dec_stage1"])
        decoder.second_pair_fusion.to(self.chunk_devices["dec_rest"])
        decoder.up_resblock.to(self.chunk_devices["dec_rest"])
        decoder.upsample.to(self.chunk_devices["dec_rest"])
        decoder.head.to(self.chunk_devices["dec_rest"])

    def predict_next(
        self,
        x: Tensor,
        temb: Tensor,
        *,
        static_features: Tensor | None = None,
    ) -> Tensor:
        encoder = self.model.encoder
        decoder = self.model.decoder
        original_size = tuple(int(value) for value in x.shape[-2:])

        embed_device = self.chunk_devices["enc_embed"]
        down_device = self.chunk_devices["enc_down"]
        enc_stage0_device = self.chunk_devices["enc_stage0"]
        enc_stage1_device = self.chunk_devices["enc_stage1"]
        dec_stage0_device = self.chunk_devices["dec_stage0"]
        dec_stage1_device = self.chunk_devices["dec_stage1"]
        dec_rest_device = self.chunk_devices["dec_rest"]

        x = x.to(device=embed_device, dtype=torch.float32, non_blocking=True)
        temb = temb.to(device=embed_device, dtype=torch.float32, non_blocking=True)
        static_features = (
            None
            if static_features is None
            else static_features.to(device=embed_device, dtype=torch.float32, non_blocking=True)
        )

        with torch.inference_mode():
            with _autocast_context(embed_device, self.amp_dtype):
                x_resized = encoder._resize_steps(x, encoder.config.resized_input_size)
                static = encoder._prepare_static_features(x.shape[0], static_features)
                patch_tokens = encoder.patch_embed(
                    x_resized.to(dtype=encoder._model_dtype()),
                    static,
                )
                patch_grid_features = patch_tokens.permute(0, 3, 1, 2)
                temb_emb = encoder.time_embed(temb.to(dtype=encoder._model_dtype()))

            patch_grid_features = patch_grid_features.to(
                device=down_device,
                dtype=torch.float32,
                non_blocking=True,
            )
            temb_emb = temb_emb.to(device=down_device, dtype=torch.float32, non_blocking=True)
            with _autocast_context(down_device, self.amp_dtype):
                hidden = encoder.downsample(patch_grid_features.to(dtype=encoder._model_dtype()))
                hidden = encoder.down_resblock(hidden, temb_emb.to(dtype=encoder._model_dtype()))

            hidden = hidden.permute(0, 2, 3, 1).to(
                device=enc_stage0_device,
                dtype=torch.float32,
                non_blocking=True,
            )
            with _autocast_context(enc_stage0_device, self.amp_dtype):
                hidden = encoder.first_pair_layers[0](hidden)

            hidden = hidden.to(device=enc_stage1_device, dtype=torch.float32, non_blocking=True)
            with _autocast_context(enc_stage1_device, self.amp_dtype):
                hidden = encoder.first_pair_layers[1](hidden)

            encoded = FuXiEncoderOutput(
                second_block_features=hidden.permute(0, 3, 1, 2),
                output_size=original_size,
            )

            decoder_hidden = encoded.second_block_features.to(
                device=dec_stage0_device,
                dtype=torch.float32,
                non_blocking=True,
            ).permute(0, 2, 3, 1)
            with _autocast_context(dec_stage0_device, self.amp_dtype):
                s2 = decoder.second_pair_layers[0](decoder_hidden)

            s2_for_stage1 = s2.to(device=dec_stage1_device, dtype=torch.float32, non_blocking=True)
            with _autocast_context(dec_stage1_device, self.amp_dtype):
                s3 = decoder.second_pair_layers[1](s2_for_stage1)

            s2 = s2.to(device=dec_rest_device, dtype=torch.float32, non_blocking=True)
            s3 = s3.to(device=dec_rest_device, dtype=torch.float32, non_blocking=True)
            with _autocast_context(dec_rest_device, self.amp_dtype):
                fused = decoder.second_pair_fusion(torch.cat([s2, s3], dim=-1))
                decoder_map = fused.permute(0, 3, 1, 2)
                decoder_map = decoder.up_resblock(decoder_map)
                decoder_map = decoder.upsample(decoder_map)

                decoder_tokens = decoder_map.permute(0, 2, 3, 1)
                decoder_tokens = decoder.head(decoder_tokens)
                batch, height_bins, width_bins, _ = decoder_tokens.shape
                patch_height, patch_width = decoder.config.patch_size
                decoder_tokens = decoder_tokens.reshape(
                    batch,
                    height_bins,
                    width_bins,
                    decoder.config.forecast_steps,
                    patch_height,
                    patch_width,
                    decoder.config.out_chans,
                )
                decoder_tokens = decoder_tokens.permute(0, 3, 6, 1, 4, 2, 5)
                decoder_tokens = decoder_tokens.reshape(
                    batch,
                    decoder.config.forecast_steps,
                    decoder.config.out_chans,
                    height_bins * patch_height,
                    width_bins * patch_width,
                )
                forecast = decoder._resize_future_maps(decoder_tokens, encoded.output_size)
        return forecast[:, 0].float()


class BenchmarkEra5Source:
    def __init__(
        self,
        dataset_path: str | Path,
        *,
        gcs_token: str = "anon",
        latitude_descending: bool = True,
        humidity_source: str = "auto",
        orography_source: str = "geopotential_at_surface",
        convert_geopotential_to_height: bool = True,
    ) -> None:
        self.dataset_path = str(dataset_path)
        self.gcs_token = gcs_token
        self.latitude_descending = bool(latitude_descending)
        self.humidity_source = str(humidity_source)
        self.orography_source = str(orography_source)
        self.convert_geopotential_to_height = bool(convert_geopotential_to_height)

        self.dataset = open_arco_era5_dataset(dataset_path, gcs_token=gcs_token)
        self.time_values = np.asarray(self.dataset["time"].values)
        self.time_index = {
            pd.Timestamp(value).value: index for index, value in enumerate(self.time_values.tolist())
        }
        self.latitudes = np.asarray(
            prepare_arco_spatial_dataarray(
                self.dataset["latitude"],
                latitude_descending=self.latitude_descending,
            ).values,
            dtype=np.float32,
        )
        self.longitudes = np.asarray(
            prepare_arco_spatial_dataarray(
                self.dataset["longitude"],
                latitude_descending=self.latitude_descending,
            ).values,
            dtype=np.float32,
        )
        if len(self.time_values) < 2:
            raise ValueError("Benchmark dataset must contain at least two time steps.")
        time_delta = pd.Timestamp(self.time_values[1]) - pd.Timestamp(self.time_values[0])
        self.time_step_hours = int(time_delta / pd.Timedelta(hours=1))
        self._static_cache: dict[tuple[str, ...], np.ndarray] = {}

    def time_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return pd.Timestamp(self.time_values[0]), pd.Timestamp(self.time_values[-1])

    def resolve_time_index(self, timestamp: pd.Timestamp | str) -> int:
        key = pd.Timestamp(timestamp).value
        try:
            return self.time_index[key]
        except KeyError as exc:
            raise KeyError(f"Timestamp {pd.Timestamp(timestamp)} is not present in {self.dataset_path}") from exc

    def canonical_absolute_times(
        self,
        init_time: pd.Timestamp,
        lead_hours: Sequence[int],
    ) -> list[pd.Timestamp]:
        return [pd.Timestamp(init_time) + pd.Timedelta(hours=int(hour)) for hour in lead_hours]

    def build_static_stack(self, static_variable_names: Sequence[str]) -> np.ndarray:
        key = tuple(str(name) for name in static_variable_names)
        cached = self._static_cache.get(key)
        if cached is not None:
            return cached.copy()

        land_sea_mask = prepare_arco_spatial_dataarray(
            self.dataset["land_sea_mask"],
            latitude_descending=self.latitude_descending,
        ).load().values.astype(np.float32)
        orography = prepare_arco_spatial_dataarray(
            self.dataset[self.orography_source],
            latitude_descending=self.latitude_descending,
        ).load().values.astype(np.float32)
        if self.convert_geopotential_to_height and self.orography_source == "geopotential_at_surface":
            orography = orography / 9.80665

        derived_maps = build_fuxi_derived_static_maps(self.latitudes, self.longitudes)
        all_maps = {
            "land_sea_mask": land_sea_mask,
            "orography": orography,
            **derived_maps,
        }
        stacked = np.stack([all_maps[name] for name in key], axis=0).astype(np.float32, copy=False)
        self._static_cache[key] = stacked
        return stacked.copy()

    def load_canonical_cube(
        self,
        absolute_times: Sequence[pd.Timestamp | str],
        canonical_variable_names: Sequence[str],
    ) -> np.ndarray:
        if len(absolute_times) <= 0:
            raise ValueError("absolute_times must be non-empty.")
        if len(canonical_variable_names) <= 0:
            raise ValueError("canonical_variable_names must be non-empty.")

        resolved_times = [pd.Timestamp(value) for value in absolute_times]
        time_indices = [self.resolve_time_index(value) for value in resolved_times]
        specs = [_parse_canonical_variable_name(name) for name in canonical_variable_names]

        level_requests: dict[str, set[int]] = {}
        needs_relative_humidity = False
        for spec in specs:
            if spec["kind"] == "upper_air":
                level_requests.setdefault(spec["source"], set()).add(int(spec["level"]))
                if spec["source"] == "relative_humidity":
                    needs_relative_humidity = True

        if needs_relative_humidity and "relative_humidity" not in self.dataset.data_vars:
            level_requests.setdefault("specific_humidity", set()).update(
                spec["level"]
                for spec in specs
                if spec["kind"] == "upper_air" and spec["source"] == "relative_humidity"
            )
            level_requests.setdefault("temperature", set()).update(
                spec["level"]
                for spec in specs
                if spec["kind"] == "upper_air" and spec["source"] == "relative_humidity"
            )

        loaded_upper_air: dict[str, xr.DataArray] = {}
        for source_name, requested_levels in level_requests.items():
            if source_name == "relative_humidity" and "relative_humidity" not in self.dataset.data_vars:
                continue
            loaded_upper_air[source_name] = prepare_arco_spatial_dataarray(
                self.dataset[source_name]
                .isel(time=time_indices)
                .sel(level=sorted(int(level) for level in requested_levels)),
                latitude_descending=self.latitude_descending,
            ).load()

        derived_relative_humidity: xr.DataArray | None = None
        if needs_relative_humidity and "relative_humidity" not in self.dataset.data_vars:
            specific_humidity = loaded_upper_air["specific_humidity"]
            temperature = loaded_upper_air["temperature"]
            derived_relative_humidity = specific_humidity_to_relative_humidity(
                specific_humidity,
                temperature,
                temperature["level"],
            )

        loaded_surface: dict[str, xr.DataArray] = {}
        if any(spec["kind"] == "surface" for spec in specs):
            surface_sources = sorted(
                {
                    spec["source"]
                    for spec in specs
                    if spec["kind"] == "surface" and spec["source"] != "total_precipitation"
                }
            )
            for source_name in surface_sources:
                loaded_surface[source_name] = prepare_arco_spatial_dataarray(
                    self.dataset[source_name].isel(time=time_indices),
                    latitude_descending=self.latitude_descending,
                ).load()

        accumulation_cache: dict[int, np.ndarray] = {}
        cube = np.empty(
            (len(resolved_times), len(canonical_variable_names), len(self.latitudes), len(self.longitudes)),
            dtype=np.float32,
        )
        for channel_index, spec in enumerate(specs):
            if spec["kind"] == "upper_air":
                if spec["source"] == "relative_humidity" and derived_relative_humidity is not None:
                    source_array = derived_relative_humidity
                else:
                    source_array = loaded_upper_air[spec["source"]]
                level_values = source_array["level"].values.tolist()
                level_position = level_values.index(int(spec["level"]))
                cube[:, channel_index] = source_array.values[:, level_position].astype(np.float32, copy=False)
                continue

            if spec["source"] == "total_precipitation":
                accumulation_hours = int(spec["accumulation_hours"])
                if accumulation_hours not in accumulation_cache:
                    accumulation_cache[accumulation_hours] = self._load_tp_accumulation(
                        time_indices=time_indices,
                        accumulation_hours=accumulation_hours,
                    )
                cube[:, channel_index] = accumulation_cache[accumulation_hours]
                continue

            cube[:, channel_index] = loaded_surface[spec["source"]].values.astype(np.float32, copy=False)

        return cube

    def _load_tp_accumulation(
        self,
        *,
        time_indices: Sequence[int],
        accumulation_hours: int,
    ) -> np.ndarray:
        if accumulation_hours <= 0:
            raise ValueError(f"accumulation_hours must be positive, got {accumulation_hours}")
        if accumulation_hours % self.time_step_hours != 0:
            raise ValueError(
                f"Cannot build {accumulation_hours}h precipitation accumulation from "
                f"{self.time_step_hours}h data."
            )
        accumulation_steps = accumulation_hours // self.time_step_hours
        required_start = min(int(index) for index in time_indices) - accumulation_steps + 1
        if required_start < 0:
            raise ValueError(
                f"Not enough history in {self.dataset_path} to build trailing {accumulation_hours}h accumulation."
            )
        source = prepare_arco_spatial_dataarray(
            self.dataset["total_precipitation"].isel(
                time=slice(required_start, max(int(index) for index in time_indices) + 1)
            ),
            latitude_descending=self.latitude_descending,
        ).load()
        accumulations: list[np.ndarray] = []
        for absolute_index in time_indices:
            relative_stop = int(absolute_index) - required_start + 1
            relative_start = relative_stop - accumulation_steps
            accumulations.append(
                source.values[relative_start:relative_stop].sum(axis=0, dtype=np.float32)
            )
        return np.stack(accumulations, axis=0).astype(np.float32, copy=False)


def download_earth2studio_models(
    *,
    model_cache_dir: str | Path = DEFAULT_EARTH2STUDIO_MODEL_CACHE,
    model_names: Sequence[str] = ("fuxi", "pangu24", "graphcastsmall"),
    device: str = "cpu",
) -> dict[str, Any]:
    px = _import_earth2studio_px()
    resolved_cache_dir = Path(model_cache_dir).expanduser().resolve()
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["EARTH2STUDIO_CACHE"] = str(resolved_cache_dir)

    report: list[dict[str, Any]] = []
    resolved_device = _resolve_device(device)
    for model_name in model_names:
        model_cls = _earth2studio_model_class(px, model_name)
        model = model_cls.from_pretrained()
        try:
            if hasattr(model, "to"):
                model = model.to(resolved_device)
            input_coords = _model_input_coords(model)
            report.append(
                {
                    "model_name": str(model_name),
                    "device": str(resolved_device),
                    "cache_dir": str(resolved_cache_dir),
                    "input_variable_count": int(len(np.asarray(input_coords["variable"]))),
                    "input_lat_size": int(len(np.asarray(input_coords["lat"]))),
                    "input_lon_size": int(len(np.asarray(input_coords["lon"]))),
                    "lead_time_hours": _lead_time_hours_list(input_coords["lead_time"]),
                }
            )
        finally:
            _release_runtime_models(model)

    return {
        "model_cache_dir": str(resolved_cache_dir),
        "models": report,
    }


def compare_forecast_models(
    config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
    *,
    data_path: str | Path,
    main_checkpoint_path: str | Path,
    output_dir: str | Path = DEFAULT_BENCHMARK_OUTPUT_DIR,
    model_cache_dir: str | Path = DEFAULT_EARTH2STUDIO_MODEL_CACHE,
    device: str = "auto",
    device_map: dict[str, str | torch.device] | None = None,
    start_time: str | pd.Timestamp | None = None,
    end_time: str | pd.Timestamp | None = None,
    init_stride_hours: int = 24,
    horizon_hours: int = 120,
    max_init_times: int | None = None,
    normalization_stats_path: str | Path | None = None,
    highres_metric_step_hours: int = 24,
    lowres_metric_step_hours: int = 6,
    download_models_first: bool = True,
    allow_cpu_fallback: bool = True,
) -> dict[str, Any]:
    if horizon_hours <= 0:
        raise ValueError(f"horizon_hours must be positive, got {horizon_hours}")
    if init_stride_hours <= 0:
        raise ValueError(f"init_stride_hours must be positive, got {init_stride_hours}")
    if highres_metric_step_hours <= 0 or lowres_metric_step_hours <= 0:
        raise ValueError("Metric lead-time step sizes must be positive.")

    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_checkpoint_path = resolve_repo_path(main_checkpoint_path, config_path=config_path)
    resolved_data_path = resolve_repo_path(data_path, config_path=config_path)
    resolved_model_cache = Path(model_cache_dir).expanduser().resolve()
    resolved_model_cache.mkdir(parents=True, exist_ok=True)

    data_config = ArcoEra5FuXiDataConfig.from_yaml(config_path)

    stats_path = _resolve_normalization_stats_path(
        normalization_stats_path,
        config_path=config_path,
        config_data_path=data_config.normalization_stats_path,
    )
    if data_config.apply_normalization:
        if stats_path is None or not stats_path.exists():
            raise FileNotFoundError(
                "The benchmark needs an existing normalization stats file for the main model. "
                f"Expected {stats_path}."
            )
        normalizer = ChannelNormalizer.from_json(stats_path)
    else:
        normalizer = None

    source = BenchmarkEra5Source(
        resolved_data_path,
        gcs_token=data_config.gcs_token,
        latitude_descending=data_config.latitude_descending,
        humidity_source=data_config.humidity_source,
        orography_source=data_config.orography_source,
        convert_geopotential_to_height=data_config.convert_geopotential_to_height,
    )

    if source.latitudes.shape != (_HIGHRES_SIZE[0],):
        raise ValueError(
            f"Expected benchmark latitude size {_HIGHRES_SIZE[0]}, got {len(source.latitudes)}"
        )
    if source.longitudes.shape != (_HIGHRES_SIZE[1],):
        raise ValueError(
            f"Expected benchmark longitude size {_HIGHRES_SIZE[1]}, got {len(source.longitudes)}"
        )

    lowres_lat = np.linspace(90.0, -90.0, _LOWRES_SIZE[0], endpoint=True, dtype=np.float32)
    lowres_lon = np.linspace(0.0, 360.0, _LOWRES_SIZE[1], endpoint=False, dtype=np.float32)
    remapper = ConservativeNestedLatLonRemapper.from_grids(
        src_lat=source.latitudes,
        src_lon=source.longitudes,
        dst_lat=lowres_lat,
        dst_lon=lowres_lon,
    )

    resolved_device_map = _resolve_benchmark_device_map(device, overrides=device_map)
    if download_models_first:
        _run_loading_stage(
            "download Earth2Studio models",
            lambda: download_earth2studio_models(
                model_cache_dir=resolved_model_cache,
                model_names=("pangu24", "graphcastsmall"),
                device=str(resolved_device_map["graphcastsmall"]),
            ),
        )

    common_variables = _build_common_eval_variables(data_config.pressure_levels)
    lowres_common_variables = list(common_variables)
    pangu_lowres_leads = list(range(highres_metric_step_hours, horizon_hours + 1, highres_metric_step_hours))
    lowres_leads = list(range(lowres_metric_step_hours, horizon_hours + 1, lowres_metric_step_hours))
    init_times = _select_init_times(
        source=source,
        start_time=start_time,
        end_time=end_time,
        init_stride_hours=init_stride_hours,
        horizon_hours=horizon_hours,
        max_init_times=max_init_times,
    )

    lowres_latitude_weights = _latitude_weights_numpy(
        lowres_lat,
        latitude_descending=data_config.latitude_descending,
    )

    lowres_accumulators = _initialize_metric_accumulators(
        model_names=("ours", "graphcastsmall"),
        lead_hours=lowres_leads,
        resolution_group="lowres_common_6h",
    )
    lowres_variable_accumulators = _initialize_variable_metric_accumulators(
        model_names=("ours", "graphcastsmall"),
        lead_hours=lowres_leads,
        resolution_group="lowres_common_6h",
        variable_names=lowres_common_variables,
    )
    lowres_24h_accumulators = _initialize_metric_accumulators(
        model_names=("ours", "pangu24"),
        lead_hours=pangu_lowres_leads,
        resolution_group="lowres_common_24h",
    )
    lowres_24h_variable_accumulators = _initialize_variable_metric_accumulators(
        model_names=("ours", "pangu24"),
        lead_hours=pangu_lowres_leads,
        resolution_group="lowres_common_24h",
        variable_names=lowres_common_variables,
    )

    static_highres_raw = source.build_static_stack(data_config.static_variables)
    static_lowres_raw = remapper.remap(static_highres_raw)
    if normalizer is not None:
        static_lowres = normalizer.normalize_static(
            static_lowres_raw,
            static_variable_names=data_config.static_variables,
        )
    else:
        static_lowres = static_lowres_raw
    our_runtime_device_assignment: list[str] | str | None = None
    our_rollout_cache: dict[pd.Timestamp, dict[int, dict[str, Any]]] = {}

    graphcast_model, actual_graphcast_device = _run_loading_stage(
        "load graphcastsmall runtime",
        lambda: _load_single_earth2studio_model(
            "graphcastsmall",
            model_cache_dir=resolved_model_cache,
            device=resolved_device_map["graphcastsmall"],
            allow_cpu_fallback=allow_cpu_fallback,
        ),
    )
    try:
        graphcast_native_step_hours = _earth2studio_native_step_hours(graphcast_model)
        graphcast_progress = _create_progress_bar(
            total=len(init_times) * _rollout_step_count(max(lowres_leads), graphcast_native_step_hours),
            description=_progress_description(
                "graphcastsmall",
                native_step_hours=graphcast_native_step_hours,
                horizon_hours=max(lowres_leads),
            ),
        )
        graphcast_progress.refresh()
        try:
            for init_index, init_time in enumerate(init_times, start=1):
                graphcast_progress.set_postfix_str(
                    _progress_postfix(
                        init_index=init_index,
                        init_count=len(init_times),
                        init_time=init_time,
                    ),
                    refresh=False,
                )
                lowres_truth = remapper.remap(
                    source.load_canonical_cube(
                        source.canonical_absolute_times(init_time, lowres_leads),
                        lowres_common_variables,
                    )
                )
                graphcast_rollout = _run_earth2studio_rollout(
                    model=graphcast_model,
                    source=source,
                    init_time=init_time,
                    requested_lead_hours=lowres_leads,
                    output_grid="lowres",
                    remapper=remapper,
                    progress_callback=graphcast_progress.update,
                    progress_step_hours=graphcast_native_step_hours,
                )
                _update_metric_accumulators(
                    lowres_accumulators,
                    model_name="graphcastsmall",
                    prediction_by_lead=graphcast_rollout,
                    truth_by_lead=lowres_truth,
                    lead_hours=lowres_leads,
                    canonical_variable_names=lowres_common_variables,
                    expected_variable_names=None,
                    latitude_weights=lowres_latitude_weights,
                )
                _update_variable_metric_accumulators(
                    lowres_variable_accumulators,
                    model_name="graphcastsmall",
                    prediction_by_lead=graphcast_rollout,
                    truth_by_lead=lowres_truth,
                    lead_hours=lowres_leads,
                    canonical_variable_names=lowres_common_variables,
                    expected_variable_names=None,
                    latitude_weights=lowres_latitude_weights,
                )
        finally:
            graphcast_progress.close()
        _plot_single_model_summary(
            _summaries_for_model(
                "graphcastsmall",
                highres_accumulators=lowres_24h_accumulators,
                lowres_accumulators=lowres_accumulators,
            ),
            plot_path=resolved_output_dir / "graphcastsmall_metrics.png",
            highres_leads=pangu_lowres_leads,
            lowres_leads=lowres_leads,
            model_name="graphcastsmall",
        )
    finally:
        _release_runtime_models(graphcast_model)

    main_model = _run_loading_stage(
        "load our model runtime",
        lambda: _load_main_model_runtime(
            config_path=config_path,
            checkpoint_path=resolved_checkpoint_path,
            device=resolved_device_map["ours_lowres"],
            input_size=_LOWRES_SIZE,
        ),
    )
    try:
        our_runtime_device_assignment = _stringify_device_assignment(main_model.device_assignment)
        main_native_step_hours = int(data_config.lead_time_hours)
        ours_progress = _create_progress_bar(
            total=len(init_times) * _rollout_step_count(horizon_hours, main_native_step_hours),
            description=_progress_description(
                "ours",
                native_step_hours=main_native_step_hours,
                horizon_hours=horizon_hours,
            ),
        )
        ours_progress.refresh()
        try:
            for init_index, init_time in enumerate(init_times, start=1):
                ours_progress.set_postfix_str(
                    _progress_postfix(
                        init_index=init_index,
                        init_count=len(init_times),
                        init_time=init_time,
                    ),
                    refresh=False,
                )
                lowres_truth_24h = remapper.remap(
                    source.load_canonical_cube(
                        source.canonical_absolute_times(init_time, pangu_lowres_leads),
                        lowres_common_variables,
                    )
                )
                lowres_truth_6h = remapper.remap(
                    source.load_canonical_cube(
                        source.canonical_absolute_times(init_time, lowres_leads),
                        lowres_common_variables,
                    )
                )
                ours_lowres_rollout = _run_main_model_rollout(
                    model=main_model,
                    data_config=data_config,
                    normalizer=normalizer,
                    source=source,
                    static_features=static_lowres,
                    init_time=init_time,
                    horizon_hours=horizon_hours,
                    output_grid="lowres",
                    remapper=remapper,
                    progress_callback=ours_progress.update,
                )
                our_rollout_cache[pd.Timestamp(init_time)] = ours_lowres_rollout
                _update_metric_accumulators(
                    lowres_24h_accumulators,
                    model_name="ours",
                    prediction_by_lead=ours_lowres_rollout,
                    truth_by_lead=lowres_truth_24h,
                    lead_hours=pangu_lowres_leads,
                    canonical_variable_names=lowres_common_variables,
                    expected_variable_names=_canonical_from_our_channel_names(data_config.channel_names),
                    latitude_weights=lowres_latitude_weights,
                )
                _update_variable_metric_accumulators(
                    lowres_24h_variable_accumulators,
                    model_name="ours",
                    prediction_by_lead=ours_lowres_rollout,
                    truth_by_lead=lowres_truth_24h,
                    lead_hours=pangu_lowres_leads,
                    canonical_variable_names=lowres_common_variables,
                    expected_variable_names=_canonical_from_our_channel_names(data_config.channel_names),
                    latitude_weights=lowres_latitude_weights,
                )
                _update_metric_accumulators(
                    lowres_accumulators,
                    model_name="ours",
                    prediction_by_lead=ours_lowres_rollout,
                    truth_by_lead=lowres_truth_6h,
                    lead_hours=lowres_leads,
                    canonical_variable_names=lowres_common_variables,
                    expected_variable_names=_canonical_from_our_channel_names(data_config.channel_names),
                    latitude_weights=lowres_latitude_weights,
                )
                _update_variable_metric_accumulators(
                    lowres_variable_accumulators,
                    model_name="ours",
                    prediction_by_lead=ours_lowres_rollout,
                    truth_by_lead=lowres_truth_6h,
                    lead_hours=lowres_leads,
                    canonical_variable_names=lowres_common_variables,
                    expected_variable_names=_canonical_from_our_channel_names(data_config.channel_names),
                    latitude_weights=lowres_latitude_weights,
                )
        finally:
            ours_progress.close()
        _plot_single_model_summary(
            _summaries_for_model(
                "ours",
                highres_accumulators=lowres_24h_accumulators,
                lowres_accumulators=lowres_accumulators,
            ),
            plot_path=resolved_output_dir / "ours_metrics.png",
            highres_leads=pangu_lowres_leads,
            lowres_leads=lowres_leads,
            model_name="ours",
        )
    finally:
        _release_runtime_models(main_model)

    pangu_model, actual_pangu_device = _run_loading_stage(
        "load pangu24 runtime",
        lambda: _load_single_earth2studio_model(
            "pangu24",
            model_cache_dir=resolved_model_cache,
            device=resolved_device_map["pangu24"],
            allow_cpu_fallback=allow_cpu_fallback,
        ),
    )
    try:
        pangu_native_step_hours = _earth2studio_native_step_hours(pangu_model)
        pangu_progress = _create_progress_bar(
            total=len(init_times) * _rollout_step_count(max(pangu_lowres_leads), pangu_native_step_hours),
            description=_progress_description(
                "pangu24",
                native_step_hours=pangu_native_step_hours,
                horizon_hours=max(pangu_lowres_leads),
            ),
        )
        pangu_progress.refresh()
        try:
            for init_index, init_time in enumerate(init_times, start=1):
                pangu_progress.set_postfix_str(
                    _progress_postfix(
                        init_index=init_index,
                        init_count=len(init_times),
                        init_time=init_time,
                    ),
                    refresh=False,
                )
                lowres_truth_24h = remapper.remap(
                    source.load_canonical_cube(
                        source.canonical_absolute_times(init_time, pangu_lowres_leads),
                        lowres_common_variables,
                    )
                )
                pangu_highres_rollout = _run_earth2studio_rollout(
                    model=pangu_model,
                    source=source,
                    init_time=init_time,
                    requested_lead_hours=pangu_lowres_leads,
                    output_grid="highres",
                    remapper=None,
                    progress_callback=pangu_progress.update,
                    progress_step_hours=pangu_native_step_hours,
                )
                pangu_rollout = _remap_rollout_entries(pangu_highres_rollout, remapper)
                _update_metric_accumulators(
                    lowres_24h_accumulators,
                    model_name="pangu24",
                    prediction_by_lead=pangu_rollout,
                    truth_by_lead=lowres_truth_24h,
                    lead_hours=pangu_lowres_leads,
                    canonical_variable_names=lowres_common_variables,
                    expected_variable_names=None,
                    latitude_weights=lowres_latitude_weights,
                )
                _update_variable_metric_accumulators(
                    lowres_24h_variable_accumulators,
                    model_name="pangu24",
                    prediction_by_lead=pangu_rollout,
                    truth_by_lead=lowres_truth_24h,
                    lead_hours=pangu_lowres_leads,
                    canonical_variable_names=lowres_common_variables,
                    expected_variable_names=None,
                    latitude_weights=lowres_latitude_weights,
                )
        finally:
            pangu_progress.close()
        _plot_single_model_summary(
            _summaries_for_model(
                "pangu24",
                highres_accumulators=lowres_24h_accumulators,
                lowres_accumulators=lowres_accumulators,
            ),
            plot_path=resolved_output_dir / "pangu24_metrics.png",
            highres_leads=pangu_lowres_leads,
            lowres_leads=lowres_leads,
            model_name="pangu24",
        )
    finally:
        _release_runtime_models(pangu_model)

    metrics_rows = [
        accumulator.summary()
        for accumulator in list(lowres_24h_accumulators.values()) + list(lowres_accumulators.values())
    ]
    variable_metrics_rows = [
        accumulator.summary()
        for accumulator in list(lowres_24h_variable_accumulators.values()) + list(lowres_variable_accumulators.values())
        if accumulator.count > 0
    ]
    metrics_json_path = resolved_output_dir / "benchmark_metrics.json"
    metrics_csv_path = resolved_output_dir / "benchmark_metrics.csv"
    variable_metrics_csv_path = resolved_output_dir / "benchmark_variable_metrics.csv"
    plot_path = resolved_output_dir / "benchmark_metrics.png"
    variable_plot_dir = resolved_output_dir / "variable_metrics"
    individual_plot_paths = {
        "graphcastsmall": str(resolved_output_dir / "graphcastsmall_metrics.png"),
        "ours": str(resolved_output_dir / "ours_metrics.png"),
        "pangu24": str(resolved_output_dir / "pangu24_metrics.png"),
    }
    requested_device_summary = {
        "ours": _stringify_device_assignment(resolved_device_map["ours_lowres"]),
        "pangu24": _stringify_device_assignment(resolved_device_map["pangu24"]),
        "graphcastsmall": _stringify_device_assignment(resolved_device_map["graphcastsmall"]),
    }

    payload = {
        "config_path": str(resolve_repo_path(config_path, config_path=config_path)),
        "data_path": str(resolved_data_path),
        "main_checkpoint_path": str(resolved_checkpoint_path),
        "model_cache_dir": str(resolved_model_cache),
        "device_map_requested": requested_device_summary,
        "device_map_effective": {
            "ours": our_runtime_device_assignment,
            "pangu24": str(actual_pangu_device),
            "graphcastsmall": str(actual_graphcast_device),
        },
        "init_times": [str(init_time) for init_time in init_times],
        "horizon_hours": int(horizon_hours),
        "lowres_common_variables": list(lowres_common_variables),
        "our_cached_rollout_count": len(our_rollout_cache),
        "individual_plot_paths": individual_plot_paths,
        "variable_metric_plot_paths": _plot_variable_metric_summaries(
            variable_metrics_rows,
            output_dir=variable_plot_dir,
            highres_leads=pangu_lowres_leads,
            lowres_leads=lowres_leads,
        ),
        "metrics": metrics_rows,
        "variable_metrics": variable_metrics_rows,
    }
    metrics_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_metrics_csv(metrics_csv_path, metrics_rows)
    _write_metrics_csv(variable_metrics_csv_path, variable_metrics_rows)
    _plot_metric_summaries(
        metrics_rows,
        plot_path=plot_path,
        highres_leads=pangu_lowres_leads,
        lowres_leads=lowres_leads,
    )

    return {
        "metrics_json_path": str(metrics_json_path),
        "metrics_csv_path": str(metrics_csv_path),
        "variable_metrics_csv_path": str(variable_metrics_csv_path),
        "plot_path": str(plot_path),
        "init_time_count": len(init_times),
        "highres_metric_count": len(pangu_lowres_leads),
        "lowres_metric_count": len(lowres_leads),
        "device_map_requested": requested_device_summary,
        "device_map_effective": {
            "ours": our_runtime_device_assignment,
            "pangu24": str(actual_pangu_device),
            "graphcastsmall": str(actual_graphcast_device),
        },
        "model_cache_dir": str(resolved_model_cache),
        "our_cached_rollout_count": len(our_rollout_cache),
        "individual_plot_paths": individual_plot_paths,
        "variable_metric_plot_paths": payload["variable_metric_plot_paths"],
    }


def _import_earth2studio_px():
    _configure_onnxruntime_logging()
    try:
        from earth2studio.models import px  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional runtime package
        raise ImportError(
            "Earth2Studio is required for this benchmark. Install the library in the "
            "MPAS-develop environment before running the Earth2Studio benchmark scripts."
        ) from exc
    return px


def _configure_onnxruntime_logging() -> None:
    os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")
    os.environ.setdefault("ORT_LOG_VERBOSITY_LEVEL", "0")
    try:
        import onnxruntime as ort  # type: ignore
    except ImportError:
        return
    if hasattr(ort, "set_default_logger_severity"):
        ort.set_default_logger_severity(3)
    if hasattr(ort, "set_default_logger_verbosity"):
        ort.set_default_logger_verbosity(0)


def _earth2studio_model_class(px: Any, model_name: str) -> Any:
    normalized = model_name.strip().lower()
    mapping = {
        "fuxi": "FuXi",
        "pangu24": "Pangu24",
        "graphcastsmall": "GraphCastSmall",
    }
    try:
        return getattr(px, mapping[normalized])
    except KeyError as exc:
        raise ValueError(
            f"Unsupported Earth2Studio model {model_name!r}. Expected one of {sorted(mapping)}."
        ) from exc


def _model_input_coords(model: Any) -> OrderedDict[str, np.ndarray]:
    coords_attr = getattr(model, "input_coords")
    coords = coords_attr() if callable(coords_attr) else coords_attr
    if not isinstance(coords, OrderedDict):
        coords = OrderedDict(coords)
    return OrderedDict((str(key), np.asarray(value)) for key, value in coords.items())


def _model_output_coords(model: Any) -> OrderedDict[str, np.ndarray]:
    coords_attr = getattr(model, "output_coords")
    if callable(coords_attr):
        coords = coords_attr(_model_input_coords(model))
    else:
        coords = coords_attr
    if not isinstance(coords, OrderedDict):
        coords = OrderedDict(coords)
    return OrderedDict((str(key), np.asarray(value)) for key, value in coords.items())


def _lead_time_hours_list(lead_times: np.ndarray) -> list[int]:
    return [int(np.rint(float(value / np.timedelta64(1, "h")))) for value in np.asarray(lead_times)]


def _earth2studio_native_step_hours(model: Any) -> int:
    output_coords = _model_output_coords(model)
    lead_hours = [hour for hour in _lead_time_hours_list(output_coords["lead_time"]) if hour > 0]
    if len(lead_hours) <= 0:
        raise ValueError(f"{type(model).__name__} does not expose a positive output lead_time.")
    return int(min(lead_hours))


def _resolve_device(device: str) -> torch.device:
    normalized = str(device).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(normalized)


def _resolve_device_assignment(
    assignment: str | torch.device | Sequence[str | torch.device],
) -> tuple[torch.device, ...]:
    if isinstance(assignment, (list, tuple)):
        values = assignment
    else:
        raw_value = str(assignment)
        if "," in raw_value:
            values = [part.strip() for part in raw_value.split(",") if part.strip()]
        else:
            values = [assignment]
    resolved = tuple(torch.device(value) for value in values)
    if len(resolved) <= 0:
        raise ValueError("Device assignment must contain at least one device.")
    return resolved


def _stringify_device_assignment(value: torch.device | Sequence[torch.device]) -> str | list[str]:
    if isinstance(value, torch.device):
        return str(value)
    return [str(device) for device in value]


def _autocast_context(device: torch.device, amp_dtype: torch.dtype):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def _remap_rollout_entries(
    prediction_by_lead: dict[int, dict[str, Any]],
    remapper: ConservativeNestedLatLonRemapper,
) -> dict[int, dict[str, Any]]:
    return {
        int(lead_hours): {
            "variables": list(entry["variables"]),
            "values": remapper.remap(np.asarray(entry["values"], dtype=np.float32)).astype(
                np.float32,
                copy=False,
            ),
        }
        for lead_hours, entry in prediction_by_lead.items()
    }


def _resolve_benchmark_device_map(
    device: str,
    *,
    overrides: dict[str, str | torch.device] | None = None,
) -> dict[str, torch.device | tuple[torch.device, ...]]:
    keys = ("ours_highres", "ours_lowres", "fuxi", "pangu24", "graphcastsmall")
    override_devices = {
        key: (None if overrides is None else overrides.get(key))
        for key in keys
    }
    if str(device).strip().lower() != "auto":
        base_device = _resolve_device(device)
        return {
            key: (
                (
                    _resolve_device_assignment(override_devices[key])
                    if key.startswith("ours_")
                    else _resolve_device(str(override_devices[key]))
                )
                if override_devices[key] is not None
                else ((base_device,) if key.startswith("ours_") else base_device)
            )
            for key in keys
        }

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        all_cuda_devices = tuple(torch.device(f"cuda:{index}") for index in range(gpu_count))
        proposed = {
            "ours_highres": all_cuda_devices,
            "ours_lowres": all_cuda_devices,
            "fuxi": torch.device("cpu"),
            "pangu24": torch.device("cpu"),
            "graphcastsmall": torch.device("cpu"),
        }
    else:
        cpu_assignment = (torch.device("cpu"),)
        proposed = {
            "ours_highres": cpu_assignment,
            "ours_lowres": cpu_assignment,
            "fuxi": torch.device("cpu"),
            "pangu24": torch.device("cpu"),
            "graphcastsmall": torch.device("cpu"),
        }

    return {
        key: (
            (
                _resolve_device_assignment(override_devices[key])
                if key.startswith("ours_")
                else _resolve_device(str(override_devices[key]))
            )
            if override_devices[key] is not None
            else proposed[key]
        )
        for key in keys
    }


def _resolve_normalization_stats_path(
    override_path: str | Path | None,
    *,
    config_path: str | Path,
    config_data_path: Path | None,
) -> Path | None:
    if override_path is not None:
        return resolve_repo_path(override_path, config_path=config_path)
    if config_data_path is None:
        return None
    return resolve_repo_path(config_data_path, config_path=config_path)


def _build_common_eval_variables(pressure_levels: Sequence[int]) -> list[str]:
    names: list[str] = []
    for prefix in _COMMON_LEVEL_PREFIXES:
        names.extend(f"{prefix}{int(level)}" for level in pressure_levels)
    names.extend(_COMMON_SURFACE_VARIABLES)
    return names


def _canonical_from_our_channel_names(channel_names: Sequence[str]) -> list[str]:
    canonical: list[str] = []
    for name in channel_names:
        upper = str(name).upper()
        if upper in _OUR_SURFACE_CANONICALS:
            canonical.append(_OUR_SURFACE_CANONICALS[upper])
        else:
            canonical.append(upper.lower())
    return canonical


def _parse_canonical_variable_name(name: str) -> dict[str, Any]:
    normalized = str(name).strip().lower()
    if normalized in _CANONICAL_SURFACE_SOURCES:
        return {"kind": "surface", "source": _CANONICAL_SURFACE_SOURCES[normalized]}
    if normalized == "tp":
        return {
            "kind": "surface",
            "source": "total_precipitation",
            "accumulation_hours": 1,
        }
    if normalized == "tp06":
        return {
            "kind": "surface",
            "source": "total_precipitation",
            "accumulation_hours": 6,
        }
    prefix = "".join(character for character in normalized if character.isalpha())
    suffix = normalized[len(prefix) :]
    if prefix in _CANONICAL_UPPER_AIR_SOURCES and suffix.isdigit():
        return {
            "kind": "upper_air",
            "source": _CANONICAL_UPPER_AIR_SOURCES[prefix],
            "level": int(suffix),
        }
    raise ValueError(f"Unsupported canonical variable name {name!r}")


def _latitude_cell_bounds(latitudes: np.ndarray) -> np.ndarray:
    latitudes = np.asarray(latitudes, dtype=np.float64)
    if latitudes.ndim != 1:
        raise ValueError(f"Expected 1D latitude coordinates, got {latitudes.shape}")
    descending = latitudes[0] > latitudes[-1]
    midpoints = (latitudes[:-1] + latitudes[1:]) / 2.0
    start = 90.0 if descending else -90.0
    end = -90.0 if descending else 90.0
    return np.concatenate([[start], midpoints, [end]])


def _longitude_cell_bounds(longitudes: np.ndarray) -> np.ndarray:
    longitudes = np.asarray(longitudes, dtype=np.float64)
    if longitudes.ndim != 1:
        raise ValueError(f"Expected 1D longitude coordinates, got {longitudes.shape}")
    midpoints = (longitudes[:-1] + longitudes[1:]) / 2.0
    first_step = longitudes[1] - longitudes[0]
    last_step = longitudes[-1] - longitudes[-2]
    start = longitudes[0] - first_step / 2.0
    end = longitudes[-1] + last_step / 2.0
    return np.concatenate([[start], midpoints, [end]])


def _latitude_overlap_weights(src_lat: np.ndarray, dst_lat: np.ndarray) -> np.ndarray:
    src_bounds = _latitude_cell_bounds(src_lat)
    dst_bounds = _latitude_cell_bounds(dst_lat)
    weights = np.zeros((len(dst_lat), len(src_lat)), dtype=np.float64)
    for dst_index in range(len(dst_lat)):
        dst_low = min(dst_bounds[dst_index], dst_bounds[dst_index + 1])
        dst_high = max(dst_bounds[dst_index], dst_bounds[dst_index + 1])
        dst_area = abs(
            np.sin(np.deg2rad(dst_high)) - np.sin(np.deg2rad(dst_low))
        )
        for src_index in range(len(src_lat)):
            src_low = min(src_bounds[src_index], src_bounds[src_index + 1])
            src_high = max(src_bounds[src_index], src_bounds[src_index + 1])
            overlap_low = max(dst_low, src_low)
            overlap_high = min(dst_high, src_high)
            if overlap_high <= overlap_low:
                continue
            weights[dst_index, src_index] = abs(
                np.sin(np.deg2rad(overlap_high)) - np.sin(np.deg2rad(overlap_low))
            ) / dst_area
    return weights


def _longitude_overlap_weights(src_lon: np.ndarray, dst_lon: np.ndarray) -> np.ndarray:
    src_bounds = _longitude_cell_bounds(src_lon)
    dst_bounds = _longitude_cell_bounds(dst_lon)
    weights = np.zeros((len(dst_lon), len(src_lon)), dtype=np.float64)
    for dst_index in range(len(dst_lon)):
        dst_low = min(dst_bounds[dst_index], dst_bounds[dst_index + 1])
        dst_high = max(dst_bounds[dst_index], dst_bounds[dst_index + 1])
        dst_width = dst_high - dst_low
        for src_index in range(len(src_lon)):
            src_low = min(src_bounds[src_index], src_bounds[src_index + 1])
            src_high = max(src_bounds[src_index], src_bounds[src_index + 1])
            overlap_low = max(dst_low, src_low)
            overlap_high = min(dst_high, src_high)
            if overlap_high <= overlap_low:
                continue
            weights[dst_index, src_index] = (overlap_high - overlap_low) / dst_width
    return weights


def _latitude_weights_numpy(
    latitudes: Sequence[float],
    *,
    latitude_descending: bool,
) -> np.ndarray:
    latitudes_array = np.asarray(latitudes, dtype=np.float64)
    if latitude_descending and latitudes_array[0] < latitudes_array[-1]:
        latitudes_array = latitudes_array[::-1]
    weights = np.cos(np.deg2rad(latitudes_array)).clip(min=0.0)
    mean_weight = max(float(weights.mean()), np.finfo(np.float64).eps)
    normalized = weights / mean_weight
    if latitude_descending and latitudes_array[0] < latitudes_array[-1]:
        normalized = normalized[::-1]
    return normalized.astype(np.float64)


def _select_init_times(
    *,
    source: BenchmarkEra5Source,
    start_time: str | pd.Timestamp | None,
    end_time: str | pd.Timestamp | None,
    init_stride_hours: int,
    horizon_hours: int,
    max_init_times: int | None,
) -> list[pd.Timestamp]:
    dataset_start, dataset_end = source.time_range()
    effective_start = dataset_start + pd.Timedelta(hours=12)
    effective_end = dataset_end - pd.Timedelta(hours=horizon_hours)
    if start_time is not None:
        effective_start = max(effective_start, pd.Timestamp(start_time))
    if end_time is not None:
        effective_end = min(effective_end, pd.Timestamp(end_time))
    if effective_end < effective_start:
        raise ValueError(
            f"No valid benchmark init times remain inside {effective_start} .. {effective_end}."
        )

    init_times: list[pd.Timestamp] = []
    current = pd.Timestamp(effective_start)
    while current <= effective_end:
        if current.value in source.time_index:
            init_times.append(current)
        current += pd.Timedelta(hours=init_stride_hours)
    if max_init_times is not None:
        init_times = init_times[: int(max_init_times)]
    if len(init_times) <= 0:
        raise ValueError("No valid init times were selected for the benchmark.")
    return init_times


def _load_single_earth2studio_model(
    model_name: str,
    *,
    model_cache_dir: Path,
    device: torch.device,
    allow_cpu_fallback: bool,
) -> tuple[Any, torch.device]:
    px = _import_earth2studio_px()
    os.environ["EARTH2STUDIO_CACHE"] = str(model_cache_dir)
    model_cls = _earth2studio_model_class(px, model_name)

    def _instantiate(target_device: torch.device) -> Any:
        model = model_cls.from_pretrained()
        if hasattr(model, "to"):
            model.to(target_device)
        return model

    try:
        return _instantiate(device), device
    except Exception:
        if not allow_cpu_fallback or device.type != "cuda":
            raise
        cpu_device = torch.device("cpu")
        return _instantiate(cpu_device), cpu_device


def _load_main_model_runtime(
    *,
    config_path: str | Path,
    checkpoint_path: Path,
    device: torch.device | Sequence[torch.device],
    input_size: tuple[int, int],
) -> _SingleDeviceMainModelRunner | _ShardedMainModelRunner:
    base_config = FuXiLowerResConfig.from_yaml(config_path)
    device_assignment = _resolve_device_assignment(device)
    model_config = replace(
        base_config,
        input_size=tuple(int(value) for value in input_size),
        device="cpu",
        dtype=torch.float32,
    )
    model = FuXiLowerRes(model_config)
    _load_main_forecast_checkpoint(model, checkpoint_path)
    model.eval()
    if len(device_assignment) > 1 and all(device_item.type == "cuda" for device_item in device_assignment):
        return _ShardedMainModelRunner(
            model,
            devices=device_assignment,
        )
    resolved_device = device_assignment[0]
    model = model.to(resolved_device)
    return _SingleDeviceMainModelRunner(
        model,
        device=resolved_device,
    )


def _load_main_models(
    *,
    config_path: str | Path,
    checkpoint_path: Path,
    highres_device: torch.device | Sequence[torch.device],
    lowres_device: torch.device | Sequence[torch.device],
    highres_input_size: tuple[int, int],
) -> dict[str, _SingleDeviceMainModelRunner | _ShardedMainModelRunner]:
    base_config = FuXiLowerResConfig.from_yaml(config_path)
    highres_assignment = _resolve_device_assignment(highres_device)
    lowres_assignment = _resolve_device_assignment(lowres_device)
    highres_config = replace(
        base_config,
        input_size=tuple(int(value) for value in highres_input_size),
        device="cpu",
        dtype=torch.float32,
    )
    lowres_config = replace(
        base_config,
        device="cpu",
        dtype=torch.float32,
    )
    highres_model = FuXiLowerRes(highres_config)
    lowres_model = FuXiLowerRes(lowres_config)
    _load_main_forecast_checkpoint(highres_model, checkpoint_path)
    _load_main_forecast_checkpoint(lowres_model, checkpoint_path)
    highres_model.eval()
    lowres_model.eval()
    highres_runner: _SingleDeviceMainModelRunner | _ShardedMainModelRunner
    lowres_runner: _SingleDeviceMainModelRunner | _ShardedMainModelRunner
    if len(highres_assignment) > 1 and all(device.type == "cuda" for device in highres_assignment):
        highres_runner = _ShardedMainModelRunner(
            highres_model,
            devices=highres_assignment,
        )
    else:
        highres_device_resolved = highres_assignment[0]
        highres_model = highres_model.to(highres_device_resolved)
        highres_runner = _SingleDeviceMainModelRunner(
            highres_model,
            device=highres_device_resolved,
        )

    if len(lowres_assignment) > 1 and all(device.type == "cuda" for device in lowres_assignment):
        lowres_runner = _ShardedMainModelRunner(
            lowres_model,
            devices=lowres_assignment,
        )
    else:
        lowres_device_resolved = lowres_assignment[0]
        lowres_model = lowres_model.to(lowres_device_resolved)
        lowres_runner = _SingleDeviceMainModelRunner(
            lowres_model,
            device=lowres_device_resolved,
        )
    return {"highres": highres_runner, "lowres": lowres_runner}


def _release_runtime_models(*models: Any) -> None:
    for model in models:
        if model is None:
            continue
        wrapped_model = getattr(model, "model", None)
        if wrapped_model is not None:
            model = wrapped_model
        try:
            if hasattr(model, "ort") and getattr(model, "ort") is not None:
                setattr(model, "ort", None)
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _runtime_model_device(model: Any) -> torch.device:
    model_device = getattr(model, "device", None)
    if model_device is not None:
        return torch.device(model_device)
    if hasattr(model, "parameters"):
        try:
            first_parameter = next(model.parameters())
        except StopIteration:
            pass
        else:
            return first_parameter.device
    if hasattr(model, "buffers"):
        try:
            first_buffer = next(model.buffers())
        except StopIteration:
            pass
        else:
            return first_buffer.device
    return torch.device("cpu")


def _run_main_model_rollout(
    *,
    model: _SingleDeviceMainModelRunner | _ShardedMainModelRunner,
    data_config: ArcoEra5FuXiDataConfig,
    normalizer: ChannelNormalizer | None,
    source: BenchmarkEra5Source,
    static_features: np.ndarray,
    init_time: pd.Timestamp,
    horizon_hours: int,
    output_grid: str,
    remapper: ConservativeNestedLatLonRemapper | None,
    progress_callback: Callable[[int], None] | None = None,
) -> dict[int, dict[str, Any]]:
    model_channel_names = _canonical_from_our_channel_names(data_config.channel_names)
    input_times = [
        pd.Timestamp(init_time) + pd.Timedelta(hours=int(offset))
        for offset in data_config.input_time_offsets_hours
    ]
    raw_inputs = source.load_canonical_cube(input_times, model_channel_names)
    if output_grid == "lowres":
        if remapper is None:
            raise ValueError("A remapper is required for low-resolution main-model rollout.")
        raw_inputs = remapper.remap(raw_inputs)
    elif output_grid != "highres":
        raise ValueError(f"Unsupported output_grid {output_grid!r}")

    frame_store = {
        pd.Timestamp(time_value): raw_inputs[index].astype(np.float32, copy=True)
        for index, time_value in enumerate(input_times)
    }
    results: dict[int, dict[str, Any]] = {}
    total_steps = horizon_hours // int(data_config.lead_time_hours)
    for rollout_step in range(total_steps):
        current_anchor_time = pd.Timestamp(init_time) + pd.Timedelta(
            hours=int(rollout_step * data_config.lead_time_hours)
        )
        model_input_times = [
            current_anchor_time + pd.Timedelta(hours=int(offset))
            for offset in data_config.input_time_offsets_hours
        ]
        raw_model_inputs = np.stack([frame_store[time_value] for time_value in model_input_times], axis=0)
        if normalizer is not None:
            normalized_inputs = normalizer.normalize_dynamic(
                raw_model_inputs,
                canonical_variable_names=model_channel_names,
            )
        else:
            normalized_inputs = raw_model_inputs
        temb = build_fuxi_time_embeddings(
            current_anchor_time,
            total_steps=1,
            freq_hours=int(data_config.lead_time_hours),
        )[0, 0].astype(np.float32, copy=False)

        forecast = model.predict_next(
            torch.from_numpy(normalized_inputs).unsqueeze(0),
            torch.from_numpy(temb).unsqueeze(0),
            static_features=torch.from_numpy(static_features).unsqueeze(0),
        )
        prediction_normalized = forecast[0].detach().cpu().numpy().astype(np.float32, copy=False)
        if normalizer is not None:
            prediction_raw = normalizer.denormalize_dynamic(
                prediction_normalized,
                canonical_variable_names=model_channel_names,
            )
        else:
            prediction_raw = prediction_normalized
        target_time = current_anchor_time + pd.Timedelta(hours=int(data_config.lead_time_hours))
        frame_store[target_time] = prediction_raw.astype(np.float32, copy=True)
        lead_hours = int((target_time - pd.Timestamp(init_time)) / pd.Timedelta(hours=1))
        results[lead_hours] = {
            "variables": list(model_channel_names),
            "values": prediction_raw.astype(np.float32, copy=False),
        }
        if progress_callback is not None:
            progress_callback(1)
    return results


def _run_earth2studio_rollout(
    *,
    model: Any,
    source: BenchmarkEra5Source,
    init_time: pd.Timestamp,
    requested_lead_hours: Sequence[int],
    output_grid: str,
    remapper: ConservativeNestedLatLonRemapper | None,
    progress_callback: Callable[[int], None] | None = None,
    progress_step_hours: int | None = None,
) -> dict[int, dict[str, Any]]:
    input_coords = _model_input_coords(model)
    variable_names = [str(name) for name in np.asarray(input_coords["variable"]).tolist()]
    input_lead_hours = _lead_time_hours_list(np.asarray(input_coords["lead_time"]))
    absolute_times = [
        pd.Timestamp(init_time) + pd.Timedelta(hours=int(hour)) for hour in input_lead_hours
    ]
    cube = source.load_canonical_cube(absolute_times, variable_names)
    if output_grid == "lowres":
        if remapper is None:
            raise ValueError("A remapper is required for low-resolution Earth2Studio rollout.")
        cube = remapper.remap(cube)
    elif output_grid != "highres":
        raise ValueError(f"Unsupported output_grid {output_grid!r}")

    if "time" not in input_coords:
        if cube.shape[0] != 1:
            raise ValueError(
                f"{type(model).__name__} expects no explicit time dimension, "
                f"but the prepared input cube has shape {cube.shape}."
            )
        cube = cube[0]

    model_device = _runtime_model_device(model)
    x = torch.from_numpy(cube).unsqueeze(0).to(device=model_device, dtype=torch.float32)
    coords = OrderedDict()
    for key, value in input_coords.items():
        if key == "batch":
            continue
        if key == "time":
            coords[key] = np.asarray([np.datetime64(pd.Timestamp(init_time).to_datetime64())])
        else:
            coords[key] = np.asarray(value)

    requested = {int(hour) for hour in requested_lead_hours}
    max_requested = max(requested)
    results: dict[int, dict[str, Any]] = {}
    iterator = model.create_iterator(x, coords)
    max_progress_lead = 0
    for state, state_coords in iterator:
        lead_hours = _max_lead_hours(state_coords)
        if lead_hours <= 0:
            continue
        if progress_callback is not None:
            bounded_lead = min(int(lead_hours), int(max_requested))
            if bounded_lead > max_progress_lead:
                if progress_step_hours is None:
                    progress_increment = 1
                else:
                    progress_increment = max(
                        1,
                        int((bounded_lead - max_progress_lead) // int(progress_step_hours)),
                    )
                progress_callback(progress_increment)
                max_progress_lead = bounded_lead
        if lead_hours in requested:
            values = _extract_last_frame(state)
            results[lead_hours] = {
                "variables": [str(name) for name in np.asarray(state_coords["variable"]).tolist()],
                "values": values.astype(np.float32, copy=False),
            }
            if len(results) == len(requested):
                break
        if lead_hours > max_requested:
            break
    missing = sorted(requested - set(results))
    if missing:
        raise RuntimeError(
            f"{type(model).__name__} did not produce all requested leads {missing} from init {init_time}."
        )
    return results


def _max_lead_hours(coords: OrderedDict[str, np.ndarray]) -> int:
    if "lead_time" not in coords:
        raise KeyError("Model output coords do not contain a lead_time entry.")
    values = np.asarray(coords["lead_time"])
    return int(np.rint(float(np.max(values / np.timedelta64(1, "h")))))


def _extract_last_frame(state: Tensor) -> np.ndarray:
    if state.ndim == 5:
        return state[0, -1].detach().cpu().numpy()
    if state.ndim == 4:
        return state[0].detach().cpu().numpy()
    raise ValueError(f"Unsupported model state tensor shape {tuple(state.shape)}")


def _initialize_metric_accumulators(
    *,
    model_names: Sequence[str],
    lead_hours: Sequence[int],
    resolution_group: str,
) -> dict[tuple[str, int], MetricAccumulator]:
    return {
        (str(model_name), int(lead_hour)): MetricAccumulator(
            lead_hours=int(lead_hour),
            model_name=str(model_name),
            resolution_group=resolution_group,
        )
        for model_name in model_names
        for lead_hour in lead_hours
    }


def _initialize_variable_metric_accumulators(
    *,
    model_names: Sequence[str],
    lead_hours: Sequence[int],
    resolution_group: str,
    variable_names: Sequence[str],
) -> dict[tuple[str, str, int], MetricAccumulator]:
    return {
        (str(model_name), str(variable_name), int(lead_hour)): MetricAccumulator(
            lead_hours=int(lead_hour),
            model_name=str(model_name),
            resolution_group=resolution_group,
            variable_name=str(variable_name),
        )
        for model_name in model_names
        for variable_name in variable_names
        for lead_hour in lead_hours
    }


def _select_channels(
    prediction_entry: dict[str, Any],
    canonical_variable_names: Sequence[str],
    expected_variable_names: Sequence[str] | None,
) -> np.ndarray:
    variable_names = [str(name).lower() for name in prediction_entry["variables"]]
    if expected_variable_names is not None and variable_names == [str(name).lower() for name in expected_variable_names]:
        values = np.asarray(prediction_entry["values"], dtype=np.float32)
        channel_lookup = {
            str(name).lower(): index for index, name in enumerate(expected_variable_names)
        }
    else:
        values = np.asarray(prediction_entry["values"], dtype=np.float32)
        channel_lookup = {name: index for index, name in enumerate(variable_names)}
    return np.stack(
        [values[channel_lookup[str(name).lower()]] for name in canonical_variable_names],
        axis=0,
    ).astype(np.float32, copy=False)


def _update_metric_accumulators(
    accumulators: dict[tuple[str, int], MetricAccumulator],
    *,
    model_name: str,
    prediction_by_lead: dict[int, dict[str, Any]],
    truth_by_lead: np.ndarray,
    lead_hours: Sequence[int],
    canonical_variable_names: Sequence[str],
    expected_variable_names: Sequence[str] | None,
    latitude_weights: np.ndarray,
) -> None:
    for truth_index, lead_hour in enumerate(lead_hours):
        prediction_entry = prediction_by_lead[int(lead_hour)]
        selected_prediction = _select_channels(
            prediction_entry,
            canonical_variable_names=canonical_variable_names,
            expected_variable_names=expected_variable_names,
        )
        accumulators[(model_name, int(lead_hour))].update(
            selected_prediction,
            truth_by_lead[truth_index],
            latitude_weights=latitude_weights,
        )


def _update_variable_metric_accumulators(
    accumulators: dict[tuple[str, str, int], MetricAccumulator],
    *,
    model_name: str,
    prediction_by_lead: dict[int, dict[str, Any]],
    truth_by_lead: np.ndarray,
    lead_hours: Sequence[int],
    canonical_variable_names: Sequence[str],
    expected_variable_names: Sequence[str] | None,
    latitude_weights: np.ndarray,
) -> None:
    for truth_index, lead_hour in enumerate(lead_hours):
        prediction_entry = prediction_by_lead[int(lead_hour)]
        selected_prediction = _select_channels(
            prediction_entry,
            canonical_variable_names=canonical_variable_names,
            expected_variable_names=expected_variable_names,
        )
        truth_values = truth_by_lead[truth_index]
        for channel_index, variable_name in enumerate(canonical_variable_names):
            accumulators[(model_name, str(variable_name), int(lead_hour))].update(
                selected_prediction[channel_index : channel_index + 1],
                truth_values[channel_index : channel_index + 1],
                latitude_weights=latitude_weights,
            )


def _write_metrics_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if len(rows) <= 0:
        raise ValueError("rows must be non-empty for CSV export.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["resolution_group", "model_name", "variable_name", "lead_hours", "mae", "rmse", "count"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _summaries_for_model(
    model_name: str,
    *,
    highres_accumulators: dict[tuple[str, int], MetricAccumulator],
    lowres_accumulators: dict[tuple[str, int], MetricAccumulator],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for accumulator in list(highres_accumulators.values()) + list(lowres_accumulators.values()):
        if accumulator.model_name != model_name or accumulator.count <= 0:
            continue
        rows.append(accumulator.summary())
    return rows


def _benchmark_lead_axes(
    *,
    highres_leads: Sequence[int],
    lowres_leads: Sequence[int],
) -> dict[str, Sequence[int]]:
    return {
        "highres_common_24h": highres_leads,
        "lowres_common_24h": highres_leads,
        "lowres_common_6h": lowres_leads,
    }


def _plot_single_model_summary(
    rows: Sequence[dict[str, Any]],
    *,
    plot_path: Path,
    highres_leads: Sequence[int],
    lowres_leads: Sequence[int],
    model_name: str,
) -> None:
    if len(rows) <= 0:
        return
    plt = _import_pyplot()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["resolution_group"]), []).append(row)

    lead_axes = _benchmark_lead_axes(
        highres_leads=highres_leads,
        lowres_leads=lowres_leads,
    )
    panels: list[tuple[str, str, Sequence[int], str]] = []
    if "highres_common_24h" in grouped:
        panels.append(("highres_common_24h", "mae", lead_axes["highres_common_24h"], "High-Res MAE"))
        panels.append(("highres_common_24h", "rmse", lead_axes["highres_common_24h"], "High-Res RMSE"))
    if "lowres_common_24h" in grouped:
        panels.append(("lowres_common_24h", "mae", lead_axes["lowres_common_24h"], "Low-Res 24h MAE"))
        panels.append(("lowres_common_24h", "rmse", lead_axes["lowres_common_24h"], "Low-Res 24h RMSE"))
    if "lowres_common_6h" in grouped:
        panels.append(("lowres_common_6h", "mae", lead_axes["lowres_common_6h"], "Low-Res 6h MAE"))
        panels.append(("lowres_common_6h", "rmse", lead_axes["lowres_common_6h"], "Low-Res 6h RMSE"))

    figure, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5), dpi=160, sharex=False)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes], dtype=object)
    for axis, (resolution_group, metric_key, lead_axis, title) in zip(axes.flat, panels, strict=True):
        ordered = {
            int(row["lead_hours"]): float(row[metric_key])
            for row in grouped[resolution_group]
        }
        axis.plot(
            list(lead_axis),
            [ordered.get(int(lead), float("nan")) for lead in lead_axis],
            marker="o",
            linewidth=1.8,
            label=model_name,
        )
        axis.set_title(title)
        axis.set_xlabel("Lead Time (hours)")
        axis.set_ylabel(metric_key.upper())
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.legend(loc="best")
    figure.suptitle(f"{model_name} benchmark summary")
    figure.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, bbox_inches="tight")
    plt.close(figure)


def _plot_metric_summaries(
    rows: Sequence[dict[str, Any]],
    *,
    plot_path: Path,
    highres_leads: Sequence[int],
    lowres_leads: Sequence[int],
) -> None:
    plt = _import_pyplot()
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        grouped.setdefault(str(row["resolution_group"]), {}).setdefault(str(row["model_name"]), []).append(row)
    lead_axes = _benchmark_lead_axes(
        highres_leads=highres_leads,
        lowres_leads=lowres_leads,
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=160, sharex=False)
    panels = [
        ("lowres_common_24h", "mae", axes[0, 0], "Low-Res 24h MAE"),
        ("lowres_common_24h", "rmse", axes[0, 1], "Low-Res 24h RMSE"),
        ("lowres_common_6h", "mae", axes[1, 0], "Low-Res 6h MAE"),
        ("lowres_common_6h", "rmse", axes[1, 1], "Low-Res 6h RMSE"),
    ]
    for resolution_group, metric_key, axis, title in panels:
        lead_axis = lead_axes[resolution_group]
        for model_name, model_rows in sorted(grouped.get(resolution_group, {}).items()):
            ordered = {
                int(row["lead_hours"]): float(row[metric_key])
                for row in model_rows
            }
            axis.plot(
                list(lead_axis),
                [ordered.get(int(lead), float("nan")) for lead in lead_axis],
                marker="o",
                linewidth=1.8,
                label=model_name,
            )
        axis.set_title(title)
        axis.set_xlabel("Lead Time (hours)")
        axis.set_ylabel(metric_key.upper())
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.legend(loc="best")
    figure.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, bbox_inches="tight")
    plt.close(figure)


def _plot_variable_metric_summaries(
    rows: Sequence[dict[str, Any]],
    *,
    output_dir: Path,
    highres_leads: Sequence[int],
    lowres_leads: Sequence[int],
) -> dict[str, dict[str, str]]:
    plt = _import_pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    lead_axes = _benchmark_lead_axes(
        highres_leads=highres_leads,
        lowres_leads=lowres_leads,
    )
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        variable_name = row.get("variable_name")
        if variable_name in (None, ""):
            continue
        key = (str(row["resolution_group"]), str(variable_name))
        grouped.setdefault(key, {}).setdefault(str(row["model_name"]), []).append(row)

    saved_paths: dict[str, dict[str, str]] = {}
    for (resolution_group, variable_name), model_groups in sorted(grouped.items()):
        lead_axis = lead_axes[resolution_group]
        figure, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=160, sharex=False)
        metric_panels = [("mae", axes[0], "MAE"), ("rmse", axes[1], "RMSE")]
        for metric_key, axis, title in metric_panels:
            for model_name, model_rows in sorted(model_groups.items()):
                ordered = {
                    int(row["lead_hours"]): float(row[metric_key])
                    for row in model_rows
                }
                axis.plot(
                    list(lead_axis),
                    [ordered.get(int(lead), float("nan")) for lead in lead_axis],
                    marker="o",
                    linewidth=1.8,
                    label=model_name,
                )
            axis.set_title(f"{variable_name} {title}")
            axis.set_xlabel("Lead Time (hours)")
            axis.set_ylabel(title)
            axis.grid(alpha=0.25, linewidth=0.6)
            axis.legend(loc="best")
        figure.suptitle(f"{resolution_group} :: {variable_name}")
        figure.tight_layout()
        variable_dir = output_dir / resolution_group
        variable_dir.mkdir(parents=True, exist_ok=True)
        plot_path = variable_dir / f"{variable_name}_metrics.png"
        figure.savefig(plot_path, bbox_inches="tight")
        plt.close(figure)
        saved_paths.setdefault(resolution_group, {})[variable_name] = str(plot_path)
    return saved_paths


def _import_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Benchmark plotting requires matplotlib in the active environment."
        ) from exc
    return plt


__all__ = [
    "BenchmarkEra5Source",
    "ChannelNormalizer",
    "ConservativeNestedLatLonRemapper",
    "DEFAULT_BENCHMARK_OUTPUT_DIR",
    "DEFAULT_EARTH2STUDIO_MODEL_CACHE",
    "compare_forecast_models",
    "download_earth2studio_models",
]
