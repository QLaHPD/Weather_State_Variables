from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

import gcsfs
import numpy as np
import pandas as pd
import requests
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from ..config import DEFAULT_MODEL_CONFIG_PATH, load_config_section
from ..models.fuxi_lower_res import FuXiLowerResConfig
from ..models.fuxi_short import build_fuxi_time_embeddings


DEFAULT_ARCO_ERA5_URL = (
    "gs://gcp-public-data-arco-era5/ar/1959-2022-1h-360x181_equiangular_with_poles_conservative.zarr"
)

FUXI_PRESSURE_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
FUXI_UPPER_AIR_VARIABLES = (
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "relative_humidity",
)
FUXI_SURFACE_VARIABLES = (
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_precipitation",
)
FUXI_STATIC_VARIABLES = (
    "land_sea_mask",
    "orography",
    "cos_latitude",
    "cos_longitude",
    "sin_longitude",
)

_STANDARD_GRAVITY = 9.80665
_EPSILON = 0.622


def _to_int_tuple(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def _to_str_tuple(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


def _as_timestamp(value: str | np.datetime64 | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.Timestamp(value)


def _normalize_arco_gs_url(dataset_url: str) -> str:
    if dataset_url.startswith("gs://"):
        return dataset_url.rstrip("/")

    storage_prefix = "https://storage.googleapis.com/"
    if dataset_url.startswith(storage_prefix):
        return f"gs://{dataset_url[len(storage_prefix):].rstrip('/')}"

    console_prefix = "https://console.cloud.google.com/storage/browser/"
    if dataset_url.startswith(console_prefix):
        path = dataset_url[len(console_prefix) :].split("?", 1)[0].strip("/")
        return f"gs://{path}"

    parsed = urlparse(dataset_url)
    if parsed.scheme == "https" and parsed.netloc == "storage.googleapis.com":
        return f"gs://{parsed.path.lstrip('/').rstrip('/')}"

    raise ValueError(
        "Unsupported ARCO ERA5 dataset URL. Expected gs://..., "
        "https://storage.googleapis.com/..., or the Google Cloud Console browser URL."
    )


def _arco_https_prefix(dataset_url: str) -> str:
    gs_url = _normalize_arco_gs_url(dataset_url)
    return "https://storage.googleapis.com/" + gs_url[len("gs://") :]


def arco_metadata_url(dataset_url: str) -> str:
    return _arco_https_prefix(dataset_url) + "/.zmetadata"


def fetch_arco_zarr_metadata(dataset_url: str, timeout: int = 30) -> dict[str, Any]:
    response = requests.get(arco_metadata_url(dataset_url), timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Malformed .zmetadata payload at {dataset_url}")
    return metadata


def list_arco_dataset_variables(dataset_url: str) -> list[str]:
    metadata = fetch_arco_zarr_metadata(dataset_url)
    return sorted(
        {
            key.split("/", 1)[0]
            for key in metadata
            if "/" in key and not key.startswith(".")
        }
    )


def build_fuxi_channel_names(
    pressure_levels: Sequence[int] = FUXI_PRESSURE_LEVELS,
) -> list[str]:
    channel_names: list[str] = []
    prefixes = {
        "geopotential": "Z",
        "temperature": "T",
        "u_component_of_wind": "U",
        "v_component_of_wind": "V",
        "relative_humidity": "R",
    }
    for variable in FUXI_UPPER_AIR_VARIABLES:
        prefix = prefixes[variable]
        channel_names.extend(f"{prefix}{int(level)}" for level in pressure_levels)
    channel_names.extend(["T2M", "U10", "V10", "MSL", "TP"])
    return channel_names


def specific_humidity_to_relative_humidity(
    specific_humidity: xr.DataArray,
    temperature: xr.DataArray,
    pressure_levels_hpa: xr.DataArray,
) -> xr.DataArray:
    """Approximate ERA5-style relative humidity in percent from q, T, and pressure."""

    pressure_pa = pressure_levels_hpa.astype(np.float32) * 100.0
    pressure_pa = xr.DataArray(
        pressure_pa.data,
        dims=("level",),
        coords={"level": pressure_levels_hpa.values},
    )

    safe_q = specific_humidity.clip(min=0.0, max=0.999999)
    mixing_ratio = safe_q / (1.0 - safe_q)
    vapor_pressure = pressure_pa * mixing_ratio / (_EPSILON + mixing_ratio)

    temperature_c = temperature - 273.15
    saturation_vapor_pressure = 611.2 * np.exp((17.67 * temperature_c) / (temperature_c + 243.5))
    relative_humidity = 100.0 * vapor_pressure / saturation_vapor_pressure
    return relative_humidity.clip(min=0.0, max=100.0).transpose(*specific_humidity.dims).astype(np.float32)


@dataclass(frozen=True)
class ArcoEra5CompatibilityReport:
    dataset_url: str
    available_variables: tuple[str, ...]
    available_levels: tuple[int, ...]
    latitude_size: int
    longitude_size: int
    missing_dynamic_sources: tuple[str, ...]
    missing_static_sources: tuple[str, ...]
    can_derive_relative_humidity: bool

    @property
    def supports_fuxi_inputs(self) -> bool:
        return (
            len(self.missing_dynamic_sources) == 0
            and len(self.missing_static_sources) == 0
            and set(FUXI_PRESSURE_LEVELS).issubset(set(self.available_levels))
        )

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_url": self.dataset_url,
            "latitude_size": self.latitude_size,
            "longitude_size": self.longitude_size,
            "available_levels": list(self.available_levels),
            "available_variable_count": len(self.available_variables),
            "missing_dynamic_sources": list(self.missing_dynamic_sources),
            "missing_static_sources": list(self.missing_static_sources),
            "can_derive_relative_humidity": self.can_derive_relative_humidity,
            "supports_fuxi_inputs": self.supports_fuxi_inputs,
        }


def inspect_arco_era5_dataset(dataset_url: str = DEFAULT_ARCO_ERA5_URL) -> ArcoEra5CompatibilityReport:
    metadata = fetch_arco_zarr_metadata(dataset_url)
    available_variables = sorted(
        {
            key.split("/", 1)[0]
            for key in metadata
            if "/" in key and not key.startswith(".")
        }
    )
    available_variable_set = set(available_variables)

    can_derive_relative_humidity = {
        "specific_humidity",
        "temperature",
        "level",
    }.issubset(available_variable_set)

    dynamic_sources = {
        "geopotential",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
    }
    if "relative_humidity" not in available_variable_set and not can_derive_relative_humidity:
        dynamic_sources.add("relative_humidity")

    static_sources = {"land_sea_mask", "geopotential_at_surface", "latitude", "longitude"}

    latitude_size = int(metadata["latitude/.zarray"]["shape"][0])
    longitude_size = int(metadata["longitude/.zarray"]["shape"][0])

    fs = gcsfs.GCSFileSystem(token="anon")
    store = gcsfs.mapping.GCSMap(
        _normalize_arco_gs_url(dataset_url)[len("gs://") :],
        gcs=fs,
        check=False,
    )
    ds = xr.open_zarr(store, consolidated=True)
    available_levels = tuple(int(level) for level in ds["level"].values.tolist())

    return ArcoEra5CompatibilityReport(
        dataset_url=_normalize_arco_gs_url(dataset_url),
        available_variables=tuple(available_variables),
        available_levels=available_levels,
        latitude_size=latitude_size,
        longitude_size=longitude_size,
        missing_dynamic_sources=tuple(sorted(dynamic_sources - available_variable_set)),
        missing_static_sources=tuple(sorted(static_sources - available_variable_set)),
        can_derive_relative_humidity=can_derive_relative_humidity,
    )


@dataclass(frozen=True)
class ArcoEra5FuXiDataConfig:
    """Configuration for building FuXi-style samples from ARCO ERA5 Zarr datasets."""

    dataset_url: str = DEFAULT_ARCO_ERA5_URL
    input_time_offsets_hours: tuple[int, int] = (-1, 0)
    lead_time_hours: int = 1
    forecast_steps: int = 2
    sample_stride_hours: int = 1
    pressure_levels: tuple[int, ...] = FUXI_PRESSURE_LEVELS
    upper_air_variables: tuple[str, ...] = FUXI_UPPER_AIR_VARIABLES
    surface_variables: tuple[str, ...] = FUXI_SURFACE_VARIABLES
    static_variables: tuple[str, ...] = FUXI_STATIC_VARIABLES
    humidity_source: str = "auto"
    orography_source: str = "geopotential_at_surface"
    convert_geopotential_to_height: bool = True
    latitude_descending: bool = True
    gcs_token: str = "anon"
    start_time: pd.Timestamp | None = None
    end_time: pd.Timestamp | None = None
    config_path: Path = DEFAULT_MODEL_CONFIG_PATH

    def __post_init__(self) -> None:
        if len(self.input_time_offsets_hours) != 2:
            raise ValueError(
                f"Expected 2 input time offsets for FuXi-style inputs, got {self.input_time_offsets_hours}"
            )
        if self.input_time_offsets_hours[-1] != 0:
            raise ValueError(
                "Expected the final input offset to be 0 hours so the second frame is the anchor time."
            )
        if self.lead_time_hours <= 0:
            raise ValueError(f"lead_time_hours must be positive, got {self.lead_time_hours}")
        if self.forecast_steps <= 0:
            raise ValueError(f"forecast_steps must be positive, got {self.forecast_steps}")
        if self.sample_stride_hours <= 0:
            raise ValueError(f"sample_stride_hours must be positive, got {self.sample_stride_hours}")

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
    ) -> "ArcoEra5FuXiDataConfig":
        resolved_config_path, data = load_config_section("data", config_path)
        default_url = data.get("dataset_url", DEFAULT_ARCO_ERA5_URL)

        return cls(
            dataset_url=str(default_url),
            input_time_offsets_hours=_to_int_tuple(data.get("input_time_offsets_hours", [-1, 0])),
            lead_time_hours=int(data.get("lead_time_hours", 1)),
            forecast_steps=int(data.get("forecast_steps", 2)),
            sample_stride_hours=int(data.get("sample_stride_hours", 1)),
            pressure_levels=_to_int_tuple(data.get("pressure_levels", list(FUXI_PRESSURE_LEVELS))),
            upper_air_variables=_to_str_tuple(data.get("upper_air_variables", list(FUXI_UPPER_AIR_VARIABLES))),
            surface_variables=_to_str_tuple(data.get("surface_variables", list(FUXI_SURFACE_VARIABLES))),
            static_variables=_to_str_tuple(data.get("static_variables", list(FUXI_STATIC_VARIABLES))),
            humidity_source=str(data.get("humidity_source", "auto")),
            orography_source=str(data.get("orography_source", "geopotential_at_surface")),
            convert_geopotential_to_height=bool(data.get("convert_geopotential_to_height", True)),
            latitude_descending=bool(data.get("latitude_descending", True)),
            gcs_token=str(data.get("gcs_token", "anon")),
            start_time=_as_timestamp(data.get("start_time")),
            end_time=_as_timestamp(data.get("end_time")),
            config_path=resolved_config_path,
        )

    @property
    def channel_names(self) -> list[str]:
        return build_fuxi_channel_names(self.pressure_levels)


class ArcoEra5FuXiDataset(Dataset[dict[str, Any]]):
    """Lazy remote dataset that produces FuXi-style samples from ARCO ERA5."""

    def __init__(self, config: ArcoEra5FuXiDataConfig | None = None) -> None:
        self.config = config or ArcoEra5FuXiDataConfig.from_yaml()
        self._dataset: xr.Dataset | None = None
        self._time_values: np.ndarray | None = None
        self._static_features: torch.Tensor | None = None
        self._valid_anchor_indices: np.ndarray | None = None
        self._dataset_step_hours: int | None = None

    def _open_dataset(self) -> xr.Dataset:
        if self._dataset is None:
            fs = gcsfs.GCSFileSystem(token=self.config.gcs_token)
            store = gcsfs.mapping.GCSMap(
                _normalize_arco_gs_url(self.config.dataset_url)[len("gs://") :],
                gcs=fs,
                check=False,
            )
            self._dataset = xr.open_zarr(store, consolidated=True)
        return self._dataset

    def _load_time_values(self) -> np.ndarray:
        if self._time_values is None:
            ds = self._open_dataset()
            self._time_values = ds["time"].values
        return self._time_values

    def _dataset_frequency_hours(self) -> int:
        if self._dataset_step_hours is None:
            times = self._load_time_values()
            delta = pd.Timestamp(times[1]) - pd.Timestamp(times[0])
            self._dataset_step_hours = int(delta / pd.Timedelta(hours=1))
        return self._dataset_step_hours

    def _step_count(self, hours: int) -> int:
        step_hours = self._dataset_frequency_hours()
        if hours % step_hours != 0:
            raise ValueError(
                f"Requested {hours} hours but dataset cadence is {step_hours} hours."
            )
        return hours // step_hours

    def _prepare_spatial_dataarray(self, data_array: xr.DataArray) -> xr.DataArray:
        ordered_dims = [dim for dim in ("time", "level", "latitude", "longitude") if dim in data_array.dims]
        data_array = data_array.transpose(*ordered_dims)
        if (
            "latitude" in data_array.coords
            and self.config.latitude_descending
            and data_array["latitude"].values[0] < data_array["latitude"].values[-1]
        ):
            data_array = data_array.isel(latitude=slice(None, None, -1))
        return data_array.astype(np.float32)

    def _select_pressure_variable(self, name: str, time_indices: Sequence[int]) -> xr.DataArray:
        ds = self._open_dataset()
        array = ds[name].isel(time=list(time_indices)).sel(level=list(self.config.pressure_levels))
        return self._prepare_spatial_dataarray(array)

    def _select_surface_variable(self, name: str, time_indices: Sequence[int]) -> xr.DataArray:
        ds = self._open_dataset()
        array = ds[name].isel(time=list(time_indices))
        return self._prepare_spatial_dataarray(array)

    def _resolve_relative_humidity(self, time_indices: Sequence[int]) -> xr.DataArray:
        ds = self._open_dataset()
        if self.config.humidity_source == "relative_humidity":
            return self._select_pressure_variable("relative_humidity", time_indices)
        if self.config.humidity_source == "specific_humidity":
            return specific_humidity_to_relative_humidity(
                self._select_pressure_variable("specific_humidity", time_indices),
                self._select_pressure_variable("temperature", time_indices),
                ds["level"].sel(level=list(self.config.pressure_levels)),
            )
        if self.config.humidity_source != "auto":
            raise ValueError(
                "humidity_source must be one of 'auto', 'relative_humidity', or 'specific_humidity'."
            )

        if "relative_humidity" in ds.data_vars:
            return self._select_pressure_variable("relative_humidity", time_indices)
        if "specific_humidity" not in ds.data_vars:
            raise KeyError(
                "Dataset has neither relative_humidity nor specific_humidity, so RH cannot be provided."
            )
        return specific_humidity_to_relative_humidity(
            self._select_pressure_variable("specific_humidity", time_indices),
            self._select_pressure_variable("temperature", time_indices),
            ds["level"].sel(level=list(self.config.pressure_levels)),
        )

    def _build_dynamic_tensor(self, time_indices: Sequence[int]) -> np.ndarray:
        upper_air_groups: list[np.ndarray] = []
        for variable_name in self.config.upper_air_variables:
            if variable_name == "relative_humidity":
                array = self._resolve_relative_humidity(time_indices)
            else:
                array = self._select_pressure_variable(variable_name, time_indices)
            values = array.load().values
            upper_air_groups.append(values)

        upper_air = np.concatenate(upper_air_groups, axis=1)

        surface_groups: list[np.ndarray] = []
        for variable_name in self.config.surface_variables:
            array = self._select_surface_variable(variable_name, time_indices)
            surface_groups.append(array.load().values[:, None, :, :])

        surface = np.concatenate(surface_groups, axis=1)
        return np.concatenate([upper_air, surface], axis=1).astype(np.float32)

    def _build_static_features(self) -> torch.Tensor:
        if self._static_features is not None:
            return self._static_features

        ds = self._open_dataset()
        land_sea_mask = self._prepare_spatial_dataarray(ds["land_sea_mask"]).load().values

        orography = self._prepare_spatial_dataarray(ds[self.config.orography_source]).load().values
        if self.config.convert_geopotential_to_height and self.config.orography_source == "geopotential_at_surface":
            orography = orography / _STANDARD_GRAVITY

        latitude = self._prepare_spatial_dataarray(ds["latitude"]).values
        longitude = self._prepare_spatial_dataarray(ds["longitude"]).values
        lat_grid, lon_grid = np.meshgrid(latitude, longitude, indexing="ij")

        static_map = {
            "land_sea_mask": land_sea_mask.astype(np.float32),
            "orography": orography.astype(np.float32),
            "cos_latitude": np.cos(np.deg2rad(lat_grid)).astype(np.float32),
            "cos_longitude": np.cos(np.deg2rad(lon_grid)).astype(np.float32),
            "sin_longitude": np.sin(np.deg2rad(lon_grid)).astype(np.float32),
        }

        stacked = np.stack([static_map[name] for name in self.config.static_variables], axis=0)
        self._static_features = torch.from_numpy(stacked)
        return self._static_features

    def _build_valid_anchor_indices(self) -> np.ndarray:
        if self._valid_anchor_indices is not None:
            return self._valid_anchor_indices

        times = self._load_time_values()
        min_input_offset = min(self._step_count(hours) for hours in self.config.input_time_offsets_hours)
        max_target_offset = self._step_count(self.config.lead_time_hours * self.config.forecast_steps)
        sample_stride = self._step_count(self.config.sample_stride_hours)

        start_index = max(0, -min_input_offset)
        end_index = len(times) - max_target_offset
        anchor_indices = np.arange(start_index, end_index, sample_stride, dtype=np.int64)

        if self.config.start_time is not None:
            anchor_indices = anchor_indices[times[anchor_indices] >= np.datetime64(self.config.start_time)]
        if self.config.end_time is not None:
            anchor_indices = anchor_indices[times[anchor_indices + max_target_offset] <= np.datetime64(self.config.end_time)]

        self._valid_anchor_indices = anchor_indices
        return self._valid_anchor_indices

    def __len__(self) -> int:
        return int(self._build_valid_anchor_indices().shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        anchor_index = int(self._build_valid_anchor_indices()[index])
        time_values = self._load_time_values()

        input_indices = [anchor_index + self._step_count(hours) for hours in self.config.input_time_offsets_hours]
        target_indices = [
            anchor_index + self._step_count(self.config.lead_time_hours * step)
            for step in range(1, self.config.forecast_steps + 1)
        ]

        x = torch.from_numpy(self._build_dynamic_tensor(input_indices))
        target = torch.from_numpy(self._build_dynamic_tensor(target_indices))
        static_features = self._build_static_features().clone()
        anchor_time = pd.Timestamp(time_values[anchor_index])
        temb = torch.from_numpy(
            build_fuxi_time_embeddings(anchor_time, total_steps=1, freq_hours=self.config.lead_time_hours)[0, 0]
        )

        return {
            "x": x,
            "target": target,
            "static_features": static_features,
            "temb": temb,
            "input_times": [str(pd.Timestamp(time_values[i])) for i in input_indices],
            "anchor_time": str(anchor_time),
            "target_times": [str(pd.Timestamp(time_values[i])) for i in target_indices],
        }

    def source_summary(self) -> dict[str, Any]:
        report = inspect_arco_era5_dataset(self.config.dataset_url)
        model_config = FuXiLowerResConfig.from_yaml(self.config.config_path)
        return {
            **report.summary(),
            "dataset_url": self.config.dataset_url,
            "input_time_offsets_hours": list(self.config.input_time_offsets_hours),
            "lead_time_hours": self.config.lead_time_hours,
            "forecast_steps": self.config.forecast_steps,
            "target_time_offsets_hours": [
                self.config.lead_time_hours * step for step in range(1, self.config.forecast_steps + 1)
            ],
            "sample_stride_hours": self.config.sample_stride_hours,
            "pressure_levels": list(self.config.pressure_levels),
            "upper_air_variables": list(self.config.upper_air_variables),
            "surface_variables": list(self.config.surface_variables),
            "static_variables": list(self.config.static_variables),
            "orography_source": self.config.orography_source,
            "latitude_descending": self.config.latitude_descending,
            "model_input_size": list(model_config.input_size),
            "model_time_steps": model_config.time_steps,
            "model_dynamic_channels": model_config.in_chans,
            "model_static_channels": model_config.aux_chans,
            "model_forecast_steps": model_config.forecast_steps,
        }


def build_arco_era5_dataloader(
    dataset: ArcoEra5FuXiDataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
    pin_memory: bool = False,
) -> DataLoader[dict[str, Any]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )


__all__ = [
    "ArcoEra5CompatibilityReport",
    "ArcoEra5FuXiDataConfig",
    "ArcoEra5FuXiDataset",
    "DEFAULT_ARCO_ERA5_URL",
    "FUXI_PRESSURE_LEVELS",
    "FUXI_STATIC_VARIABLES",
    "FUXI_SURFACE_VARIABLES",
    "FUXI_UPPER_AIR_VARIABLES",
    "arco_metadata_url",
    "build_arco_era5_dataloader",
    "build_fuxi_channel_names",
    "fetch_arco_zarr_metadata",
    "inspect_arco_era5_dataset",
    "list_arco_dataset_variables",
    "specific_humidity_to_relative_humidity",
]
