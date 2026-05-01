from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
import math
import threading
from pathlib import Path
import json
import shutil
from typing import Any, Sequence
from urllib.parse import urlparse
import warnings

import gcsfs
import numpy as np
import pandas as pd
import requests
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Sampler
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from ..config import DEFAULT_MODEL_CONFIG_PATH, load_config_section, load_yaml_config, resolve_repo_path


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
FUXI_STATIC_SOURCE_VARIABLES = ("land_sea_mask", "geopotential_at_surface")

_STANDARD_GRAVITY = 9.80665
_EPSILON = 0.622
_MAX_DYNAMIC_RAM_CACHE_TIME_STEPS = 64
_DEFAULT_DYNAMIC_PREFETCH_BLOCK_TIME_STEPS = 8
_DEFAULT_NORMALIZATION_STATS_PATH = Path("runs/cache/era5_fuxi_normalization.json")
_DEFAULT_NORMALIZATION_FIT_SAMPLE_COUNT = 128
_DEFAULT_VARIABLE_DOWNLOAD_WORKERS = 5
_DEFAULT_DOWNLOAD_PREFETCH_CHUNK_COUNT = 4
_NORMALIZATION_STATS_VERSION = 1
_MIN_NORMALIZATION_STD = 1.0e-6
_DYNAMIC_ZSCORE_KIND = "zscore"
_DYNAMIC_LOG1P_MM_ZSCORE_KIND = "log1p_mm_zscore"
_IDENTITY_KIND = "identity"

_DOWNLOAD_DATASET_CACHE = threading.local()


def _to_int_tuple(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def _to_str_tuple(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


def _as_timestamp(value: str | np.datetime64 | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.Timestamp(value)


def _to_optional_resolved_path(
    value: str | Path | None,
    *,
    config_path: str | Path,
) -> Path | None:
    if value in {None, ""}:
        return None
    return resolve_repo_path(value, config_path=config_path)


def _maybe_local_zarr_path(dataset_url: str | Path) -> Path | None:
    candidate = Path(str(dataset_url)).expanduser()
    if candidate.exists():
        return candidate.resolve()
    return None


def _normalize_arco_gs_url(dataset_url: str | Path) -> str:
    dataset_url = str(dataset_url)
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


def open_arco_era5_dataset(
    dataset_url: str | Path = DEFAULT_ARCO_ERA5_URL,
    *,
    gcs_token: str = "anon",
) -> xr.Dataset:
    local_path = _maybe_local_zarr_path(dataset_url)
    if local_path is not None:
        return xr.open_zarr(local_path, consolidated=None, chunks=None)

    fs = gcsfs.GCSFileSystem(token=gcs_token)
    store = gcsfs.mapping.GCSMap(
        _normalize_arco_gs_url(dataset_url)[len("gs://") :],
        gcs=fs,
        check=False,
    )
    return xr.open_zarr(store, consolidated=True, chunks=None)


def describe_arco_era5_dataset_location(dataset_url: str | Path) -> str:
    local_path = _maybe_local_zarr_path(dataset_url)
    if local_path is not None:
        return str(local_path)
    return _normalize_arco_gs_url(dataset_url)


def _read_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)


def inspect_local_zarr_time_axes(zarr_path: str | Path) -> list[dict[str, Any]]:
    root = Path(zarr_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Zarr store not found: {root}")

    entries: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        zarray_path = child / ".zarray"
        zattrs_path = child / ".zattrs"
        if not zarray_path.is_file() or not zattrs_path.is_file():
            continue

        zarray = _read_json_file(zarray_path)
        zattrs = _read_json_file(zattrs_path)
        dims = tuple(zattrs.get("_ARRAY_DIMENSIONS", []))
        if "time" not in dims:
            continue

        time_dim_index = dims.index("time")
        shape = [int(value) for value in zarray["shape"]]
        chunks = [int(value) for value in zarray["chunks"]]
        entries.append(
            {
                "name": child.name,
                "dims": list(dims),
                "shape": shape,
                "chunks": chunks,
                "time_dim_index": time_dim_index,
                "time_size": shape[time_dim_index],
                "zarray_path": str(zarray_path),
            }
        )
    return entries


def repair_local_zarr_time_consistency(
    zarr_path: str | Path,
    *,
    target_time_size: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    root = Path(zarr_path).resolve()
    entries = inspect_local_zarr_time_axes(root)
    if not entries:
        raise ValueError(f"No time-dependent arrays were found in {root}")

    observed_sizes = sorted({int(entry["time_size"]) for entry in entries})
    resolved_target = int(target_time_size) if target_time_size is not None else min(observed_sizes)
    if resolved_target <= 0:
        raise ValueError(f"Invalid target_time_size {resolved_target} for {root}")

    touched_arrays: list[str] = []
    for entry in entries:
        if int(entry["time_size"]) <= resolved_target:
            continue
        zarray_path = Path(entry["zarray_path"])
        zarray = _read_json_file(zarray_path)
        zarray["shape"][int(entry["time_dim_index"])] = resolved_target
        _write_json_file(zarray_path, zarray)
        touched_arrays.append(str(entry["name"]))

    if (root / ".zmetadata").exists():
        (root / ".zmetadata").unlink()

    import zarr

    zarr.consolidate_metadata(str(root))

    summary = {
        "zarr_path": str(root),
        "observed_time_sizes": observed_sizes,
        "target_time_size": resolved_target,
        "touched_arrays": touched_arrays,
        "touched_array_count": len(touched_arrays),
    }
    if verbose:
        print(
            "[repair_local_zarr_time_consistency] "
            f"trimmed {len(touched_arrays)} arrays to time_size={resolved_target}"
        )
    return summary


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


def prepare_arco_spatial_dataarray(
    data_array: xr.DataArray,
    *,
    latitude_descending: bool = True,
) -> xr.DataArray:
    ordered_dims = [dim for dim in ("time", "level", "latitude", "longitude") if dim in data_array.dims]
    data_array = data_array.transpose(*ordered_dims)
    if (
        "latitude" in data_array.coords
        and latitude_descending
        and data_array["latitude"].values.shape[0] > 1
        and data_array["latitude"].values[0] < data_array["latitude"].values[-1]
    ):
        data_array = data_array.isel(latitude=slice(None, None, -1))
    return data_array.astype(np.float32)


def build_fuxi_derived_static_maps(
    latitude: np.ndarray | Sequence[float],
    longitude: np.ndarray | Sequence[float],
) -> dict[str, np.ndarray]:
    latitude_values = np.asarray(latitude, dtype=np.float32)
    longitude_values = np.asarray(longitude, dtype=np.float32)
    if latitude_values.ndim != 1:
        raise ValueError(f"Expected 1D latitude coordinates, got shape {latitude_values.shape}")
    if longitude_values.ndim != 1:
        raise ValueError(f"Expected 1D longitude coordinates, got shape {longitude_values.shape}")

    lat_grid, lon_grid = np.meshgrid(latitude_values, longitude_values, indexing="ij")
    return {
        "cos_latitude": np.cos(np.deg2rad(lat_grid)).astype(np.float32),
        "cos_longitude": np.cos(np.deg2rad(lon_grid)).astype(np.float32),
        "sin_longitude": np.sin(np.deg2rad(lon_grid)).astype(np.float32),
    }


def load_arco_static_source_maps(
    dataset_url: str | Path = DEFAULT_ARCO_ERA5_URL,
    *,
    orography_source: str = "geopotential_at_surface",
    convert_geopotential_to_height: bool = True,
    latitude_descending: bool = True,
    gcs_token: str = "anon",
) -> dict[str, np.ndarray]:
    ds = open_arco_era5_dataset(dataset_url, gcs_token=gcs_token)

    land_sea_mask = prepare_arco_spatial_dataarray(
        ds["land_sea_mask"],
        latitude_descending=latitude_descending,
    ).load().values
    orography = prepare_arco_spatial_dataarray(
        ds[orography_source],
        latitude_descending=latitude_descending,
    ).load().values
    if convert_geopotential_to_height and orography_source == "geopotential_at_surface":
        orography = orography / _STANDARD_GRAVITY

    latitude = prepare_arco_spatial_dataarray(
        ds["latitude"],
        latitude_descending=latitude_descending,
    ).values
    longitude = prepare_arco_spatial_dataarray(
        ds["longitude"],
        latitude_descending=latitude_descending,
    ).values
    return {
        "land_sea_mask": land_sea_mask.astype(np.float32),
        "orography": orography.astype(np.float32),
        "latitude": latitude.astype(np.float32),
        "longitude": longitude.astype(np.float32),
    }


def build_fuxi_static_maps(
    dataset_url: str | Path = DEFAULT_ARCO_ERA5_URL,
    *,
    static_variables: Sequence[str] = FUXI_STATIC_VARIABLES,
    orography_source: str = "geopotential_at_surface",
    convert_geopotential_to_height: bool = True,
    latitude_descending: bool = True,
    gcs_token: str = "anon",
) -> dict[str, np.ndarray]:
    source_maps = load_arco_static_source_maps(
        dataset_url,
        orography_source=orography_source,
        convert_geopotential_to_height=convert_geopotential_to_height,
        latitude_descending=latitude_descending,
        gcs_token=gcs_token,
    )
    derived_maps = build_fuxi_derived_static_maps(
        source_maps["latitude"],
        source_maps["longitude"],
    )
    all_maps = {
        "land_sea_mask": source_maps["land_sea_mask"],
        "orography": source_maps["orography"],
        **derived_maps,
    }
    return {name: all_maps[name] for name in static_variables}


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

    ds = open_arco_era5_dataset(dataset_url, gcs_token="anon")
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
    include_sample_metadata: bool = False
    dynamic_ram_cache_time_steps: int = 32
    dynamic_prefetch_block_time_steps: int = _DEFAULT_DYNAMIC_PREFETCH_BLOCK_TIME_STEPS
    apply_normalization: bool = True
    normalization_stats_path: Path | None = _DEFAULT_NORMALIZATION_STATS_PATH
    normalization_force_recompute: bool = False
    normalization_fit_sample_count: int = _DEFAULT_NORMALIZATION_FIT_SAMPLE_COUNT
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
        if self.dynamic_ram_cache_time_steps < 0:
            raise ValueError(
                "dynamic_ram_cache_time_steps must be non-negative, "
                f"got {self.dynamic_ram_cache_time_steps}"
            )
        if self.dynamic_prefetch_block_time_steps <= 0:
            raise ValueError(
                "dynamic_prefetch_block_time_steps must be positive, "
                f"got {self.dynamic_prefetch_block_time_steps}"
            )
        if self.apply_normalization and self.normalization_fit_sample_count <= 0:
            raise ValueError(
                "normalization_fit_sample_count must be positive when apply_normalization is true, "
                f"got {self.normalization_fit_sample_count}"
            )

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
            include_sample_metadata=bool(data.get("include_sample_metadata", False)),
            dynamic_ram_cache_time_steps=int(data.get("dynamic_ram_cache_time_steps", 32)),
            dynamic_prefetch_block_time_steps=int(
                data.get(
                    "dynamic_prefetch_block_time_steps",
                    _DEFAULT_DYNAMIC_PREFETCH_BLOCK_TIME_STEPS,
                )
            ),
            apply_normalization=bool(data.get("apply_normalization", True)),
            normalization_stats_path=_to_optional_resolved_path(
                data.get("normalization_stats_path", str(_DEFAULT_NORMALIZATION_STATS_PATH)),
                config_path=resolved_config_path,
            ),
            normalization_force_recompute=bool(data.get("normalization_force_recompute", False)),
            normalization_fit_sample_count=int(
                data.get("normalization_fit_sample_count", _DEFAULT_NORMALIZATION_FIT_SAMPLE_COUNT)
            ),
            gcs_token=str(data.get("gcs_token", "anon")),
            start_time=_as_timestamp(data.get("start_time")),
            end_time=_as_timestamp(data.get("end_time")),
            config_path=resolved_config_path,
        )

    @property
    def channel_names(self) -> list[str]:
        return build_fuxi_channel_names(self.pressure_levels)


@dataclass(frozen=True)
class ArcoEra5DownloadPlan:
    source_pressure_variables: tuple[str, ...]
    source_surface_variables: tuple[str, ...]
    source_static_variables: tuple[str, ...]
    output_dynamic_variables: tuple[str, ...]
    derive_relative_humidity: bool
    pressure_levels: tuple[int, ...]

    @property
    def source_variables(self) -> tuple[str, ...]:
        ordered: list[str] = []
        for name in (
            *self.source_pressure_variables,
            *self.source_surface_variables,
            *self.source_static_variables,
        ):
            if name not in ordered:
                ordered.append(name)
        return tuple(ordered)


@dataclass(frozen=True)
class ArcoEra5DownloadWindow:
    anchor_start: pd.Timestamp | None
    anchor_end: pd.Timestamp | None
    raw_start: pd.Timestamp | None
    raw_end: pd.Timestamp | None

    def summary(self) -> dict[str, str | None]:
        return {
            "anchor_start": None if self.anchor_start is None else str(self.anchor_start),
            "anchor_end": None if self.anchor_end is None else str(self.anchor_end),
            "raw_start": None if self.raw_start is None else str(self.raw_start),
            "raw_end": None if self.raw_end is None else str(self.raw_end),
        }


@dataclass(frozen=True)
class ArcoEra5NormalizationStats:
    version: int
    dataset_url: str
    dynamic_channel_names: tuple[str, ...]
    dynamic_transform_kinds: tuple[str, ...]
    dynamic_mean: tuple[float, ...]
    dynamic_std: tuple[float, ...]
    static_channel_names: tuple[str, ...]
    static_transform_kinds: tuple[str, ...]
    static_mean: tuple[float, ...]
    static_std: tuple[float, ...]
    fit_sample_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "dataset_url": self.dataset_url,
            "dynamic_channel_names": list(self.dynamic_channel_names),
            "dynamic_transform_kinds": list(self.dynamic_transform_kinds),
            "dynamic_mean": list(self.dynamic_mean),
            "dynamic_std": list(self.dynamic_std),
            "static_channel_names": list(self.static_channel_names),
            "static_transform_kinds": list(self.static_transform_kinds),
            "static_mean": list(self.static_mean),
            "static_std": list(self.static_std),
            "fit_sample_count": int(self.fit_sample_count),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ArcoEra5NormalizationStats":
        return cls(
            version=int(payload["version"]),
            dataset_url=str(payload["dataset_url"]),
            dynamic_channel_names=_to_str_tuple(payload["dynamic_channel_names"]),
            dynamic_transform_kinds=_to_str_tuple(payload["dynamic_transform_kinds"]),
            dynamic_mean=tuple(float(value) for value in payload["dynamic_mean"]),
            dynamic_std=tuple(float(value) for value in payload["dynamic_std"]),
            static_channel_names=_to_str_tuple(payload["static_channel_names"]),
            static_transform_kinds=_to_str_tuple(payload["static_transform_kinds"]),
            static_mean=tuple(float(value) for value in payload["static_mean"]),
            static_std=tuple(float(value) for value in payload["static_std"]),
            fit_sample_count=int(payload["fit_sample_count"]),
        )


@dataclass(frozen=True)
class _DownloadStepRequest:
    step_number: int
    time_index: int
    time_value: pd.Timestamp
    variable_names: tuple[str, ...]
    include_static_sources: bool


@dataclass(frozen=True)
class _DownloadVariableTask:
    step_number: int
    variable_name: str
    time_index: int | None
    time_value: pd.Timestamp | None


@dataclass(frozen=True)
class _LoadedDownloadVariable:
    step_number: int
    variable_name: str
    values: np.ndarray
    dims: tuple[str, ...]
    coords: dict[str, np.ndarray]


def build_arco_era5_download_plan(
    available_variables: Sequence[str],
    config: ArcoEra5FuXiDataConfig | None = None,
    *,
    include_static_sources: bool = True,
) -> ArcoEra5DownloadPlan:
    data_config = config or ArcoEra5FuXiDataConfig.from_yaml()
    available = {str(name) for name in available_variables}
    missing: list[str] = []
    source_pressure_variables: list[str] = []
    source_surface_variables: list[str] = []
    source_static_variables: list[str] = []
    derive_relative_humidity = False

    def add_unique(target: list[str], name: str) -> None:
        if name not in target:
            target.append(name)

    for variable_name in data_config.upper_air_variables:
        if variable_name == "relative_humidity":
            if "relative_humidity" in available:
                add_unique(source_pressure_variables, "relative_humidity")
            elif {"specific_humidity", "temperature"}.issubset(available):
                derive_relative_humidity = True
                add_unique(source_pressure_variables, "temperature")
                add_unique(source_pressure_variables, "specific_humidity")
            else:
                missing.append("relative_humidity")
            continue

        if variable_name not in available:
            missing.append(variable_name)
            continue
        add_unique(source_pressure_variables, variable_name)

    for variable_name in data_config.surface_variables:
        if variable_name not in available:
            missing.append(variable_name)
            continue
        add_unique(source_surface_variables, variable_name)

    if include_static_sources:
        for variable_name in ("land_sea_mask", data_config.orography_source):
            if variable_name not in available:
                missing.append(variable_name)
                continue
            add_unique(source_static_variables, variable_name)

    if missing:
        missing_names = ", ".join(sorted(set(missing)))
        raise KeyError(f"Source dataset is missing required variables: {missing_names}")

    return ArcoEra5DownloadPlan(
        source_pressure_variables=tuple(source_pressure_variables),
        source_surface_variables=tuple(source_surface_variables),
        source_static_variables=tuple(source_static_variables),
        output_dynamic_variables=tuple((*data_config.upper_air_variables, *data_config.surface_variables)),
        derive_relative_humidity=derive_relative_humidity,
        pressure_levels=data_config.pressure_levels,
    )


def resolve_arco_era5_download_window(
    config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
    *,
    section_names: Sequence[str] = ("train_main", "train_intrinsic"),
) -> ArcoEra5DownloadWindow:
    resolved_config_path, config = load_yaml_config(config_path)
    data_config = ArcoEra5FuXiDataConfig.from_yaml(resolved_config_path)

    start_candidates: list[pd.Timestamp] = []
    end_candidates: list[pd.Timestamp] = []

    for section_name in section_names:
        section_value = config.get(section_name)
        if not isinstance(section_value, dict):
            continue
        for start_key in ("train_start_time", "val_start_time"):
            start_value = _as_timestamp(section_value.get(start_key))
            if start_value is not None:
                start_candidates.append(start_value)
        for end_key in ("train_end_time", "val_end_time"):
            end_value = _as_timestamp(section_value.get(end_key))
            if end_value is not None:
                end_candidates.append(end_value)

    anchor_start = min(start_candidates) if start_candidates else None
    anchor_end = max(end_candidates) if end_candidates else None

    raw_start = None
    if anchor_start is not None:
        raw_start = anchor_start + pd.Timedelta(hours=min(data_config.input_time_offsets_hours))

    raw_end = None
    if anchor_end is not None:
        raw_end = anchor_end + pd.Timedelta(hours=data_config.lead_time_hours * data_config.forecast_steps)

    return ArcoEra5DownloadWindow(
        anchor_start=anchor_start,
        anchor_end=anchor_end,
        raw_start=raw_start,
        raw_end=raw_end,
    )


def _build_download_chunk_dataset(
    source_chunk: xr.Dataset,
    plan: ArcoEra5DownloadPlan,
    data_config: ArcoEra5FuXiDataConfig,
    *,
    include_static_sources: bool,
) -> xr.Dataset:
    data_vars: dict[str, xr.DataArray] = {}
    level_values = source_chunk["level"].sel(level=list(plan.pressure_levels))

    for variable_name in plan.source_pressure_variables:
        array = source_chunk[variable_name].sel(level=list(plan.pressure_levels)).astype(np.float32)
        data_vars[variable_name] = array

    if plan.derive_relative_humidity:
        relative_humidity = specific_humidity_to_relative_humidity(
            data_vars["specific_humidity"],
            data_vars["temperature"],
            level_values,
        )
        relative_humidity.name = "relative_humidity"
        data_vars["relative_humidity"] = relative_humidity
        if "specific_humidity" not in plan.output_dynamic_variables:
            del data_vars["specific_humidity"]

    for variable_name in plan.source_surface_variables:
        data_vars[variable_name] = source_chunk[variable_name].astype(np.float32)

    if include_static_sources:
        for variable_name in plan.source_static_variables:
            data_vars[variable_name] = source_chunk[variable_name].astype(np.float32)

    output = xr.Dataset(data_vars=data_vars)
    if data_config.latitude_descending and "latitude" in output.coords:
        latitude_values = output["latitude"].values
        if latitude_values.shape[0] > 1 and latitude_values[0] < latitude_values[-1]:
            output = output.isel(latitude=slice(None, None, -1))
    return output


def _ordered_unique_names(values: Sequence[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    for value in values:
        if value not in ordered:
            ordered.append(value)
    return tuple(ordered)


def _download_output_attrs(
    *,
    data_config: ArcoEra5FuXiDataConfig,
    download_plan: ArcoEra5DownloadPlan,
    requested_start: pd.Timestamp | None,
    requested_end: pd.Timestamp | None,
) -> dict[str, Any]:
    return {
        "source_dataset_url": describe_arco_era5_dataset_location(data_config.dataset_url),
        "source_gcs_token": data_config.gcs_token,
        "pressure_levels": list(download_plan.pressure_levels),
        "derived_relative_humidity": bool(download_plan.derive_relative_humidity),
        "requested_start_time": None if requested_start is None else str(requested_start),
        "requested_end_time": None if requested_end is None else str(requested_end),
    }


def _download_output_coordinate_values(
    selected: xr.Dataset,
    data_config: ArcoEra5FuXiDataConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    level_values = np.asarray(selected["level"].values)
    latitude_values = np.asarray(selected["latitude"].values)
    longitude_values = np.asarray(selected["longitude"].values)
    if (
        data_config.latitude_descending
        and latitude_values.shape[0] > 1
        and latitude_values[0] < latitude_values[-1]
    ):
        latitude_values = latitude_values[::-1]
    return level_values, latitude_values, longitude_values


def _create_zarr_coordinate(root: Any, name: str, values: np.ndarray) -> None:
    chunk_size = max(1, int(values.shape[0]))
    array = root.create_dataset(
        name,
        data=values,
        chunks=(chunk_size,),
        fill_value=None,
        overwrite=True,
    )
    array.attrs["_ARRAY_DIMENSIONS"] = [name]


def _create_resizable_zarr_time(root: Any, *, total_time_size: int) -> None:
    chunk_size = max(1, min(int(total_time_size), 4096))
    array = root.create_dataset(
        "time",
        shape=(0,),
        chunks=(chunk_size,),
        dtype=np.dtype("datetime64[ns]"),
        fill_value=None,
        overwrite=True,
    )
    array.attrs["_ARRAY_DIMENSIONS"] = ["time"]


def _download_pressure_output_variables(
    download_plan: ArcoEra5DownloadPlan,
    data_config: ArcoEra5FuXiDataConfig,
) -> tuple[str, ...]:
    pressure_names = set(data_config.upper_air_variables)
    return tuple(name for name in download_plan.output_dynamic_variables if name in pressure_names)


def _download_surface_output_variables(
    download_plan: ArcoEra5DownloadPlan,
    data_config: ArcoEra5FuXiDataConfig,
) -> tuple[str, ...]:
    surface_names = set(data_config.surface_variables)
    return tuple(name for name in download_plan.output_dynamic_variables if name in surface_names)


def _initialize_download_zarr_store(
    output_path: Path,
    selected: xr.Dataset,
    download_plan: ArcoEra5DownloadPlan,
    data_config: ArcoEra5FuXiDataConfig,
    *,
    include_static_sources: bool,
    total_time_size: int,
    attrs: dict[str, Any],
) -> None:
    import zarr

    root = zarr.open_group(str(output_path), mode="w")
    root.attrs.update(attrs)

    level_values, latitude_values, longitude_values = _download_output_coordinate_values(selected, data_config)
    _create_resizable_zarr_time(root, total_time_size=total_time_size)
    _create_zarr_coordinate(root, "level", level_values)
    _create_zarr_coordinate(root, "latitude", latitude_values)
    _create_zarr_coordinate(root, "longitude", longitude_values)

    latitude_size = int(latitude_values.shape[0])
    longitude_size = int(longitude_values.shape[0])
    level_size = int(level_values.shape[0])
    pressure_shape = (0, level_size, latitude_size, longitude_size)
    pressure_chunks = (1, level_size, latitude_size, longitude_size)
    surface_shape = (0, latitude_size, longitude_size)
    surface_chunks = (1, latitude_size, longitude_size)
    static_shape = (latitude_size, longitude_size)
    static_chunks = static_shape

    for variable_name in _download_pressure_output_variables(download_plan, data_config):
        array = root.create_dataset(
            variable_name,
            shape=pressure_shape,
            chunks=pressure_chunks,
            dtype=np.float32,
            fill_value=np.nan,
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["time", "level", "latitude", "longitude"]

    for variable_name in _download_surface_output_variables(download_plan, data_config):
        array = root.create_dataset(
            variable_name,
            shape=surface_shape,
            chunks=surface_chunks,
            dtype=np.float32,
            fill_value=np.nan,
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["time", "latitude", "longitude"]

    if include_static_sources:
        for variable_name in download_plan.source_static_variables:
            array = root.create_dataset(
                variable_name,
                shape=static_shape,
                chunks=static_chunks,
                dtype=np.float32,
                fill_value=np.nan,
            )
            array.attrs["_ARRAY_DIMENSIONS"] = ["latitude", "longitude"]


def _remove_download_zarr_consolidated_metadata(output_path: Path) -> None:
    metadata_path = output_path / ".zmetadata"
    if metadata_path.exists():
        metadata_path.unlink()


def _open_download_zarr_store_for_writes(output_path: Path, attrs: dict[str, Any]) -> Any:
    import zarr

    _remove_download_zarr_consolidated_metadata(output_path)
    root = zarr.open_group(str(output_path), mode="r+")
    root.attrs.update(attrs)
    return root


def _resize_zarr_time_array(array: Any, target_time_size: int) -> None:
    if int(array.shape[0]) >= target_time_size:
        return
    array.resize((target_time_size, *array.shape[1:]))


def _encode_download_time_value(time_array: Any, value: pd.Timestamp) -> Any:
    if np.issubdtype(np.dtype(time_array.dtype), np.datetime64):
        return np.datetime64(value.to_datetime64(), "ns")

    units = time_array.attrs.get("units")
    calendar = time_array.attrs.get("calendar")
    if units is not None:
        encoded, _, _ = xr.coding.times.encode_cf_datetime(
            np.asarray([np.datetime64(value.to_datetime64(), "ns")]),
            units=str(units),
            calendar=None if calendar is None else str(calendar),
            dtype=np.dtype(time_array.dtype),
        )
        return encoded[0]

    return np.asarray([np.datetime64(value.to_datetime64(), "ns")]).astype(time_array.dtype)[0]


def _write_download_step_to_zarr(
    root: Any,
    output_chunk: xr.Dataset,
    step_request: _DownloadStepRequest,
    download_plan: ArcoEra5DownloadPlan,
) -> None:
    target_time_size = int(step_request.time_index) + 1
    dynamic_variable_names = _ordered_unique_names(download_plan.output_dynamic_variables)

    for variable_name in dynamic_variable_names:
        if variable_name not in output_chunk.data_vars:
            raise KeyError(f"Prepared timestep is missing output variable {variable_name!r}")
        array = root[variable_name]
        _resize_zarr_time_array(array, target_time_size)
        prepared = _select_prepared_output_time(output_chunk[variable_name], step_request)
        target_dims = tuple(array.attrs.get("_ARRAY_DIMENSIONS", prepared.dims))
        values = np.asarray(prepared.transpose(*target_dims).data, dtype=np.float32)
        if values.shape[0] != 1:
            raise ValueError(
                f"Expected one time step for {variable_name!r}, got shape {values.shape} "
                f"at step={step_request.step_number} time_index={step_request.time_index} "
                f"time_value={step_request.time_value}"
            )
        array[int(step_request.time_index) : target_time_size, ...] = values

    if step_request.include_static_sources:
        for variable_name in download_plan.source_static_variables:
            if variable_name not in output_chunk.data_vars:
                raise KeyError(f"Prepared timestep is missing static variable {variable_name!r}")
            array = root[variable_name]
            prepared = _drop_single_time_dimension(
                _select_prepared_output_time(output_chunk[variable_name], step_request)
            )
            target_dims = tuple(array.attrs.get("_ARRAY_DIMENSIONS", prepared.dims))
            root[variable_name][...] = np.asarray(
                prepared.transpose(*target_dims).data,
                dtype=np.float32,
            )

    time_array = root["time"]
    _resize_zarr_time_array(time_array, target_time_size)
    time_array[int(step_request.time_index)] = _encode_download_time_value(
        time_array,
        step_request.time_value,
    )


def _detach_loaded_download_array(
    variable_name: str,
    step_number: int,
    array: xr.DataArray,
) -> _LoadedDownloadVariable:
    loaded = array.load()
    coords: dict[str, np.ndarray] = {}
    for coord_name, coord in loaded.coords.items():
        if all(dim in loaded.dims for dim in coord.dims):
            coords[str(coord_name)] = np.asarray(coord.values)
    return _LoadedDownloadVariable(
        step_number=step_number,
        variable_name=variable_name,
        values=np.asarray(loaded.data),
        dims=tuple(str(dim) for dim in loaded.dims),
        coords=coords,
    )


def _select_download_task_time(array: xr.DataArray, task: _DownloadVariableTask) -> xr.DataArray:
    if task.time_index is None or "time" not in array.dims:
        return array

    if task.time_value is not None:
        try:
            return array.sel(time=[task.time_value])
        except (KeyError, TypeError, ValueError):
            pass
    return array.isel(time=slice(int(task.time_index), int(task.time_index) + 1))


def _validate_loaded_download_variable(loaded: _LoadedDownloadVariable, task: _DownloadVariableTask) -> None:
    if task.time_index is None or "time" not in loaded.dims:
        return
    time_axis = loaded.dims.index("time")
    time_size = int(loaded.values.shape[time_axis])
    if time_size != 1:
        raise ValueError(
            "Download worker loaded an unexpected number of time steps for "
            f"{task.variable_name!r}: expected 1 at step={task.step_number} "
            f"time_index={task.time_index} time_value={task.time_value}, "
            f"got shape {loaded.values.shape}."
        )


def _select_prepared_output_time(
    array: xr.DataArray,
    step_request: _DownloadStepRequest,
) -> xr.DataArray:
    if "time" not in array.dims:
        return array
    time_size = int(array.sizes["time"])
    if time_size == 1:
        return array
    try:
        selected = array.sel(time=[step_request.time_value])
    except (KeyError, TypeError, ValueError):
        selected = array.isel(time=slice(int(step_request.time_index), int(step_request.time_index) + 1))
    if int(selected.sizes["time"]) != 1:
        raise ValueError(
            f"Expected one time step for {array.name!r}, got shape {tuple(array.shape)} "
            f"at step={step_request.step_number} time_index={step_request.time_index} "
            f"time_value={step_request.time_value}."
        )
    return selected


def _drop_single_time_dimension(array: xr.DataArray) -> xr.DataArray:
    if "time" not in array.dims:
        return array
    if int(array.sizes["time"]) != 1:
        raise ValueError(f"Expected a single time step for {array.name!r}, got shape {tuple(array.shape)}")
    return array.isel(time=0, drop=True)


def _loaded_download_variable_to_data_array(loaded: _LoadedDownloadVariable) -> xr.DataArray:
    return xr.DataArray(
        loaded.values,
        dims=loaded.dims,
        coords=loaded.coords,
        name=loaded.variable_name,
    )


def _get_download_thread_local_dataset(
    dataset_url: str | Path,
    *,
    gcs_token: str,
    requested_start: pd.Timestamp | None,
    requested_end: pd.Timestamp | None,
    pressure_levels: Sequence[int],
) -> xr.Dataset:
    cache = getattr(_DOWNLOAD_DATASET_CACHE, "datasets", None)
    if cache is None:
        cache = {}
        _DOWNLOAD_DATASET_CACHE.datasets = cache
    key = (
        str(dataset_url),
        str(gcs_token),
        None if requested_start is None else str(requested_start),
        None if requested_end is None else str(requested_end),
        tuple(int(level) for level in pressure_levels),
    )
    dataset = cache.get(key)
    if dataset is None:
        dataset = open_arco_era5_dataset(dataset_url, gcs_token=gcs_token)
        if requested_start is not None or requested_end is not None:
            dataset = dataset.sel(time=slice(requested_start, requested_end))
        if "level" in dataset.coords:
            dataset = dataset.sel(level=list(pressure_levels))
        cache[key] = dataset
    return dataset


def _load_download_variable_task(
    dataset_url: str | Path,
    *,
    gcs_token: str,
    requested_start: pd.Timestamp | None,
    requested_end: pd.Timestamp | None,
    pressure_levels: Sequence[int],
    task: _DownloadVariableTask,
) -> _LoadedDownloadVariable:
    source_dataset = _get_download_thread_local_dataset(
        dataset_url,
        gcs_token=gcs_token,
        requested_start=requested_start,
        requested_end=requested_end,
        pressure_levels=pressure_levels,
    )
    array = source_dataset[task.variable_name]
    array = _select_download_task_time(array, task)
    loaded = _detach_loaded_download_array(task.variable_name, task.step_number, array)
    _validate_loaded_download_variable(loaded, task)
    return loaded


def _build_download_step_requests(
    *,
    time_values: pd.Index,
    start_index: int,
    stop_index: int,
    plan: ArcoEra5DownloadPlan,
    include_static_sources: bool,
) -> list[_DownloadStepRequest]:
    requests: list[_DownloadStepRequest] = []
    for step_number, time_index in enumerate(range(start_index, stop_index), start=1):
        variable_names = list(plan.source_pressure_variables)
        variable_names.extend(plan.source_surface_variables)
        include_static_for_step = include_static_sources and time_index == 0 and step_number == 1
        if include_static_for_step:
            variable_names.extend(plan.source_static_variables)
        requests.append(
            _DownloadStepRequest(
                step_number=step_number,
                time_index=time_index,
                time_value=pd.Timestamp(time_values[time_index]),
                variable_names=_ordered_unique_names(variable_names),
                include_static_sources=include_static_for_step,
            )
        )
    return requests


def _build_download_variable_tasks(
    step_request: _DownloadStepRequest,
    plan: ArcoEra5DownloadPlan,
) -> tuple[_DownloadVariableTask, ...]:
    tasks: list[_DownloadVariableTask] = []
    for variable_name in step_request.variable_names:
        tasks.append(
            _DownloadVariableTask(
                step_number=step_request.step_number,
                variable_name=variable_name,
                time_index=step_request.time_index,
                time_value=step_request.time_value,
            )
        )
    return tuple(tasks)


def _load_download_source_chunk(
    source_dataset: xr.Dataset,
    plan: ArcoEra5DownloadPlan,
    *,
    start_index: int,
    stop_index: int,
    include_static_sources: bool,
    variable_download_workers: int | None = None,
    surface_variable_download_workers: int | None = None,
) -> xr.Dataset:
    if variable_download_workers is None:
        variable_download_workers = surface_variable_download_workers
    if variable_download_workers is None:
        variable_download_workers = _DEFAULT_VARIABLE_DOWNLOAD_WORKERS
    if variable_download_workers <= 0:
        raise ValueError(
            "variable_download_workers must be positive, "
            f"got {variable_download_workers}"
        )

    variable_names = [
        *plan.source_pressure_variables,
        *plan.source_surface_variables,
    ]
    if include_static_sources:
        variable_names.extend(plan.source_static_variables)
    variable_names = list(_ordered_unique_names(variable_names))

    def load_variable(variable_name: str) -> tuple[str, xr.DataArray]:
        array = source_dataset[variable_name]
        if "time" in array.dims:
            array = array.isel(time=slice(start_index, stop_index))
        return variable_name, array.load()

    if int(variable_download_workers) == 1 or len(variable_names) <= 1:
        loaded = [load_variable(variable_name) for variable_name in variable_names]
    else:
        with ThreadPoolExecutor(max_workers=int(variable_download_workers)) as executor:
            loaded = list(executor.map(load_variable, variable_names))
    return xr.Dataset({name: array for name, array in loaded})


def download_arco_era5_subset(
    output_path: str | Path,
    *,
    config_path: str | Path = DEFAULT_MODEL_CONFIG_PATH,
    dataset_url: str | Path | None = None,
    start_time: str | pd.Timestamp | None = None,
    end_time: str | pd.Timestamp | None = None,
    include_static_sources: bool = True,
    overwrite: bool = False,
    resume: bool = False,
    chunk_size: int = 24,
    gcs_token: str | None = None,
    verbose: bool = True,
    show_progress: bool = True,
    repair_inconsistent_resume_store: bool = True,
    variable_download_workers: int | None = None,
    prefetch_chunk_count: int = _DEFAULT_DOWNLOAD_PREFETCH_CHUNK_COUNT,
    surface_variable_download_workers: int | None = None,
) -> dict[str, Any]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if variable_download_workers is None:
        variable_download_workers = surface_variable_download_workers
    if variable_download_workers is None:
        variable_download_workers = _DEFAULT_VARIABLE_DOWNLOAD_WORKERS
    if variable_download_workers <= 0:
        raise ValueError(
            "variable_download_workers must be positive, "
            f"got {variable_download_workers}"
        )
    if prefetch_chunk_count <= 0:
        raise ValueError(f"prefetch_chunk_count must be positive, got {prefetch_chunk_count}")

    data_config = ArcoEra5FuXiDataConfig.from_yaml(config_path)
    if dataset_url is not None:
        data_config = replace(data_config, dataset_url=str(dataset_url))
    if gcs_token is not None:
        data_config = replace(data_config, gcs_token=str(gcs_token))

    download_window = resolve_arco_era5_download_window(config_path)
    requested_start = _as_timestamp(start_time) if start_time is not None else download_window.raw_start
    requested_end = _as_timestamp(end_time) if end_time is not None else download_window.raw_end

    source_dataset = open_arco_era5_dataset(data_config.dataset_url, gcs_token=data_config.gcs_token)
    download_plan = build_arco_era5_download_plan(
        tuple(source_dataset.data_vars),
        data_config,
        include_static_sources=include_static_sources,
    )

    selected = source_dataset
    if requested_start is not None or requested_end is not None:
        selected = selected.sel(time=slice(requested_start, requested_end))
    selected = selected.sel(level=list(download_plan.pressure_levels))

    if selected.sizes.get("time", 0) == 0:
        raise ValueError(
            "Selected ARCO ERA5 slice is empty. "
            f"Requested time range: {requested_start!s} -> {requested_end!s}"
        )

    time_size = int(selected.sizes["time"])
    time_values = pd.Index(pd.to_datetime(selected["time"].values))
    resolved_output_path = Path(output_path).resolve()
    start_index = 0
    resumed_from_time_steps = 0
    already_complete = False
    if resolved_output_path.exists():
        if overwrite:
            if resolved_output_path.is_dir():
                shutil.rmtree(resolved_output_path)
            else:
                resolved_output_path.unlink()
        elif resume:
            try:
                existing = xr.open_zarr(resolved_output_path, consolidated=False)
            except ValueError as exc:
                if (
                    repair_inconsistent_resume_store
                    and "conflicting sizes for dimension 'time'" in str(exc)
                ):
                    if verbose:
                        print(
                            "[download_arco_era5_subset] detected an inconsistent partial Zarr store. "
                            "Attempting automatic time-axis repair before resuming."
                        )
                    repair_summary = repair_local_zarr_time_consistency(
                        resolved_output_path,
                        verbose=verbose,
                    )
                    if verbose:
                        print(
                            "[download_arco_era5_subset] repair summary: "
                            f"{json.dumps(repair_summary, sort_keys=True)}"
                        )
                    existing = xr.open_zarr(resolved_output_path, consolidated=False)
                else:
                    raise
            existing_time_size = int(existing.sizes.get("time", 0))
            if existing_time_size > time_size:
                raise ValueError(
                    f"Existing output has {existing_time_size} time steps, "
                    f"but the requested slice has only {time_size}."
                )

            expected_dynamic_variables = set(download_plan.output_dynamic_variables)
            existing_variables = set(existing.data_vars)
            missing_dynamic = expected_dynamic_variables - existing_variables
            if missing_dynamic:
                missing_names = ", ".join(sorted(missing_dynamic))
                raise ValueError(
                    f"Existing output cannot be resumed because it is missing dynamic variables: {missing_names}"
                )

            if include_static_sources:
                expected_static_variables = set(download_plan.source_static_variables)
                missing_static = expected_static_variables - existing_variables
                if missing_static:
                    missing_names = ", ".join(sorted(missing_static))
                    raise ValueError(
                        f"Existing output cannot be resumed because it is missing static variables: {missing_names}"
                    )

            if int(existing.sizes.get("latitude", -1)) != int(selected.sizes["latitude"]):
                raise ValueError("Existing output latitude size does not match the requested source slice.")
            if int(existing.sizes.get("longitude", -1)) != int(selected.sizes["longitude"]):
                raise ValueError("Existing output longitude size does not match the requested source slice.")

            existing_times = pd.Index(pd.to_datetime(existing["time"].values))
            source_prefix_times = pd.Index(
                pd.to_datetime(selected["time"].isel(time=slice(0, existing_time_size)).values)
            )
            if not existing_times.equals(source_prefix_times):
                raise ValueError(
                    "Existing output time coordinates are not a prefix of the requested source slice, "
                    "so automatic resume would risk corrupting the dataset."
                )

            start_index = existing_time_size
            resumed_from_time_steps = existing_time_size
            already_complete = existing_time_size == time_size
        else:
            raise FileExistsError(
                f"Output path already exists: {resolved_output_path}. "
                "Pass overwrite=True, use --overwrite, or resume=True / --resume."
            )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    remaining_time_steps = max(time_size - start_index, 0)
    total_chunks = (remaining_time_steps + chunk_size - 1) // chunk_size if remaining_time_steps else 0
    first_chunk_time_size = 0
    output_attrs = _download_output_attrs(
        data_config=data_config,
        download_plan=download_plan,
        requested_start=requested_start,
        requested_end=requested_end,
    )
    if already_complete:
        summary = {
            "output_path": str(resolved_output_path),
            "dataset_url": describe_arco_era5_dataset_location(data_config.dataset_url),
            "gcs_token": data_config.gcs_token,
            "time_start": str(pd.Timestamp(selected["time"].values[0])),
            "time_end": str(pd.Timestamp(selected["time"].values[-1])),
            "time_steps": time_size,
            "first_chunk_time_steps": first_chunk_time_size,
            "chunk_size": chunk_size,
            "chunk_count": total_chunks,
            "latitude_size": int(selected.sizes["latitude"]),
            "longitude_size": int(selected.sizes["longitude"]),
            "pressure_levels": list(download_plan.pressure_levels),
            "source_variables": list(download_plan.source_variables),
            "dynamic_variables": list(download_plan.output_dynamic_variables),
            "static_source_variables": list(download_plan.source_static_variables),
            "derived_relative_humidity": download_plan.derive_relative_humidity,
            "download_window": download_window.summary(),
            "requested_start_time": None if requested_start is None else str(requested_start),
            "requested_end_time": None if requested_end is None else str(requested_end),
            "resume_enabled": resume,
            "resumed_from_time_steps": resumed_from_time_steps,
            "remaining_time_steps": remaining_time_steps,
            "variable_download_workers": int(variable_download_workers),
            "surface_variable_download_workers": int(variable_download_workers),
            "prefetch_chunk_count": int(prefetch_chunk_count),
            "already_complete": True,
        }
        if verbose:
            print(f"[download_arco_era5_subset] output already complete at {resolved_output_path}")
        return summary

    if not resolved_output_path.exists():
        _initialize_download_zarr_store(
            resolved_output_path,
            selected,
            download_plan,
            data_config,
            include_static_sources=include_static_sources,
            total_time_size=time_size,
            attrs=output_attrs,
        )
    output_root = _open_download_zarr_store_for_writes(resolved_output_path, output_attrs)

    progress_bar = None
    if show_progress and tqdm is not None:
        progress_bar = tqdm(
            total=time_size,
            initial=start_index,
            unit="hour",
            desc="ARCO ERA5 download",
        )
    elif show_progress and verbose and tqdm is None:
        print("[download_arco_era5_subset] tqdm is not installed; continuing without a progress bar.")

    step_requests = _build_download_step_requests(
        time_values=time_values,
        start_index=start_index,
        stop_index=time_size,
        plan=download_plan,
        include_static_sources=include_static_sources,
    )
    step_requests_by_number = {request.step_number: request for request in step_requests}
    step_results: dict[int, dict[str, _LoadedDownloadVariable]] = {}
    ready_source_steps: dict[int, xr.Dataset] = {}
    next_submit_index = 0
    next_write_step_number = 1
    prefetch_time_steps = int(prefetch_chunk_count)

    def submit_prefetch_window(
        executor: ThreadPoolExecutor,
        pending_futures: dict[Any, _DownloadVariableTask],
    ) -> None:
        nonlocal next_submit_index
        while next_submit_index < len(step_requests):
            active_step_count = next_submit_index - (next_write_step_number - 1)
            if active_step_count >= prefetch_time_steps:
                break
            step_request = step_requests[next_submit_index]
            for task in _build_download_variable_tasks(step_request, download_plan):
                future = executor.submit(
                    _load_download_variable_task,
                    data_config.dataset_url,
                    gcs_token=data_config.gcs_token,
                    requested_start=requested_start,
                    requested_end=requested_end,
                    pressure_levels=download_plan.pressure_levels,
                    task=task,
                )
                pending_futures[future] = task
            next_submit_index += 1

    try:
        pending_futures: dict[Any, _DownloadVariableTask] = {}
        with ThreadPoolExecutor(max_workers=int(variable_download_workers)) as executor:
            submit_prefetch_window(executor, pending_futures)

            while pending_futures:
                done, _ = wait(tuple(pending_futures.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    pending_futures.pop(future)
                    loaded = future.result()
                    step_number = loaded.step_number
                    variable_name = loaded.variable_name
                    step_request = step_requests_by_number[step_number]
                    step_bucket = step_results.setdefault(step_number, {})
                    step_bucket[variable_name] = loaded
                    if len(step_bucket) == len(step_request.variable_names):
                        ready_source_steps[step_number] = xr.Dataset(
                            {
                                name: _loaded_download_variable_to_data_array(step_bucket[name])
                                for name in step_request.variable_names
                            }
                        )
                        del step_results[step_number]

                while next_write_step_number in ready_source_steps:
                    step_request = step_requests_by_number[next_write_step_number]
                    source_chunk = ready_source_steps.pop(next_write_step_number)
                    output_chunk = _build_download_chunk_dataset(
                        source_chunk,
                        download_plan,
                        data_config,
                        include_static_sources=step_request.include_static_sources,
                    )
                    _write_download_step_to_zarr(
                        output_root,
                        output_chunk,
                        step_request,
                        download_plan,
                    )
                    if step_request.time_index == 0 and next_write_step_number == 1:
                        first_chunk_time_size = int(output_chunk.sizes.get("time", 0))

                    if progress_bar is not None:
                        progress_bar.update(1)
                        progress_bar.set_postfix_str(str(step_request.time_value))

                    if verbose and progress_bar is None:
                        print(
                            f"[download_arco_era5_subset] wrote timestep {step_request.step_number}/{max(remaining_time_steps, 1)} "
                            f"({step_request.time_value})"
                        )

                    del output_chunk
                    del source_chunk
                    next_write_step_number += 1
                    submit_prefetch_window(executor, pending_futures)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    import zarr

    zarr.consolidate_metadata(str(resolved_output_path))

    final_time_values = selected["time"].values
    summary = {
        "output_path": str(resolved_output_path),
        "dataset_url": describe_arco_era5_dataset_location(data_config.dataset_url),
        "gcs_token": data_config.gcs_token,
        "time_start": str(pd.Timestamp(final_time_values[0])),
        "time_end": str(pd.Timestamp(final_time_values[-1])),
        "time_steps": time_size,
        "first_chunk_time_steps": first_chunk_time_size,
        "chunk_size": chunk_size,
        "chunk_count": total_chunks,
        "latitude_size": int(selected.sizes["latitude"]),
        "longitude_size": int(selected.sizes["longitude"]),
        "pressure_levels": list(download_plan.pressure_levels),
        "source_variables": list(download_plan.source_variables),
        "dynamic_variables": list(download_plan.output_dynamic_variables),
        "static_source_variables": list(download_plan.source_static_variables),
        "derived_relative_humidity": download_plan.derive_relative_humidity,
        "download_window": download_window.summary(),
        "requested_start_time": None if requested_start is None else str(requested_start),
        "requested_end_time": None if requested_end is None else str(requested_end),
        "resume_enabled": resume,
        "resumed_from_time_steps": resumed_from_time_steps,
        "remaining_time_steps": remaining_time_steps,
        "variable_download_workers": int(variable_download_workers),
        "surface_variable_download_workers": int(variable_download_workers),
        "prefetch_chunk_count": int(prefetch_chunk_count),
        "already_complete": False,
    }
    if verbose:
        print(f"[download_arco_era5_subset] wrote dataset to {resolved_output_path}")
    return summary


class ArcoEra5FuXiDataset(Dataset[dict[str, Any]]):
    """Lazy remote dataset that produces FuXi-style samples from ARCO ERA5."""

    def __init__(self, config: ArcoEra5FuXiDataConfig | None = None) -> None:
        self.config = config or ArcoEra5FuXiDataConfig.from_yaml()
        self._dataset: xr.Dataset | None = None
        self._time_values: np.ndarray | None = None
        self._static_features: torch.Tensor | None = None
        self._valid_anchor_indices: np.ndarray | None = None
        self._dataset_step_hours: int | None = None
        self._dynamic_ring_array: np.ndarray | None = None
        self._dynamic_ring_start: int | None = None
        self._dynamic_ring_stop: int | None = None
        self._dynamic_chunk_time_steps: int | None = None
        self._dynamic_download_plan: ArcoEra5DownloadPlan | None = None
        self._dynamic_chunk_size_warning_emitted = False
        self._dynamic_prefetch_condition = threading.Condition()
        self._dynamic_prefetch_thread: threading.Thread | None = None
        self._dynamic_prefetch_target_stop: int | None = None
        self._dynamic_prefetch_generation = 0
        self._dynamic_prefetch_error: BaseException | None = None
        self._dynamic_prefetch_shutdown = False
        self._normalization_stats: ArcoEra5NormalizationStats | None = None

    def __del__(self) -> None:
        try:
            with self._dynamic_prefetch_condition:
                self._dynamic_prefetch_shutdown = True
                self._dynamic_prefetch_condition.notify_all()
            thread = self._dynamic_prefetch_thread
            if thread is not None and thread.is_alive():
                thread.join(timeout=0.1)
        except Exception:
            pass

    def _open_dataset(self) -> xr.Dataset:
        if self._dataset is None:
            self._dataset = open_arco_era5_dataset(
                self.config.dataset_url,
                gcs_token=self.config.gcs_token,
            )
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
        return prepare_arco_spatial_dataarray(
            data_array,
            latitude_descending=self.config.latitude_descending,
        )

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

    def _dynamic_time_span_steps(self) -> int:
        dynamic_offsets = [self._step_count(hours) for hours in self.config.input_time_offsets_hours]
        dynamic_offsets.extend(
            self._step_count(self.config.lead_time_hours * step)
            for step in range(1, self.config.forecast_steps + 1)
        )
        return max(1, max(dynamic_offsets) - min(dynamic_offsets) + 1)

    def _resolved_dynamic_chunk_time_steps(self) -> int:
        if self._dynamic_chunk_time_steps is None:
            requested = max(
                self._dynamic_time_span_steps(),
                int(self.config.dynamic_ram_cache_time_steps),
            )
            self._dynamic_chunk_time_steps = min(requested, _MAX_DYNAMIC_RAM_CACHE_TIME_STEPS)
            if requested > self._dynamic_chunk_time_steps and not self._dynamic_chunk_size_warning_emitted:
                warnings.warn(
                    "Requested dynamic_ram_cache_time_steps="
                    f"{requested}, but the loader caps the active RAM window at "
                    f"{self._dynamic_chunk_time_steps} time steps to avoid multi-GB stalls. "
                    "Use the rolling cache rather than a huge monolithic window.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._dynamic_chunk_size_warning_emitted = True
        return self._dynamic_chunk_time_steps

    def _resolved_dynamic_prefetch_block_time_steps(self) -> int:
        return min(
            self._resolved_dynamic_chunk_time_steps(),
            max(
                self._dynamic_time_span_steps(),
                int(self.config.dynamic_prefetch_block_time_steps),
            ),
        )

    def _dynamic_channel_transform_kinds(self) -> tuple[str, ...]:
        kinds: list[str] = []
        for variable_name in self.config.upper_air_variables:
            kind = _DYNAMIC_ZSCORE_KIND
            kinds.extend([kind] * len(self.config.pressure_levels))
        for variable_name in self.config.surface_variables:
            kind = _DYNAMIC_LOG1P_MM_ZSCORE_KIND if variable_name == "total_precipitation" else _DYNAMIC_ZSCORE_KIND
            kinds.append(kind)
        return tuple(kinds)

    def _static_channel_transform_kinds(self) -> tuple[str, ...]:
        kinds: list[str] = []
        for variable_name in self.config.static_variables:
            if variable_name in {"land_sea_mask", "cos_latitude", "cos_longitude", "sin_longitude"}:
                kinds.append(_IDENTITY_KIND)
            else:
                kinds.append(_DYNAMIC_ZSCORE_KIND)
        return tuple(kinds)

    def _normalization_dataset_signature(self) -> str:
        return describe_arco_era5_dataset_location(self.config.dataset_url)

    def _normalization_stats_match_config(self, stats: ArcoEra5NormalizationStats) -> bool:
        return (
            stats.version == _NORMALIZATION_STATS_VERSION
            and stats.dataset_url == self._normalization_dataset_signature()
            and stats.dynamic_channel_names == tuple(self.config.channel_names)
            and stats.dynamic_transform_kinds == self._dynamic_channel_transform_kinds()
            and stats.static_channel_names == tuple(self.config.static_variables)
            and stats.static_transform_kinds == self._static_channel_transform_kinds()
        )

    def _ensure_valid_normalization_stats(self, stats: ArcoEra5NormalizationStats) -> ArcoEra5NormalizationStats:
        if not self._normalization_stats_match_config(stats):
            raise ValueError("Normalization stats do not match the current dataset configuration.")

        if any(value <= 0.0 for value in (*stats.dynamic_std, *stats.static_std)):
            raise ValueError("Normalization stats must contain strictly positive standard deviations.")
        return stats

    @staticmethod
    def _apply_pre_standardization_transform(values: np.ndarray, kind: str) -> np.ndarray:
        if kind in {_DYNAMIC_ZSCORE_KIND, _IDENTITY_KIND}:
            return values
        if kind == _DYNAMIC_LOG1P_MM_ZSCORE_KIND:
            np.maximum(values, 0.0, out=values)
            values *= 1000.0
            np.log1p(values, out=values)
            return values
        raise ValueError(f"Unsupported normalization transform kind: {kind}")

    @staticmethod
    def _invert_pre_standardization_transform(values: np.ndarray, kind: str) -> np.ndarray:
        if kind in {_DYNAMIC_ZSCORE_KIND, _IDENTITY_KIND}:
            return values
        if kind == _DYNAMIC_LOG1P_MM_ZSCORE_KIND:
            np.expm1(values, out=values)
            values /= 1000.0
            return values
        raise ValueError(f"Unsupported normalization transform kind: {kind}")

    def _apply_dynamic_pre_standardization_transforms(self, dynamic: np.ndarray) -> np.ndarray:
        transformed = dynamic.astype(np.float32, copy=True)
        for channel_index, kind in enumerate(self._dynamic_channel_transform_kinds()):
            transformed[:, channel_index] = self._apply_pre_standardization_transform(
                transformed[:, channel_index],
                kind,
            )
        return transformed

    def _apply_static_pre_standardization_transforms(self, static_stack: np.ndarray) -> np.ndarray:
        transformed = static_stack.astype(np.float32, copy=True)
        for channel_index, kind in enumerate(self._static_channel_transform_kinds()):
            transformed[channel_index] = self._apply_pre_standardization_transform(
                transformed[channel_index],
                kind,
            )
        return transformed

    def _build_raw_static_stack(self) -> np.ndarray:
        ds = self._open_dataset()
        land_sea_mask = prepare_arco_spatial_dataarray(
            ds["land_sea_mask"],
            latitude_descending=self.config.latitude_descending,
        ).load().values
        orography = prepare_arco_spatial_dataarray(
            ds[self.config.orography_source],
            latitude_descending=self.config.latitude_descending,
        ).load().values
        if self.config.convert_geopotential_to_height and self.config.orography_source == "geopotential_at_surface":
            orography = orography / _STANDARD_GRAVITY

        latitude = prepare_arco_spatial_dataarray(
            ds["latitude"],
            latitude_descending=self.config.latitude_descending,
        ).values
        longitude = prepare_arco_spatial_dataarray(
            ds["longitude"],
            latitude_descending=self.config.latitude_descending,
        ).values
        derived_maps = build_fuxi_derived_static_maps(latitude, longitude)
        static_map = {
            "land_sea_mask": land_sea_mask.astype(np.float32),
            "orography": orography.astype(np.float32),
            **derived_maps,
        }
        return np.stack(
            [static_map[name] for name in self.config.static_variables],
            axis=0,
        ).astype(np.float32, copy=False)

    def _fit_normalization_stats(self) -> ArcoEra5NormalizationStats:
        anchor_indices = self._build_valid_anchor_indices()
        if anchor_indices.size == 0:
            raise ValueError("Cannot fit normalization stats on an empty dataset split.")

        sample_count = min(int(self.config.normalization_fit_sample_count), int(anchor_indices.shape[0]))
        sample_positions = np.linspace(0, int(anchor_indices.shape[0]) - 1, num=sample_count, dtype=np.int64)
        sampled_anchor_indices = anchor_indices[np.unique(sample_positions)]

        dataset = self._open_dataset()
        plan = self._build_dynamic_download_plan()
        dynamic_sum = np.zeros(len(self.config.channel_names), dtype=np.float64)
        dynamic_sumsq = np.zeros(len(self.config.channel_names), dtype=np.float64)
        dynamic_count = 0

        for anchor_index in sampled_anchor_indices.tolist():
            input_indices = [anchor_index + self._step_count(hours) for hours in self.config.input_time_offsets_hours]
            target_indices = [
                anchor_index + self._step_count(self.config.lead_time_hours * step)
                for step in range(1, self.config.forecast_steps + 1)
            ]
            requested_indices = sorted({*input_indices, *target_indices})
            chunk_start = min(requested_indices)
            chunk_stop = max(requested_indices) + 1
            raw_chunk = self._materialize_dynamic_chunk(dataset, plan, chunk_start, chunk_stop)
            selected_chunk = raw_chunk[[index - chunk_start for index in requested_indices]]
            transformed_chunk = self._apply_dynamic_pre_standardization_transforms(selected_chunk)
            transformed_chunk64 = transformed_chunk.astype(np.float64, copy=False)
            dynamic_sum += transformed_chunk64.sum(axis=(0, 2, 3))
            dynamic_sumsq += np.square(transformed_chunk64).sum(axis=(0, 2, 3))
            dynamic_count += (
                int(transformed_chunk.shape[0]) * int(transformed_chunk.shape[2]) * int(transformed_chunk.shape[3])
            )

        if dynamic_count <= 0:
            raise ValueError("Cannot fit normalization stats without any sampled dynamic values.")

        dynamic_mean = dynamic_sum / float(dynamic_count)
        dynamic_var = np.maximum(dynamic_sumsq / float(dynamic_count) - np.square(dynamic_mean), 0.0)
        dynamic_std = np.sqrt(dynamic_var)
        dynamic_std = np.maximum(dynamic_std, _MIN_NORMALIZATION_STD)

        static_stack = self._build_raw_static_stack()
        transformed_static = self._apply_static_pre_standardization_transforms(static_stack)
        static_mean = transformed_static.reshape(transformed_static.shape[0], -1).mean(axis=1, dtype=np.float64)
        static_var = transformed_static.reshape(transformed_static.shape[0], -1).var(axis=1, dtype=np.float64)
        static_std = np.sqrt(np.maximum(static_var, 0.0))
        static_std = np.maximum(static_std, _MIN_NORMALIZATION_STD)

        for channel_index, kind in enumerate(self._static_channel_transform_kinds()):
            if kind == _IDENTITY_KIND:
                static_mean[channel_index] = 0.0
                static_std[channel_index] = 1.0

        return self._ensure_valid_normalization_stats(
            ArcoEra5NormalizationStats(
                version=_NORMALIZATION_STATS_VERSION,
                dataset_url=self._normalization_dataset_signature(),
                dynamic_channel_names=tuple(self.config.channel_names),
                dynamic_transform_kinds=self._dynamic_channel_transform_kinds(),
                dynamic_mean=tuple(float(value) for value in dynamic_mean.tolist()),
                dynamic_std=tuple(float(value) for value in dynamic_std.tolist()),
                static_channel_names=tuple(self.config.static_variables),
                static_transform_kinds=self._static_channel_transform_kinds(),
                static_mean=tuple(float(value) for value in static_mean.tolist()),
                static_std=tuple(float(value) for value in static_std.tolist()),
                fit_sample_count=int(sampled_anchor_indices.shape[0]),
            )
        )

    def _load_or_fit_normalization_stats(self) -> ArcoEra5NormalizationStats:
        if self._normalization_stats is not None:
            return self._normalization_stats

        if not self.config.apply_normalization:
            raise RuntimeError("Normalization stats were requested while apply_normalization is false.")

        stats_path = self.config.normalization_stats_path
        if (
            stats_path is not None
            and stats_path.is_file()
            and not self.config.normalization_force_recompute
        ):
            try:
                loaded = ArcoEra5NormalizationStats.from_dict(_read_json_file(stats_path))
                self._normalization_stats = self._ensure_valid_normalization_stats(loaded)
                return self._normalization_stats
            except Exception:
                pass

        fitted = self._fit_normalization_stats()
        if stats_path is not None:
            _write_json_file(stats_path, fitted.to_dict())
        self._normalization_stats = fitted
        return self._normalization_stats

    def ensure_normalization_stats(self) -> ArcoEra5NormalizationStats | None:
        if not self.config.apply_normalization:
            return None
        return self._load_or_fit_normalization_stats()

    def _normalize_dynamic_chunk(self, dynamic: np.ndarray) -> np.ndarray:
        if not self.config.apply_normalization:
            return dynamic.astype(np.float32, copy=False)

        stats = self._load_or_fit_normalization_stats()
        transformed = self._apply_dynamic_pre_standardization_transforms(dynamic)
        standardize_mask = np.asarray(
            [kind != _IDENTITY_KIND for kind in stats.dynamic_transform_kinds],
            dtype=bool,
        )
        if standardize_mask.any():
            mean = np.asarray(stats.dynamic_mean, dtype=np.float32)[standardize_mask]
            std = np.asarray(stats.dynamic_std, dtype=np.float32)[standardize_mask]
            transformed[:, standardize_mask] = (
                transformed[:, standardize_mask] - mean[None, :, None, None]
            ) / std[None, :, None, None]
        return transformed.astype(np.float32, copy=False)

    def _normalize_static_stack(self, static_stack: np.ndarray) -> np.ndarray:
        if not self.config.apply_normalization:
            return static_stack.astype(np.float32, copy=False)

        stats = self._load_or_fit_normalization_stats()
        transformed = self._apply_static_pre_standardization_transforms(static_stack)
        standardize_mask = np.asarray(
            [kind != _IDENTITY_KIND for kind in stats.static_transform_kinds],
            dtype=bool,
        )
        if standardize_mask.any():
            mean = np.asarray(stats.static_mean, dtype=np.float32)[standardize_mask]
            std = np.asarray(stats.static_std, dtype=np.float32)[standardize_mask]
            transformed[standardize_mask] = (
                transformed[standardize_mask] - mean[:, None, None]
            ) / std[:, None, None]
        return transformed.astype(np.float32, copy=False)

    def denormalize_dynamic_tensor(self, dynamic: Tensor) -> Tensor:
        if not self.config.apply_normalization:
            return dynamic

        stats = self._load_or_fit_normalization_stats()
        restored = dynamic.float().clone()
        if restored.ndim == 4:
            restored = restored.unsqueeze(0)
            squeeze_batch = True
        elif restored.ndim == 5:
            squeeze_batch = False
        else:
            raise ValueError(
                f"Expected normalized dynamic tensor shaped [T, C, H, W] or [B, T, C, H, W], got {tuple(dynamic.shape)}"
            )

        for channel_index, kind in enumerate(stats.dynamic_transform_kinds):
            if kind != _IDENTITY_KIND:
                restored[:, :, channel_index] = (
                    restored[:, :, channel_index] * float(stats.dynamic_std[channel_index])
                    + float(stats.dynamic_mean[channel_index])
                )
            if kind == _DYNAMIC_LOG1P_MM_ZSCORE_KIND:
                restored[:, :, channel_index] = torch.expm1(restored[:, :, channel_index]) / 1000.0

        if squeeze_batch:
            restored = restored.squeeze(0)
        return restored

    def _build_dynamic_download_plan(self) -> ArcoEra5DownloadPlan:
        if self._dynamic_download_plan is None:
            dataset = self._open_dataset()
            self._dynamic_download_plan = build_arco_era5_download_plan(
                tuple(dataset.data_vars),
                self.config,
                include_static_sources=False,
            )
        return self._dynamic_download_plan

    def _materialize_dynamic_chunk(
        self,
        dataset: xr.Dataset,
        plan: ArcoEra5DownloadPlan,
        start_index: int,
        stop_index: int,
    ) -> np.ndarray:
        level_selection = list(self.config.pressure_levels)

        source_arrays: dict[str, xr.DataArray] = {}
        for variable_name in plan.source_pressure_variables:
            source_arrays[variable_name] = dataset[variable_name].isel(time=slice(start_index, stop_index)).sel(
                level=level_selection
            )
        for variable_name in plan.source_surface_variables:
            source_arrays[variable_name] = dataset[variable_name].isel(time=slice(start_index, stop_index))

        source_chunk = xr.Dataset(source_arrays).load()

        relative_humidity: xr.DataArray | None = None
        if plan.derive_relative_humidity:
            relative_humidity = specific_humidity_to_relative_humidity(
                prepare_arco_spatial_dataarray(
                    source_chunk["specific_humidity"],
                    latitude_descending=self.config.latitude_descending,
                ),
                prepare_arco_spatial_dataarray(
                    source_chunk["temperature"],
                    latitude_descending=self.config.latitude_descending,
                ),
                source_chunk["temperature"]["level"],
            )

        upper_air_groups: list[np.ndarray] = []
        for variable_name in self.config.upper_air_variables:
            if variable_name == "relative_humidity":
                if relative_humidity is not None:
                    array = relative_humidity
                else:
                    array = prepare_arco_spatial_dataarray(
                        source_chunk["relative_humidity"],
                        latitude_descending=self.config.latitude_descending,
                    )
            else:
                array = prepare_arco_spatial_dataarray(
                    source_chunk[variable_name],
                    latitude_descending=self.config.latitude_descending,
                )
            upper_air_groups.append(array.values)

        upper_air = np.concatenate(upper_air_groups, axis=1)

        surface_groups: list[np.ndarray] = []
        for variable_name in self.config.surface_variables:
            array = prepare_arco_spatial_dataarray(
                source_chunk[variable_name],
                latitude_descending=self.config.latitude_descending,
            )
            surface_groups.append(array.values[:, None, :, :])

        surface = np.concatenate(surface_groups, axis=1)
        return np.concatenate([upper_air, surface], axis=1).astype(np.float32, copy=False)

    def _raise_dynamic_prefetch_error_locked(self) -> None:
        if self._dynamic_prefetch_error is not None:
            raise RuntimeError("Dynamic RAM prefetch thread failed.") from self._dynamic_prefetch_error

    def _ensure_dynamic_prefetch_thread_locked(self) -> None:
        if self._dynamic_prefetch_thread is not None:
            return
        self._dynamic_prefetch_thread = threading.Thread(
            target=self._dynamic_prefetch_loop,
            name="era5-dynamic-prefetch",
            daemon=True,
        )
        self._dynamic_prefetch_thread.start()

    def _ensure_dynamic_ring_storage_locked(self, block: np.ndarray) -> None:
        capacity = self._resolved_dynamic_chunk_time_steps()
        expected_shape = (capacity, *block.shape[1:])
        if self._dynamic_ring_array is None or self._dynamic_ring_array.shape != expected_shape:
            self._dynamic_ring_array = np.empty(expected_shape, dtype=np.float32)

    def _write_dynamic_block_to_ring_locked(
        self,
        start_index: int,
        block: np.ndarray,
    ) -> None:
        if self._dynamic_ring_array is None:
            raise RuntimeError("Dynamic ring buffer was not allocated before writing.")

        capacity = self._resolved_dynamic_chunk_time_steps()
        block_size = int(block.shape[0])
        slot = start_index % capacity
        first_span = min(capacity - slot, block_size)
        self._dynamic_ring_array[slot : slot + first_span] = block[:first_span]
        if first_span < block_size:
            self._dynamic_ring_array[: block_size - first_span] = block[first_span:]

        if self._dynamic_ring_start is None:
            self._dynamic_ring_start = start_index
        self._dynamic_ring_stop = start_index + block_size

    def _dynamic_prefetch_loop(self) -> None:
        try:
            dataset = open_arco_era5_dataset(
                self.config.dataset_url,
                gcs_token=self.config.gcs_token,
            )
            plan = build_arco_era5_download_plan(
                tuple(dataset.data_vars),
                self.config,
                include_static_sources=False,
            )
            while True:
                with self._dynamic_prefetch_condition:
                    while True:
                        if self._dynamic_prefetch_shutdown:
                            return

                        self._raise_dynamic_prefetch_error_locked()

                        start_index = self._dynamic_ring_start
                        stop_index = self._dynamic_ring_stop
                        target_stop = self._dynamic_prefetch_target_stop
                        if start_index is None or stop_index is None or target_stop is None:
                            self._dynamic_prefetch_condition.wait()
                            continue

                        capacity = self._resolved_dynamic_chunk_time_steps()
                        loaded_time_steps = stop_index - start_index
                        free_time_steps = capacity - loaded_time_steps
                        remaining_time_steps = target_stop - stop_index
                        if free_time_steps <= 0 or remaining_time_steps <= 0:
                            self._dynamic_prefetch_condition.wait()
                            continue

                        load_start = stop_index
                        load_stop = load_start + min(
                            free_time_steps,
                            remaining_time_steps,
                            self._resolved_dynamic_prefetch_block_time_steps(),
                        )
                        generation = self._dynamic_prefetch_generation
                        break

                block = self._normalize_dynamic_chunk(
                    self._materialize_dynamic_chunk(dataset, plan, load_start, load_stop)
                )

                with self._dynamic_prefetch_condition:
                    if self._dynamic_prefetch_shutdown:
                        return
                    if generation != self._dynamic_prefetch_generation:
                        continue
                    if self._dynamic_ring_stop != load_start:
                        continue

                    self._ensure_dynamic_ring_storage_locked(block)
                    self._write_dynamic_block_to_ring_locked(load_start, block)
                    self._dynamic_prefetch_condition.notify_all()
        except BaseException as exc:  # pragma: no cover - exercised through waiting consumers
            with self._dynamic_prefetch_condition:
                self._dynamic_prefetch_error = exc
                self._dynamic_prefetch_condition.notify_all()

    def _ensure_dynamic_chunk(self, time_indices: Sequence[int]) -> None:
        min_index = min(int(time_index) for time_index in time_indices)
        max_index = max(int(time_index) for time_index in time_indices)
        request_stop = max_index + 1
        capacity = self._resolved_dynamic_chunk_time_steps()
        total_time_steps = len(self._load_time_values())
        target_stop = min(
            total_time_steps,
            max(request_stop, min_index + capacity),
        )

        with self._dynamic_prefetch_condition:
            self._raise_dynamic_prefetch_error_locked()

            needs_reset = (
                self._dynamic_ring_start is None
                or self._dynamic_ring_stop is None
                or min_index < self._dynamic_ring_start
                or request_stop > self._dynamic_ring_start + capacity
            )
            if needs_reset:
                self._dynamic_ring_start = min_index
                self._dynamic_ring_stop = min_index
                self._dynamic_prefetch_generation += 1
            elif min_index > self._dynamic_ring_start:
                self._dynamic_ring_start = min_index

            self._dynamic_prefetch_target_stop = target_stop
            self._ensure_dynamic_prefetch_thread_locked()
            self._dynamic_prefetch_condition.notify_all()

            while True:
                self._raise_dynamic_prefetch_error_locked()
                if (
                    self._dynamic_ring_start is not None
                    and self._dynamic_ring_stop is not None
                    and min_index >= self._dynamic_ring_start
                    and request_stop <= self._dynamic_ring_stop
                ):
                    return
                self._dynamic_prefetch_condition.wait()

    def _read_dynamic_tensor_from_ring(self, time_indices: Sequence[int]) -> np.ndarray:
        with self._dynamic_prefetch_condition:
            if self._dynamic_ring_array is None or self._dynamic_ring_start is None:
                raise RuntimeError("Dynamic RAM cache was not initialized before sample assembly.")

            positions = np.asarray(time_indices, dtype=np.int64) % self._resolved_dynamic_chunk_time_steps()
            return self._dynamic_ring_array[positions].astype(np.float32, copy=False)

    def _build_dynamic_tensor(self, time_indices: Sequence[int]) -> np.ndarray:
        self._ensure_dynamic_chunk(time_indices)
        return self._read_dynamic_tensor_from_ring(time_indices)

    def _build_static_features(self) -> torch.Tensor:
        if self._static_features is not None:
            return self._static_features

        stacked = self._normalize_static_stack(self._build_raw_static_stack())
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

    def _build_sample(self, anchor_index: int) -> dict[str, Any]:
        from ..models.fuxi_short import build_fuxi_time_embeddings

        time_values = self._load_time_values()
        input_indices = [anchor_index + self._step_count(hours) for hours in self.config.input_time_offsets_hours]
        target_indices = [
            anchor_index + self._step_count(self.config.lead_time_hours * step)
            for step in range(1, self.config.forecast_steps + 1)
        ]

        anchor_time = pd.Timestamp(time_values[anchor_index])
        self._ensure_dynamic_chunk((*input_indices, *target_indices))
        sample = {
            "x": torch.from_numpy(self._read_dynamic_tensor_from_ring(input_indices)),
            "target": torch.from_numpy(self._read_dynamic_tensor_from_ring(target_indices)),
            "static_features": self._build_static_features(),
            "temb": torch.from_numpy(
                build_fuxi_time_embeddings(anchor_time, total_steps=1, freq_hours=self.config.lead_time_hours)[0, 0]
            ),
        }
        if self.config.include_sample_metadata:
            sample.update(
                {
                    "input_times": [str(pd.Timestamp(time_values[i])) for i in input_indices],
                    "anchor_time": str(anchor_time),
                    "target_times": [str(pd.Timestamp(time_values[i])) for i in target_indices],
                }
            )
        return sample

    def __len__(self) -> int:
        return int(self._build_valid_anchor_indices().shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        anchor_index = int(self._build_valid_anchor_indices()[index])
        return self._build_sample(anchor_index)

    def source_summary(self) -> dict[str, Any]:
        from ..models.fuxi_lower_res import FuXiLowerResConfig

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
            "include_sample_metadata": self.config.include_sample_metadata,
            "dynamic_ram_cache_time_steps": self.config.dynamic_ram_cache_time_steps,
            "dynamic_prefetch_block_time_steps": self.config.dynamic_prefetch_block_time_steps,
            "apply_normalization": self.config.apply_normalization,
            "normalization_stats_path": (
                None if self.config.normalization_stats_path is None else str(self.config.normalization_stats_path)
            ),
            "normalization_force_recompute": self.config.normalization_force_recompute,
            "normalization_fit_sample_count": self.config.normalization_fit_sample_count,
            "model_input_size": list(model_config.input_size),
            "model_time_steps": model_config.time_steps,
            "model_dynamic_channels": model_config.in_chans,
            "model_static_channels": model_config.aux_chans,
            "model_forecast_steps": model_config.forecast_steps,
        }


class ContiguousDistributedSampler(Sampler[int]):
    """Distributed sampler that keeps each rank on a contiguous region of the time axis."""

    def __init__(
        self,
        dataset: Dataset[Any],
        *,
        num_replicas: int,
        rank: int,
        drop_last: bool = False,
    ) -> None:
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")

        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.drop_last = bool(drop_last)
        dataset_length = len(dataset)
        if self.drop_last:
            self.num_samples = dataset_length // self.num_replicas
        else:
            self.num_samples = int(math.ceil(dataset_length / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.epoch = 0

    def __iter__(self):
        dataset_length = len(self.dataset)
        if dataset_length == 0 or self.num_samples == 0:
            return iter(())

        indices = list(range(dataset_length))
        if self.drop_last:
            indices = indices[: self.total_size]
        else:
            indices.extend([indices[-1]] * (self.total_size - dataset_length))

        start_index = self.rank * self.num_samples
        end_index = start_index + self.num_samples
        return iter(indices[start_index:end_index])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def build_arco_era5_dataloader(
    dataset: ArcoEra5FuXiDataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
    pin_memory: bool = False,
    sampler: Sampler[Any] | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = 2,
) -> DataLoader[dict[str, Any]]:
    resolved_num_workers = num_workers
    ram_cached_sequential = (
        not shuffle
        and getattr(getattr(dataset, "config", None), "dynamic_ram_cache_time_steps", 0) > 0
    )
    # When sequential loading is backed by a large RAM cache, in-process fetches avoid copying
    # very large batches from worker subprocesses back to the training rank.
    if ram_cached_sequential:
        resolved_num_workers = 0
    # Without the RAM cache, overlapping sequential windows still should not be split across many
    # workers because that breaks temporal locality and defeats overlap reuse.
    elif not shuffle and num_workers > 1:
        resolved_num_workers = 1

    loader_kwargs: dict[str, Any] = {}
    if resolved_num_workers > 0:
        loader_kwargs["persistent_workers"] = (
            bool(persistent_workers) if persistent_workers is not None else True
        )
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        num_workers=resolved_num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        sampler=sampler,
        **loader_kwargs,
    )


__all__ = [
    "ArcoEra5CompatibilityReport",
    "ArcoEra5DownloadPlan",
    "ArcoEra5DownloadWindow",
    "ContiguousDistributedSampler",
    "ArcoEra5FuXiDataConfig",
    "ArcoEra5FuXiDataset",
    "ArcoEra5NormalizationStats",
    "DEFAULT_ARCO_ERA5_URL",
    "FUXI_PRESSURE_LEVELS",
    "FUXI_STATIC_SOURCE_VARIABLES",
    "FUXI_STATIC_VARIABLES",
    "FUXI_SURFACE_VARIABLES",
    "FUXI_UPPER_AIR_VARIABLES",
    "arco_metadata_url",
    "build_arco_era5_download_plan",
    "build_arco_era5_dataloader",
    "build_fuxi_derived_static_maps",
    "build_fuxi_static_maps",
    "build_fuxi_channel_names",
    "download_arco_era5_subset",
    "fetch_arco_zarr_metadata",
    "inspect_arco_era5_dataset",
    "inspect_local_zarr_time_axes",
    "list_arco_dataset_variables",
    "load_arco_static_source_maps",
    "open_arco_era5_dataset",
    "prepare_arco_spatial_dataarray",
    "repair_local_zarr_time_consistency",
    "resolve_arco_era5_download_window",
    "specific_humidity_to_relative_humidity",
]
