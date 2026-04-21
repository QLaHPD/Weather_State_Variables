"""Local modeling package for weather experiments."""

from __future__ import annotations

from importlib import import_module


_DATA_MODULE = f"{__name__}.data"
_MODELS_MODULE = f"{__name__}.models"
_TRAINING_MODULE = f"{__name__}.training"
_CONFIG_MODULE = f"{__name__}.config"

_LAZY_IMPORTS = {
    "DEFAULT_MODEL_CONFIG_PATH": (_CONFIG_MODULE, "DEFAULT_MODEL_CONFIG_PATH"),
    "DEFAULT_ARCO_ERA5_URL": (_DATA_MODULE, "DEFAULT_ARCO_ERA5_URL"),
    "DEFAULT_MODEL_PATH": (_MODELS_MODULE, "DEFAULT_MODEL_PATH"),
    "ArcoEra5CompatibilityReport": (_DATA_MODULE, "ArcoEra5CompatibilityReport"),
    "ArcoEra5DownloadPlan": (_DATA_MODULE, "ArcoEra5DownloadPlan"),
    "ArcoEra5DownloadWindow": (_DATA_MODULE, "ArcoEra5DownloadWindow"),
    "ArcoEra5FuXiDataConfig": (_DATA_MODULE, "ArcoEra5FuXiDataConfig"),
    "ArcoEra5FuXiDataset": (_DATA_MODULE, "ArcoEra5FuXiDataset"),
    "ArcoEra5NormalizationStats": (_DATA_MODULE, "ArcoEra5NormalizationStats"),
    "FUXI_STATIC_SOURCE_VARIABLES": (_DATA_MODULE, "FUXI_STATIC_SOURCE_VARIABLES"),
    "FuXiEncoderOutput": (_MODELS_MODULE, "FuXiEncoderOutput"),
    "FuXiBottleneckCompressor": (_MODELS_MODULE, "FuXiBottleneckCompressor"),
    "FuXiBottleneckCompressorConfig": (_MODELS_MODULE, "FuXiBottleneckCompressorConfig"),
    "FuXiIntrinsic": (_MODELS_MODULE, "FuXiIntrinsic"),
    "FuXiIntrinsicConfig": (_MODELS_MODULE, "FuXiIntrinsicConfig"),
    "FuXiLowerRes": (_MODELS_MODULE, "FuXiLowerRes"),
    "FuXiLowerResConfig": (_MODELS_MODULE, "FuXiLowerResConfig"),
    "FuXiLowerResDecoder": (_MODELS_MODULE, "FuXiLowerResDecoder"),
    "FuXiLowerResEncoder": (_MODELS_MODULE, "FuXiLowerResEncoder"),
    "FuXiShort": (_MODELS_MODULE, "FuXiShort"),
    "FuXiShortConfig": (_MODELS_MODULE, "FuXiShortConfig"),
    "build_exact_fuxi_short_graph": (_MODELS_MODULE, "build_exact_fuxi_short_graph"),
    "build_arco_era5_dataloader": (_DATA_MODULE, "build_arco_era5_dataloader"),
    "build_arco_era5_download_plan": (_DATA_MODULE, "build_arco_era5_download_plan"),
    "build_fuxi_derived_static_maps": (_DATA_MODULE, "build_fuxi_derived_static_maps"),
    "build_fuxi_static_maps": (_DATA_MODULE, "build_fuxi_static_maps"),
    "build_fuxi_time_embeddings": (_MODELS_MODULE, "build_fuxi_time_embeddings"),
    "download_arco_era5_subset": (_DATA_MODULE, "download_arco_era5_subset"),
    "inspect_arco_era5_dataset": (_DATA_MODULE, "inspect_arco_era5_dataset"),
    "inspect_local_zarr_time_axes": (_DATA_MODULE, "inspect_local_zarr_time_axes"),
    "inspect_exact_fuxi_short_graph": (_MODELS_MODULE, "inspect_exact_fuxi_short_graph"),
    "load_arco_static_source_maps": (_DATA_MODULE, "load_arco_static_source_maps"),
    "open_arco_era5_dataset": (_DATA_MODULE, "open_arco_era5_dataset"),
    "prepare_arco_spatial_dataarray": (_DATA_MODULE, "prepare_arco_spatial_dataarray"),
    "repair_local_zarr_time_consistency": (_DATA_MODULE, "repair_local_zarr_time_consistency"),
    "resolve_arco_era5_download_window": (_DATA_MODULE, "resolve_arco_era5_download_window"),
    "summarize_short_onnx_architecture": (_MODELS_MODULE, "summarize_short_onnx_architecture"),
    "BottleneckCompressorTrainingConfig": (_TRAINING_MODULE, "BottleneckCompressorTrainingConfig"),
    "IntrinsicTrainingConfig": (_TRAINING_MODULE, "IntrinsicTrainingConfig"),
    "MainTrainingConfig": (_TRAINING_MODULE, "MainTrainingConfig"),
    "run_bottleneck_compressor_smoke_test": (_TRAINING_MODULE, "run_bottleneck_compressor_smoke_test"),
    "run_intrinsic_model_smoke_test": (_TRAINING_MODULE, "run_intrinsic_model_smoke_test"),
    "run_main_model_smoke_test": (_TRAINING_MODULE, "run_main_model_smoke_test"),
    "train_bottleneck_compressor_model": (_TRAINING_MODULE, "train_bottleneck_compressor_model"),
    "train_intrinsic_model": (_TRAINING_MODULE, "train_intrinsic_model"),
    "train_main_model": (_TRAINING_MODULE, "train_main_model"),
    "validate_main_model": (_TRAINING_MODULE, "validate_main_model"),
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute_name = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
