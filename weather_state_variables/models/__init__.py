"""Model definitions used in this repository."""

from __future__ import annotations

from importlib import import_module

from ..config import DEFAULT_MODEL_CONFIG_PATH

_INTRINSIC_MODULE = f"{__name__}.fuxi_intrinsic"
_LOWER_RES_MODULE = f"{__name__}.fuxi_lower_res"
_SHORT_MODULE = f"{__name__}.fuxi_short"

_LAZY_IMPORTS = {
    "DEFAULT_MODEL_PATH": (_SHORT_MODULE, "DEFAULT_MODEL_PATH"),
    "FuXiEncoderOutput": (_LOWER_RES_MODULE, "FuXiEncoderOutput"),
    "FuXiIntrinsic": (_INTRINSIC_MODULE, "FuXiIntrinsic"),
    "FuXiIntrinsicConfig": (_INTRINSIC_MODULE, "FuXiIntrinsicConfig"),
    "FuXiLowerRes": (_LOWER_RES_MODULE, "FuXiLowerRes"),
    "FuXiLowerResConfig": (_LOWER_RES_MODULE, "FuXiLowerResConfig"),
    "FuXiLowerResDecoder": (_LOWER_RES_MODULE, "FuXiLowerResDecoder"),
    "FuXiLowerResEncoder": (_LOWER_RES_MODULE, "FuXiLowerResEncoder"),
    "FuXiShort": (_SHORT_MODULE, "FuXiShort"),
    "FuXiShortConfig": (_SHORT_MODULE, "FuXiShortConfig"),
    "build_exact_fuxi_short_graph": (_SHORT_MODULE, "build_exact_fuxi_short_graph"),
    "build_fuxi_time_embeddings": (_SHORT_MODULE, "build_fuxi_time_embeddings"),
    "inspect_exact_fuxi_short_graph": (_SHORT_MODULE, "inspect_exact_fuxi_short_graph"),
    "summarize_short_onnx_architecture": (_SHORT_MODULE, "summarize_short_onnx_architecture"),
}

__all__ = ["DEFAULT_MODEL_CONFIG_PATH", *_LAZY_IMPORTS]


def __getattr__(name: str):
    try:
        module_name, attribute_name = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
