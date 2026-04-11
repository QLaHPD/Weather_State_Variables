from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterator

import numpy as np
import onnx
import pandas as pd
import torch
from onnx import TensorProto, external_data_helper, numpy_helper
from onnx2torch import convert
from onnx2torch.node_converters.clip import OnnxClip
from onnx2torch.node_converters.conv import _CONV_CLASS_FROM_SPATIAL_RANK
from onnx2torch.node_converters.gemm import OnnxGemm
from onnx2torch.node_converters.instance_norm import (
    OnnxInstanceNorm,
    _IN_CLASS_FROM_SPATIAL_RANK,
)
from onnx2torch.node_converters.layer_norm import (
    AXIS_DEFAULT_VALUE,
    EPSILON_DEFAULT_VALUE,
    OnnxLayerNorm,
)
from onnx2torch.node_converters.registry import _CONVERTER_REGISTRY, OperationDescription
from onnx2torch.onnx_tensor import OnnxTensor
from onnx2torch.utils.common import (
    OnnxMapping,
    OperationConverterResult,
    get_const_value,
    get_shape_from_value_info,
    onnx_mapping_from_node,
)
from onnx2torch.utils.padding import onnx_auto_pad_to_torch_padding
from torch import Tensor, nn
from torch.fx import GraphModule


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = REPO_ROOT / "assets" / "fuxi_teacher" / "short.onnx"

_TORCH_DTYPES = {
    TensorProto.FLOAT: torch.float32,
    TensorProto.UINT8: torch.uint8,
    TensorProto.INT8: torch.int8,
    TensorProto.UINT16: torch.uint16,
    TensorProto.INT16: torch.int16,
    TensorProto.INT32: torch.int32,
    TensorProto.INT64: torch.int64,
    TensorProto.BOOL: torch.bool,
    TensorProto.FLOAT16: torch.float16,
    TensorProto.DOUBLE: torch.float64,
    TensorProto.BFLOAT16: torch.bfloat16,
}

_LAYER_BLOCK_RE = re.compile(r"/decoder\.0/layers\.(\d+)/blocks\.(\d+)/")


def build_fuxi_time_embeddings(
    init_time: str | np.datetime64 | pd.Timestamp,
    total_steps: int,
    freq_hours: int = 6,
) -> np.ndarray:
    """Match the official FuXi time embedding construction."""

    init_time = np.array([pd.to_datetime(init_time)])
    tembs: list[np.ndarray] = []
    for i in range(total_steps):
        hours = np.array([pd.Timedelta(hours=t * freq_hours) for t in [i - 1, i, i + 1]])
        times = init_time[:, None] + hours[None]
        periods = [pd.Period(t, "h") for t in times.reshape(-1)]
        encoded = np.array(
            [(period.day_of_year / 366, period.hour / 24) for period in periods],
            dtype=np.float32,
        )
        encoded = np.concatenate([np.sin(encoded), np.cos(encoded)], axis=-1)
        tembs.append(encoded.reshape(1, -1))
    return np.stack(tembs)


def _check_model_files(model_path: Path) -> Path:
    model_path = model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    external_data_path = model_path.with_suffix("")
    if not external_data_path.is_file():
        raise FileNotFoundError(
            "FuXi short.onnx requires a sibling external-data file named "
            f"'{external_data_path.name}'. Missing: {external_data_path}"
        )
    return external_data_path


def _node_ints_attr(node: onnx.NodeProto, name: str) -> list[int]:
    for attribute in node.attribute:
        if attribute.name == name:
            return [int(value) for value in attribute.ints]
    raise KeyError(f"Attribute '{name}' not found on node '{node.name}'")


def _find_node(model: onnx.ModelProto, node_name: str) -> onnx.NodeProto:
    for node in model.graph.node:
        if node.name == node_name:
            return node
    raise KeyError(f"Node '{node_name}' not found in ONNX graph")


def summarize_short_onnx_architecture(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Extract a compact architecture recipe from the fixed short-range ONNX graph."""

    resolved_model_path = Path(model_path).resolve()
    _check_model_files(resolved_model_path)
    model = onnx.load(str(resolved_model_path), load_external_data=False)
    initializers = {
        initializer.name: [int(dim) for dim in initializer.dims]
        for initializer in model.graph.initializer
    }

    input_value = model.graph.input[0]
    input_shape = [
        dim.dim_value if dim.HasField("dim_value") else dim.dim_param
        for dim in input_value.type.tensor_type.shape.dim
    ]
    time_steps = int(input_shape[1])
    in_chans = int(input_shape[2])
    input_height = int(input_shape[3])
    input_width = int(input_shape[4])

    patch_weight_shape = initializers["decoder.0.patch_embed.proj.weight"]
    patch_node = _find_node(model, "/decoder.0/patch_embed/proj/Conv")
    patch_stride = _node_ints_attr(patch_node, "strides")
    patch_kernel = _node_ints_attr(patch_node, "kernel_shape")
    embed_dim = int(patch_weight_shape[0])
    merged_input_channels = int(patch_weight_shape[1])
    aux_chans = merged_input_channels // time_steps - in_chans

    resized_height = (input_height // patch_stride[0]) * patch_stride[0]
    resized_width = (input_width // patch_stride[1]) * patch_stride[1]
    patch_grid = [resized_height // patch_stride[0], resized_width // patch_stride[1]]
    latent_grid = [(patch_grid[0] + 1) // 2, (patch_grid[1] + 1) // 2]

    downsample_node = _find_node(model, "/decoder.0/down_blocks.0/downsample/op/Conv")
    upsample_node = _find_node(model, "/decoder.0/up_blocks.0/upsample/op/ConvTranspose")
    fpn_node = _find_node(model, "/decoder.0/fpn/fpn.0/MatMul")
    head_node = _find_node(model, "/decoder.0/head/MatMul")
    cpb_node = _find_node(model, "/decoder.0/layers.0/blocks.0/attn/cpb_mlp/cpb_mlp.0/MatMul")
    fc1_node = _find_node(model, "/decoder.0/layers.0/blocks.0/mlp/fc1/MatMul")
    fc2_node = _find_node(model, "/decoder.0/layers.0/blocks.0/mlp/fc2/MatMul")

    cpb_shape = initializers[cpb_node.input[0]]
    window_size = [
        (cpb_shape[1] + 1) // 2,
        (cpb_shape[2] + 1) // 2,
    ]
    num_heads = int(initializers["decoder.0.layers.0.blocks.0.attn.logit_scale"][0])
    mlp_fc1_shape = initializers[fc1_node.input[1]]
    mlp_fc2_shape = initializers[fc2_node.input[1]]

    block_ids: dict[int, set[int]] = {}
    for node in model.graph.node:
        match = _LAYER_BLOCK_RE.match(node.name)
        if match is None:
            continue
        layer_index = int(match.group(1))
        block_index = int(match.group(2))
        block_ids.setdefault(layer_index, set()).add(block_index)

    depths = [len(block_ids[index]) for index in sorted(block_ids)]

    return {
        "model_path": str(resolved_model_path),
        "input_shape": input_shape,
        "time_steps": time_steps,
        "in_chans": in_chans,
        "out_chans": int(initializers["decoder.0.head.bias"][0] // (patch_kernel[0] * patch_kernel[1])),
        "temb_dim": int(model.graph.input[1].type.tensor_type.shape.dim[-1].dim_value),
        "aux_chans": int(aux_chans),
        "resized_input_size": [resized_height, resized_width],
        "patch_size": patch_kernel,
        "patch_grid": patch_grid,
        "latent_grid": latent_grid,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "window_size": window_size,
        "depths": depths,
        "time_embed_dims": [
            int(initializers["decoder.0.time_embed.0.weight"][1]),
            int(initializers["decoder.0.time_embed.0.weight"][0]),
            int(initializers["decoder.0.time_embed.2.weight"][0]),
        ],
        "resblock_scale_shift_dim": int(
            initializers["decoder.0.down_blocks.0.resnets.0.emb_layers.1.weight"][0]
        ),
        "mlp_fc1_dim": int(mlp_fc1_shape[1]),
        "mlp_hidden_dim": int(mlp_fc2_shape[0]),
        "fpn_in_dim": int(initializers[fpn_node.input[1]][0]),
        "fpn_out_dim": int(initializers[fpn_node.input[1]][1]),
        "head_out_dim": int(initializers[head_node.input[1]][1]),
        "downsample": {
            "kernel_size": _node_ints_attr(downsample_node, "kernel_shape"),
            "stride": _node_ints_attr(downsample_node, "strides"),
            "padding": _node_ints_attr(downsample_node, "pads"),
        },
        "upsample": {
            "kernel_size": _node_ints_attr(upsample_node, "kernel_shape"),
            "stride": _node_ints_attr(upsample_node, "strides"),
            "padding": _node_ints_attr(upsample_node, "pads"),
        },
    }


def _to_torch_external_meta(base_dir: Path, onnx_tensor: OnnxTensor) -> torch.Tensor:
    if not external_data_helper.uses_external_data(onnx_tensor.proto):
        array = numpy_helper.to_array(onnx_tensor.proto, base_dir=str(base_dir)).copy()
        return torch.from_numpy(array)

    dims = tuple(int(dim) for dim in onnx_tensor.proto.dims)
    dtype = _TORCH_DTYPES.get(onnx_tensor.proto.data_type)
    if dtype is None:
        raise NotImplementedError(
            f"Unsupported ONNX dtype {onnx_tensor.proto.data_type} for {onnx_tensor.name}"
        )
    return torch.empty(dims, dtype=dtype, device="meta")


@contextmanager
def _patched_exact_onnx2torch_context(base_dir: Path) -> Iterator[None]:
    """Patch onnx2torch so external tensors stay on meta during conversion.

    This keeps the large FuXi weight file out of host RAM while still producing
    an exact GraphModule topology from the ONNX graph.
    """

    original_to_torch = OnnxTensor.to_torch
    original_registry = dict(_CONVERTER_REGISTRY)

    def to_torch_external_meta(self: OnnxTensor) -> torch.Tensor:
        return _to_torch_external_meta(base_dir=base_dir, onnx_tensor=self)

    def patched_conv(node, graph):
        weights = graph.initializers[node.input_values[1]].to_torch()
        bias = graph.initializers[node.input_values[2]].to_torch() if len(node.input_values) == 3 else None
        op_type = node.operation_type
        spatial_rank = len(weights.shape) - 2
        conv_class = _CONV_CLASS_FROM_SPATIAL_RANK[op_type, spatial_rank]
        node_attributes = node.attributes
        padding, input_padding_module = onnx_auto_pad_to_torch_padding(
            onnx_padding=node_attributes.get("pads", [0] * spatial_rank * 2),
            auto_pad=node_attributes.get("auto_pad", "NOTSET"),
        )
        common_kwargs = {
            "kernel_size": node_attributes.get("kernel_shape", weights.shape[2:]),
            "stride": node_attributes.get("strides", 1),
            "dilation": node_attributes.get("dilations", 1),
            "groups": node_attributes.get("group", 1),
            "padding": padding,
            "bias": bias is not None,
        }
        if op_type == "Conv":
            special_kwargs = {
                "out_channels": weights.shape[0],
                "in_channels": weights.shape[1] * common_kwargs["groups"],
            }
        else:
            special_kwargs = {
                "out_channels": weights.shape[1] * common_kwargs["groups"],
                "in_channels": weights.shape[0],
                "output_padding": node_attributes.get("output_padding", [0] * spatial_rank),
            }

        torch_module = conv_class(**common_kwargs, **special_kwargs).to(
            device=weights.device,
            dtype=weights.dtype,
        )
        with torch.no_grad():
            torch_module.weight.data = weights
            if bias is not None:
                torch_module.bias.data = bias

        if input_padding_module is not None:
            input_padding_module = input_padding_module.to(device=weights.device)
            torch_module = nn.Sequential(input_padding_module, torch_module)

        return OperationConverterResult(
            torch_module=torch_module,
            onnx_mapping=OnnxMapping(inputs=(node.input_values[0],), outputs=node.output_values),
        )

    def patched_gemm(node, graph):
        a_name = node.input_values[0]
        b_name = node.input_values[1]
        c_name = node.input_values[2] if len(node.input_values) > 2 else None
        node_attributes = node.attributes
        alpha = node_attributes.get("alpha", 1.0)
        beta = node_attributes.get("beta", 1.0)
        trans_a = node_attributes.get("transA", 0) != 0
        trans_b = node_attributes.get("transB", 0) != 0

        if not trans_a and b_name in graph.initializers and (c_name is None or c_name in graph.initializers):
            bias = graph.initializers[c_name].to_torch() if c_name is not None else None
            if bias is None or bias.dim() == 1:
                weights = graph.initializers[b_name].to_torch()
                if not trans_b:
                    weights = weights.T

                torch_module = nn.Linear(
                    weights.shape[1],
                    weights.shape[0],
                    bias=bias is not None,
                ).to(device=weights.device, dtype=weights.dtype)
                with torch.no_grad():
                    torch_module.weight.data = weights * alpha
                    if bias is not None:
                        torch_module.bias.data = bias * beta

                return OperationConverterResult(
                    torch_module=torch_module,
                    onnx_mapping=OnnxMapping(inputs=(a_name,), outputs=node.output_values),
                )

        return OperationConverterResult(
            torch_module=OnnxGemm(alpha=alpha, beta=beta, trans_a=trans_a, trans_b=trans_b),
            onnx_mapping=onnx_mapping_from_node(node),
        )

    def patched_layer_norm(node, graph):
        axis = node.attributes.get("axis", AXIS_DEFAULT_VALUE)
        epsilon = node.attributes.get("epsilon", EPSILON_DEFAULT_VALUE)
        if all(value_name in graph.initializers for value_name in node.input_values[1:]):
            input_shape = get_shape_from_value_info(graph.value_info[node.input_values[0]])
            scale = graph.initializers[node.input_values[1]].to_torch()
            bias_name = node.input_values[2] if len(node.input_values) > 2 else None

            torch_module = nn.LayerNorm(
                normalized_shape=input_shape[axis:],
                eps=epsilon,
                elementwise_affine=True,
            ).to(device=scale.device, dtype=scale.dtype)
            with torch.no_grad():
                torch_module.weight.data = scale
                if bias_name is not None:
                    torch_module.bias.data = graph.initializers[bias_name].to_torch()

            return OperationConverterResult(
                torch_module=torch_module,
                onnx_mapping=OnnxMapping(inputs=(node.input_values[0],), outputs=node.output_values),
            )

        return OperationConverterResult(
            torch_module=OnnxLayerNorm(axis=axis, epsilon=epsilon),
            onnx_mapping=onnx_mapping_from_node(node),
        )

    def patched_instance_norm(node, graph):
        epsilon = node.attributes.get("epsilon", 1e-5)
        momentum = 0.1
        if all(value_name in graph.initializers for value_name in node.input_values[1:]):
            input_shape = get_shape_from_value_info(graph.value_info[node.input_values[0]])
            spatial_rank = len(input_shape) - 2
            instance_norm_class = _IN_CLASS_FROM_SPATIAL_RANK[spatial_rank]
            scale = graph.initializers[node.input_values[1]].to_torch()
            bias = graph.initializers[node.input_values[2]].to_torch()

            torch_module = instance_norm_class(
                num_features=scale.size()[0],
                eps=epsilon,
                momentum=momentum,
                affine=True,
                track_running_stats=False,
            ).to(device=scale.device, dtype=scale.dtype)
            with torch.no_grad():
                torch_module.weight.data = scale
                torch_module.bias.data = bias

            return OperationConverterResult(
                torch_module=torch_module,
                onnx_mapping=OnnxMapping(inputs=(node.input_values[0],), outputs=node.output_values),
            )

        return OperationConverterResult(
            torch_module=OnnxInstanceNorm(momentum=momentum, epsilon=epsilon),
            onnx_mapping=onnx_mapping_from_node(node),
        )

    def patched_clip(node, graph):
        min_name = node.input_values[1] if len(node.input_values) > 1 and node.input_values[1] != "" else None
        max_name = node.input_values[2] if len(node.input_values) > 2 and node.input_values[2] != "" else None

        try:
            min_value = float(get_const_value(min_name, graph)) if min_name is not None else None
            max_value = float(get_const_value(max_name, graph)) if max_name is not None else None
        except KeyError as exc:
            raise NotImplementedError("Dynamic value of min/max is not implemented") from exc

        if min_value is None and max_value is None:
            torch_module = nn.Identity()
        elif min_value == 0 and max_value is None:
            torch_module = nn.ReLU()
        elif min_value == 0 and max_value == 6:
            torch_module = nn.ReLU6()
        else:
            torch_module = OnnxClip(min_val=min_value, max_val=max_value)

        return OperationConverterResult(
            torch_module=torch_module,
            onnx_mapping=OnnxMapping(inputs=(node.input_values[0],), outputs=node.output_values),
        )

    OnnxTensor.to_torch = to_torch_external_meta

    for version in (1, 11):
        _CONVERTER_REGISTRY[OperationDescription("", "Conv", version)] = patched_conv
        _CONVERTER_REGISTRY[OperationDescription("", "ConvTranspose", version)] = patched_conv
    for version in (9, 11, 13):
        _CONVERTER_REGISTRY[OperationDescription("", "Gemm", version)] = patched_gemm
    for version in (17,):
        _CONVERTER_REGISTRY[OperationDescription("", "LayerNormalization", version)] = patched_layer_norm
    for version in (1, 6):
        _CONVERTER_REGISTRY[OperationDescription("", "InstanceNormalization", version)] = patched_instance_norm
    for version in (11, 12, 13):
        _CONVERTER_REGISTRY[OperationDescription("", "Clip", version)] = patched_clip

    try:
        yield
    finally:
        OnnxTensor.to_torch = original_to_torch
        _CONVERTER_REGISTRY.clear()
        _CONVERTER_REGISTRY.update(original_registry)


def build_exact_fuxi_short_graph(model_path: str | Path = DEFAULT_MODEL_PATH) -> GraphModule:
    """Build the exact PyTorch graph from ``short.onnx``.

    The returned module mirrors the ONNX topology exactly. Large external
    weights remain on the ``meta`` device so this step stays RAM-safe on this
    machine; the goal here is exact architecture reconstruction, not eager
    materialization of every parameter.
    """

    resolved_model_path = Path(model_path).resolve()
    _check_model_files(resolved_model_path)
    onnx_model = onnx.load(str(resolved_model_path), load_external_data=False)

    with _patched_exact_onnx2torch_context(resolved_model_path.parent):
        graph_module = convert(onnx_model, save_input_names=True)

    graph_module._fuxi_source_onnx_path = str(resolved_model_path)
    return graph_module


def inspect_exact_fuxi_short_graph(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    graph_module: GraphModule | None = None,
) -> dict[str, Any]:
    """Return a compact summary of the exact ONNX-backed graph build."""

    resolved_model_path = Path(model_path).resolve()
    external_data_path = _check_model_files(resolved_model_path)
    onnx_model = onnx.load(str(resolved_model_path), load_external_data=False)
    graph_module = graph_module or build_exact_fuxi_short_graph(resolved_model_path)

    fx_nodes = list(graph_module.graph.nodes)
    state_tensors = list(graph_module.state_dict().values())
    op_counts = Counter(node.op for node in fx_nodes)
    forward_arg_names = list(
        graph_module.forward.__code__.co_varnames[1 : graph_module.forward.__code__.co_argcount]
    )

    return {
        "model_path": str(resolved_model_path),
        "external_data_path": str(external_data_path),
        "onnx_node_count": len(onnx_model.graph.node),
        "graph_node_count": len(fx_nodes),
        "op_counts": dict(op_counts),
        "forward_arg_names": forward_arg_names,
        "placeholder_names": [node.name for node in fx_nodes if node.op == "placeholder"],
        "input_names": [value.name for value in onnx_model.graph.input],
        "output_names": [value.name for value in onnx_model.graph.output],
        "state_tensor_count": len(state_tensors),
        "meta_state_tensor_count": sum(1 for tensor in state_tensors if tensor.device.type == "meta"),
        "cpu_state_tensor_count": sum(1 for tensor in state_tensors if tensor.device.type == "cpu"),
    }


@dataclass(frozen=True)
class FuXiShortConfig:
    """Configuration for building the exact short-range FuXi graph."""

    model_path: Path = DEFAULT_MODEL_PATH


class FuXiShort(nn.Module):
    """ONNX-backed exact architecture builder for FuXi short.

    This is not a hand-written approximation of FuXi. Instead it converts the
    local ``short.onnx`` graph into an exact PyTorch ``GraphModule`` so the
    Python-side architecture matches the checkpoint topology node-for-node.
    """

    def __init__(self, config: FuXiShortConfig | None = None) -> None:
        super().__init__()
        self.config = config or FuXiShortConfig()
        self.graph_module = build_exact_fuxi_short_graph(self.config.model_path)

    def forward(self, input: Tensor, temb: Tensor) -> Tensor:
        return self.graph_module(input, temb)

    def summary(self) -> dict[str, Any]:
        return inspect_exact_fuxi_short_graph(
            model_path=self.config.model_path,
            graph_module=self.graph_module,
        )


__all__ = [
    "DEFAULT_MODEL_PATH",
    "FuXiShort",
    "FuXiShortConfig",
    "build_exact_fuxi_short_graph",
    "build_fuxi_time_embeddings",
    "inspect_exact_fuxi_short_graph",
    "summarize_short_onnx_architecture",
]
