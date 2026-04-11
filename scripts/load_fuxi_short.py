#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch

from weather_state_variables.models import (
    build_exact_fuxi_short_graph,
    inspect_exact_fuxi_short_graph,
    summarize_short_onnx_architecture,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = REPO_ROOT / "assets" / "fuxi_teacher" / "short.onnx"
DEFAULT_EXPORT_PATH = REPO_ROOT / "assets" / "fuxi_teacher" / "short_tensors.pt"


def _check_model_files(model_path: Path) -> Path:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    external_data_path = model_path.with_suffix("")
    if not external_data_path.is_file():
        raise FileNotFoundError(
            "FuXi short.onnx requires a sibling external-data file named "
            f"'{external_data_path.name}'. Missing: {external_data_path}"
        )
    return external_data_path


def _build_session(model_path: Path, provider: str, num_threads: int):
    import onnxruntime as ort

    ort.set_default_logger_severity(3)
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    options.intra_op_num_threads = num_threads

    provider = provider.lower()
    if provider == "cpu":
        providers: list[Any] = ["CPUExecutionProvider"]
    elif provider == "cuda":
        providers = [("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})]
    else:
        raise ValueError(f"Unsupported provider '{provider}'. Use 'cpu' or 'cuda'.")

    return ort.InferenceSession(str(model_path), sess_options=options, providers=providers)


def _tensor_info(value: Any) -> dict[str, Any]:
    return {
        "name": value.name,
        "shape": list(value.shape),
        "type": value.type,
    }


def describe_session(session, model_path: Path, external_data_path: Path) -> dict[str, Any]:
    providers = session.get_providers()
    inputs = [_tensor_info(value) for value in session.get_inputs()]
    outputs = [_tensor_info(value) for value in session.get_outputs()]

    summary = {
        "model_path": str(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "external_data_path": str(external_data_path),
        "external_data_size_bytes": external_data_path.stat().st_size,
        "providers": providers,
        "inputs": inputs,
        "outputs": outputs,
    }
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    def fmt_size(size_bytes: int) -> str:
        value = float(size_bytes)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if value < 1024.0 or unit == "TB":
                return f"{value:.1f}{unit}"
            value /= 1024.0
        return f"{size_bytes}B"

    print("FuXi short model loaded successfully.")
    print(f"model: {summary['model_path']}")
    print(f"model_size: {fmt_size(summary['model_size_bytes'])}")
    print(f"external_data: {summary['external_data_path']}")
    print(f"external_data_size: {fmt_size(summary['external_data_size_bytes'])}")
    print(f"providers: {', '.join(summary['providers'])}")
    print("inputs:")
    for value in summary["inputs"]:
        print(f"  - {value['name']}: shape={value['shape']} type={value['type']}")
    print("outputs:")
    for value in summary["outputs"]:
        print(f"  - {value['name']}: shape={value['shape']} type={value['type']}")


def print_graph_summary(summary: dict[str, Any]) -> None:
    print("exact_graph:")
    print(f"  - model_path: {summary['model_path']}")
    print(f"  - external_data_path: {summary['external_data_path']}")
    print(f"  - input_names: {summary['input_names']}")
    print(f"  - output_names: {summary['output_names']}")
    print(f"  - forward_arg_names: {summary['forward_arg_names']}")
    print(f"  - onnx_node_count: {summary['onnx_node_count']}")
    print(f"  - graph_node_count: {summary['graph_node_count']}")
    print(f"  - call_module_count: {summary['op_counts'].get('call_module', 0)}")
    print(f"  - meta_state_tensor_count: {summary['meta_state_tensor_count']}")
    print(f"  - cpu_state_tensor_count: {summary['cpu_state_tensor_count']}")


def print_architecture_summary(summary: dict[str, Any]) -> None:
    print("architecture_recipe:")
    print(f"  - input_shape: {summary['input_shape']}")
    print(f"  - resized_input_size: {summary['resized_input_size']}")
    print(f"  - patch_size: {summary['patch_size']}")
    print(f"  - patch_grid: {summary['patch_grid']}")
    print(f"  - latent_grid: {summary['latent_grid']}")
    print(f"  - aux_chans: {summary['aux_chans']}")
    print(f"  - embed_dim: {summary['embed_dim']}")
    print(f"  - num_heads: {summary['num_heads']}")
    print(f"  - window_size: {summary['window_size']}")
    print(f"  - depths: {summary['depths']}")
    print(f"  - mlp_hidden_dim: {summary['mlp_hidden_dim']}")
    print(f"  - fpn_in_dim: {summary['fpn_in_dim']}")
    print(f"  - fpn_out_dim: {summary['fpn_out_dim']}")
    print(f"  - head_out_dim: {summary['head_out_dim']}")


def export_onnx_tensors_to_torch(model_path: Path, export_path: Path) -> dict[str, Any]:
    try:
        import onnx
        from onnx import external_data_helper, numpy_helper
    except ImportError as exc:
        raise RuntimeError(
            "The 'onnx' package is required for --export-tensors. "
            "Install it in the active environment first."
        ) from exc

    model = onnx.load(str(model_path), load_external_data=False)
    base_dir = str(model_path.parent)
    initializers: dict[str, torch.Tensor] = {}
    total_numel = 0
    total_bytes = 0
    initializer_count = len(model.graph.initializer)

    for index, initializer in enumerate(model.graph.initializer, start=1):
        if external_data_helper.uses_external_data(initializer):
            array = numpy_helper.to_array(initializer, base_dir=base_dir)
        else:
            array = numpy_helper.to_array(initializer)
        tensor = torch.tensor(array)
        initializers[initializer.name] = tensor
        total_numel += tensor.numel()
        total_bytes += tensor.numel() * tensor.element_size()

        del array
        initializer.ClearField("raw_data")

        if index % 25 == 0 or index == initializer_count:
            print(
                f"export progress: {index}/{initializer_count} tensors "
                f"({total_bytes} bytes converted)",
                flush=True,
            )
            gc.collect()

    del model
    gc.collect()

    payload = {
        "model_path": str(model_path),
        "graph_name": model.graph.name,
        "ir_version": model.ir_version,
        "opset_import": {entry.domain or "ai.onnx": entry.version for entry in model.opset_import},
        "inputs": [
            {
                "name": value.name,
                "shape": [
                    dim.dim_value if dim.HasField("dim_value") else dim.dim_param
                    for dim in value.type.tensor_type.shape.dim
                ],
                "elem_type": value.type.tensor_type.elem_type,
            }
            for value in model.graph.input
        ],
        "outputs": [
            {
                "name": value.name,
                "shape": [
                    dim.dim_value if dim.HasField("dim_value") else dim.dim_param
                    for dim in value.type.tensor_type.shape.dim
                ],
                "elem_type": value.type.tensor_type.elem_type,
            }
            for value in model.graph.output
        ],
        "initializers": initializers,
        "initializer_count": len(initializers),
        "initializer_total_numel": total_numel,
        "initializer_total_bytes": total_bytes,
    }

    export_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, export_path)

    return {
        "export_path": str(export_path),
        "initializer_count": len(initializers),
        "initializer_total_numel": total_numel,
        "initializer_total_bytes": total_bytes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load the local FuXi short ONNX checkpoint, describe its I/O contract, "
            "and optionally export ONNX tensors into a Torch .pt file."
        )
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to short.onnx (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--provider",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Execution provider used to validate that the ONNX model loads.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="onnxruntime intra-op thread count for the load check.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the session summary as JSON instead of plain text.",
    )
    parser.add_argument(
        "--build-exact-graph",
        action="store_true",
        help=(
            "Build the exact PyTorch GraphModule from short.onnx using the ONNX graph itself. "
            "Large external tensors stay on the meta device to avoid RAM blow-ups."
        ),
    )
    parser.add_argument(
        "--print-architecture",
        action="store_true",
        help="Print a concise architecture recipe extracted from the short ONNX graph.",
    )
    parser.add_argument(
        "--export-tensors",
        type=Path,
        nargs="?",
        const=DEFAULT_EXPORT_PATH,
        help=(
            "Optional path for a Torch .pt export containing ONNX initializers and metadata. "
            f"If omitted, defaults to {DEFAULT_EXPORT_PATH}."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model_path.resolve()
    external_data_path = _check_model_files(model_path)

    session = _build_session(model_path, provider=args.provider, num_threads=args.threads)
    summary = describe_session(session, model_path, external_data_path)
    graph_summary: dict[str, Any] | None = None
    architecture_summary: dict[str, Any] | None = None
    export_result: dict[str, Any] | None = None

    if args.build_exact_graph:
        graph_module = build_exact_fuxi_short_graph(model_path)
        graph_summary = inspect_exact_fuxi_short_graph(model_path=model_path, graph_module=graph_module)
    if args.print_architecture:
        architecture_summary = summarize_short_onnx_architecture(model_path)

    if args.json:
        payload: dict[str, Any] = {"session": summary}
        if graph_summary is not None:
            payload["exact_graph"] = graph_summary
        if architecture_summary is not None:
            payload["architecture_recipe"] = architecture_summary
        if args.export_tensors is not None:
            export_path = args.export_tensors.resolve()
            export_result = export_onnx_tensors_to_torch(model_path, export_path)
            payload["export"] = export_result
        print(json.dumps(payload, indent=2))
    else:
        print_summary(summary)
        if graph_summary is not None:
            print_graph_summary(graph_summary)
        if architecture_summary is not None:
            print_architecture_summary(architecture_summary)
        if args.export_tensors is not None:
            export_path = args.export_tensors.resolve()
            export_result = export_onnx_tensors_to_torch(model_path, export_path)
            print(
                "torch export complete: "
                f"{export_result['export_path']} "
                f"({export_result['initializer_count']} tensors, "
                f"{export_result['initializer_total_bytes']} bytes)"
            )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
