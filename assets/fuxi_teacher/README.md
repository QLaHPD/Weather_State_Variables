FuXi teacher assets live in this folder.

The ONNX checkpoints use ONNX external data:
- `short.onnx` contains the graph definition.
- `short` contains external tensor data for `short.onnx`.
- The same pattern applies to `medium` and `long`.

Expected layout:

```text
assets/fuxi_teacher/
  short.onnx
  short
  medium.onnx
  medium
  long.onnx
  long
```

Direct download URLs, based on `ai-models-fuxi`:

```text
https://get.ecmwf.int/repository/test-data/ai-models/fuxi/short.onnx
https://get.ecmwf.int/repository/test-data/ai-models/fuxi/short
https://get.ecmwf.int/repository/test-data/ai-models/fuxi/medium.onnx
https://get.ecmwf.int/repository/test-data/ai-models/fuxi/medium
https://get.ecmwf.int/repository/test-data/ai-models/fuxi/long.onnx
https://get.ecmwf.int/repository/test-data/ai-models/fuxi/long
```

If you only want to make `short.onnx` loadable right now, the minimum missing file is:

```text
assets/fuxi_teacher/short
```
