from modelopt_compat import patch_modelopt_environment

patch_modelopt_environment()

import modelopt.torch.opt as mto
from modelopt.torch._deploy.utils.torch_onnx import get_onnx_bytes_and_metadata, OnnxBytes
from modelopt.onnx import autocast as atc
import torch
import os
import logging
from pathlib import Path
from yaml import safe_dump

import tensorrt as trt
import onnx
from ultralytics import YOLO

main_dir = Path(__file__).resolve().parent.parent

raw_modelopt_path = os.path.join(main_dir, "PTCheckpoints", "best", "model.pth")
base_model_path = os.path.join(main_dir, "TrainingRuns", "train", "weights", "best.pt")
base_model = YOLO(base_model_path, task="detect")
base_model_names = base_model.names

device = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger("modelopt")

export_imgsz = 640
export_input_shape = (1, 3, export_imgsz, export_imgsz)
workspace_bytes = 2 << 30


def _normalize_class_names(names):
    if isinstance(names, dict):
        return {int(key): value for key, value in names.items()}
    return {index: value for index, value in enumerate(names)}


if not os.path.exists(raw_modelopt_path):
    raise FileNotFoundError(f"Missing ModelOpt checkpoint: {raw_modelopt_path}")

checkpoint = torch.load(raw_modelopt_path, map_location=device, weights_only=False)
if not isinstance(checkpoint, dict) or "modelopt_state" not in checkpoint or "model_state_dict" not in checkpoint:
    raise RuntimeError(f"{raw_modelopt_path} is not a raw ModelOpt checkpoint with modelopt_state/model_state_dict.")


def scrub_empty_quantizer_state(modelopt_state):
    """Remove empty quantizer_state dicts that break ModelOpt restore."""
    for _, payload in modelopt_state.get("modelopt_state_dict", []):
        metadata = payload.get("metadata", {})
        if metadata.get("quantizer_state") == {}:
            del metadata["quantizer_state"]


scrub_empty_quantizer_state(checkpoint["modelopt_state"])

base_yolo = YOLO(base_model_path, task="detect")
model = base_yolo.model

model = model.to(device).eval()

try:
    model = mto.restore_from_modelopt_state(model, modelopt_state=checkpoint["modelopt_state"])
    model.load_state_dict(checkpoint["model_state_dict"])
except Exception:
    raise RuntimeError(
        f"Failed to restore {raw_modelopt_path}. If this checkpoint was created before the "
        "no-fuse validation fix in Exporters/exportPTQAT.py, regenerate it before exporting."
    ) from None


model = model.to(device).eval()


class ExportWrapper(torch.nn.Module):
    def __init__(self, wrapped_model):
        super().__init__()
        self.wrapped_model = wrapped_model
        self.detect_head = wrapped_model.model[-1]
        self.detect_head.export = True
        self.detect_head.format = "onnx"

    def forward(self, images):
        outputs = self.wrapped_model(images)
        return outputs[0] if isinstance(outputs, tuple) else outputs


weights_dtype = "fp32"
dummy_input = torch.randn(export_input_shape, device=device)

export_model = ExportWrapper(model).to(device).eval()

onnx_dir = os.path.join(main_dir, "ONNXExports")
os.makedirs(onnx_dir, exist_ok=True)

print(f"Restored raw ModelOpt checkpoint from {raw_modelopt_path}")

use_dynamo_export = False

onnx_path = os.path.join(onnx_dir, "yolo26nApriltagINT8.onnx")
mixed_precision_onnx_path = os.path.join(onnx_dir, "yolo26nApriltagMixedINT8.onnx")


def confirm_onnx_export(onnx_model_path):
    m = onnx.load(onnx_model_path)
    ops = set(node.op_type for node in m.graph.node)
    print(f"Exported ONNX Model Ops: {sorted(ops)}")


def _save_mixed_precision_onnx(source_path, destination_path):
    print("Converting FP32/INT8 ONNX graph to mixed FP16/INT8 for higher performance...")
    
    converted_model = atc.convert_to_mixed_precision(
        onnx_path=str(source_path),
        low_precision_type="fp16",
        keep_io_types=True,
        providers=["cuda:0", "cpu"] if device == "cuda" else ["cpu"],
        use_standalone_type_inference=True,
    )

    onnx.checker.check_model(converted_model)
    onnx.save(converted_model, destination_path)
    print(f"Saved ModelOpt AutoCast mixed-precision ONNX to {destination_path}")
    return destination_path


def export_onnx_model(model, dummy_input, onnx_dir, onnx_path, weights_dtype, dynamo_export):
    
    print("Exporting ONNX model with ModelOpt...")
    
    onnx_bytes, _ = get_onnx_bytes_and_metadata(
        model=model,
        dummy_input=dummy_input,
        model_name="yolo26nApriltagINT8",
        weights_dtype=weights_dtype,
        dynamo_export=dynamo_export
    )

    onnx_obj = OnnxBytes.from_bytes(onnx_bytes)
    onnx_obj.write_to_disk(onnx_dir, clean_dir=True)

    print('-' * 60)
    confirm_onnx_export(onnx_path)

    print('-' * 60)
    print(f"Saved ONNX to {onnx_dir}")


export_onnx_model(export_model, dummy_input, onnx_dir, onnx_path, weights_dtype, use_dynamo_export)
build_onnx_path = _save_mixed_precision_onnx(onnx_path, mixed_precision_onnx_path)


def _create_network(builder):
    if hasattr(trt, "NetworkDefinitionCreationFlag") and hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
        return builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    return builder.create_network()


def _load_timing_cache(config, cache_path):
    if not hasattr(config, "create_timing_cache") or not hasattr(config, "set_timing_cache"):
        return None

    cache_path = Path(cache_path)
    cache_bytes = cache_path.read_bytes() if cache_path.exists() else b""
    try:
        timing_cache = config.create_timing_cache(cache_bytes)
        config.set_timing_cache(timing_cache, ignore_mismatch=False)
        return timing_cache
    except Exception as exc:
        print(f"Ignoring incompatible TensorRT timing cache {cache_path}: {exc}")
        timing_cache = config.create_timing_cache(b"")
        config.set_timing_cache(timing_cache, ignore_mismatch=True)
        return timing_cache


def _save_timing_cache(config, cache_path):
    if not hasattr(config, "get_timing_cache"):
        return

    timing_cache = config.get_timing_cache()
    if timing_cache is None:
        return

    Path(cache_path).write_bytes(bytes(timing_cache.serialize()))
    print(f"Saved TensorRT timing cache to {cache_path}")


def build_trt_engine(onnx_path):
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    trt_builder = trt.Builder(TRT_LOGGER)
    trt_runtime = trt.Runtime(TRT_LOGGER)
    network = _create_network(trt_builder)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = trt_builder.create_builder_config()
    config.builder_optimization_level = 5
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    timing_cache_path = Path(trt_path).with_suffix(".timing.cache")
    _load_timing_cache(config, timing_cache_path)

    with open(onnx_path, "rb") as f:
        onnx_bytes = f.read()

    if not parser.parse(onnx_bytes):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parse failed")

    print("Inputs:", network.num_inputs)
    print("Outputs:", network.num_outputs)
    print("Layers:", network.num_layers)
    print("Building TensorRT engine...")

    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    serialized_engine = trt_builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT engine build failed")

    _save_timing_cache(config, timing_cache_path)

    with open(trt_path, "wb") as f:
        f.write(bytes(serialized_engine))

    write_engine_metadata(trt_path, base_model_names, export_imgsz)

    engine = trt_runtime.deserialize_cuda_engine(serialized_engine)
    inspector = engine.create_engine_inspector()

    print('-' * 60)
    print(f"TensorRT Engine Information:\n{inspector.get_engine_information(trt.LayerInformationFormat.JSON)}")
    print('-' * 60)
    print(f"Saved TensorRT engine to {trt_path}")


def write_engine_metadata(engine_path, names, imgsz):
    metadata_path = Path(engine_path).with_name(f"{Path(engine_path).stem}.metadata.yaml")
    metadata = {
        "task": "detect",
        "nc": int(len(names)),
        "names": list(_normalize_class_names(names).values()),
        "imgsz": int(imgsz),
        "precision": "explicit_int8_qdq_plus_fp16",
    }
    metadata_path.write_text(safe_dump(metadata, sort_keys=False), encoding="utf-8")
    print(f"Saved class metadata to {metadata_path}")
    return metadata_path

if device == 'cuda':
    trt_dir = os.path.join(main_dir, "TensorRTExports")
    os.makedirs(trt_dir, exist_ok=True)
    trt_path = os.path.join(trt_dir, "yolo26nApriltagMixedINT8.engine")
    build_trt_engine(build_onnx_path)

else:
    print('-' * 60)
    print("Device is CPU. Skipping TensorRT engine build.")
    print(f"To build a TensorRT engine, run this script on an NVIDIA GPU.")
    print('-' * 60)