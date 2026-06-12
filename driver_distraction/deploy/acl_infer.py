"""AscendCL Python inference wrapper for fixed-batch image classification OM models."""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Sequence

import numpy as np
from PIL import Image

from driver_distraction.constants import STATE_FARM_CLASS_NAMES, STATE_FARM_CLASS_NAMES_ZH


ACL_SUCCESS = 0
ACL_HOST = 0
ACL_DEVICE = 1
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class AscendAclError(RuntimeError):
    """Raised when an AscendCL API call fails."""


@dataclass(frozen=True)
class Prediction:
    index: int
    label: str
    label_zh: str
    probability: float


def load_acl_module() -> ModuleType:
    try:
        return importlib.import_module("acl")
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import the Ascend ACL Python module. Source the CANN environment first, for example:\n"
            "  source /usr/local/Ascend/ascend-toolkit/set_env.sh\n"
            "or:\n"
            "  source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh"
        ) from exc


def preprocess_image(
    image_path: str | Path,
    input_size: int = 224,
    resize_size: int = 256,
) -> np.ndarray:
    """Apply the same deterministic preprocessing used for validation and export."""

    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")

    return preprocess_pil_image(image, input_size=input_size, resize_size=resize_size)


def preprocess_bgr_frame(
    frame: np.ndarray,
    input_size: int = 224,
    resize_size: int = 256,
) -> np.ndarray:
    """Convert an OpenCV BGR frame to the exported model input tensor."""

    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected a BGR frame shaped (H, W, 3), got {frame.shape}")
    rgb = np.ascontiguousarray(frame[:, :, ::-1])
    image = Image.fromarray(rgb, mode="RGB")
    return preprocess_pil_image(image, input_size=input_size, resize_size=resize_size)


def preprocess_pil_image(
    image: Image.Image,
    input_size: int = 224,
    resize_size: int = 256,
) -> np.ndarray:
    """Preprocess an RGB PIL image using torchvision-compatible geometry."""

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {image.size}")

    if width < height:
        resized_width = resize_size
        resized_height = int(round(height * resize_size / width))
    else:
        resized_height = resize_size
        resized_width = int(round(width * resize_size / height))

    image = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    left = max(0, int(round((resized_width - input_size) / 2.0)))
    top = max(0, int(round((resized_height - input_size) / 2.0)))
    image = image.crop((left, top, left + input_size, top + input_size))

    array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    array = (array - mean) / std
    array = np.transpose(array, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(array, dtype=np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    values = values - np.max(values)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values)


class AscendOMClassifier:
    """Load one OM model and execute repeated synchronous inference calls."""

    def __init__(
        self,
        model_path: str | Path,
        device_id: int = 0,
        num_classes: int = 10,
        output_dtype: str = "auto",
        acl_module: ModuleType | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"OM model not found: {self.model_path}")
        if output_dtype not in {"auto", "float32", "float16"}:
            raise ValueError("output_dtype must be one of: auto, float32, float16")

        self.device_id = int(device_id)
        self.num_classes = int(num_classes)
        self.output_dtype = output_dtype
        self.acl = acl_module or load_acl_module()

        self.context = None
        self.model_id = None
        self.model_desc = None
        self.input_dataset = None
        self.output_dataset = None
        self.input_data_buffers: list[object] = []
        self.output_data_buffers: list[object] = []
        self.input_device_buffers: list[tuple[int, int]] = []
        self.output_device_buffers: list[tuple[int, int]] = []
        self.run_mode = ACL_HOST
        self._acl_initialized = False
        self._device_set = False
        self._closed = False

        try:
            self._initialize()
        except Exception:
            self.close()
            raise

    def _initialize(self) -> None:
        self._check(self.acl.init(), "acl.init")
        self._acl_initialized = True

        self._check(self.acl.rt.set_device(self.device_id), "acl.rt.set_device")
        self._device_set = True

        self.context = self._value(self.acl.rt.create_context(self.device_id), "acl.rt.create_context")
        self.run_mode = int(self._value(self.acl.rt.get_run_mode(), "acl.rt.get_run_mode"))

        self.model_id = self._value(
            self.acl.mdl.load_from_file(str(self.model_path)),
            "acl.mdl.load_from_file",
        )
        self.model_desc = self.acl.mdl.create_desc()
        if self.model_desc is None:
            raise AscendAclError("acl.mdl.create_desc returned None")
        self._check(
            self.acl.mdl.get_desc(self.model_desc, self.model_id),
            "acl.mdl.get_desc",
        )

        num_inputs = int(self.acl.mdl.get_num_inputs(self.model_desc))
        num_outputs = int(self.acl.mdl.get_num_outputs(self.model_desc))
        if num_inputs != 1:
            raise ValueError(f"Expected exactly one model input, found {num_inputs}")
        if num_outputs < 1:
            raise ValueError("The OM model has no outputs")

        self.input_dataset = self._create_dataset(
            [
                int(self.acl.mdl.get_input_size_by_index(self.model_desc, index))
                for index in range(num_inputs)
            ],
            self.input_device_buffers,
            self.input_data_buffers,
            "input",
        )
        self.output_dataset = self._create_dataset(
            [
                int(self.acl.mdl.get_output_size_by_index(self.model_desc, index))
                for index in range(num_outputs)
            ],
            self.output_device_buffers,
            self.output_data_buffers,
            "output",
        )

    def _create_dataset(
        self,
        sizes: Sequence[int],
        device_buffers: list[tuple[int, int]],
        data_buffers: list[object],
        name: str,
    ):
        dataset = self.acl.mdl.create_dataset()
        if dataset is None:
            raise AscendAclError(f"acl.mdl.create_dataset returned None for {name}")

        device_start = len(device_buffers)
        data_start = len(data_buffers)
        try:
            for index, size in enumerate(sizes):
                pointer = self._value(
                    self.acl.rt.malloc(size, ACL_MEM_MALLOC_HUGE_FIRST),
                    f"acl.rt.malloc({name}[{index}])",
                )
                device_buffers.append((pointer, size))

                data_buffer = self.acl.create_data_buffer(pointer, size)
                if data_buffer is None:
                    raise AscendAclError(f"acl.create_data_buffer returned None for {name}[{index}]")
                data_buffers.append(data_buffer)
                self._check_result(
                    self.acl.mdl.add_dataset_buffer(dataset, data_buffer),
                    f"acl.mdl.add_dataset_buffer({name}[{index}])",
                )
        except Exception:
            for data_buffer in data_buffers[data_start:]:
                self._ignore_error(self.acl.destroy_data_buffer, data_buffer)
            del data_buffers[data_start:]
            self._ignore_error(self.acl.mdl.destroy_dataset, dataset)
            for pointer, _ in device_buffers[device_start:]:
                self._ignore_error(self.acl.rt.free, pointer)
            del device_buffers[device_start:]
            raise
        return dataset

    @property
    def input_nbytes(self) -> int:
        return self.input_device_buffers[0][1]

    @property
    def output_nbytes(self) -> list[int]:
        return [size for _, size in self.output_device_buffers]

    def infer(self, input_tensor: np.ndarray, output_index: int = 0) -> np.ndarray:
        if self._closed:
            raise RuntimeError("AscendOMClassifier is already closed")
        if output_index < 0 or output_index >= len(self.output_device_buffers):
            raise IndexError(f"output_index out of range: {output_index}")

        tensor = np.ascontiguousarray(input_tensor, dtype=np.float32)
        if tensor.nbytes != self.input_nbytes:
            raise ValueError(
                f"Input byte size mismatch: model expects {self.input_nbytes}, "
                f"but tensor {tensor.shape} contains {tensor.nbytes} bytes"
            )

        self._set_context()
        input_pointer, input_size = self.input_device_buffers[0]
        source_pointer = self.acl.util.numpy_to_ptr(tensor)
        input_copy_kind = (
            ACL_MEMCPY_HOST_TO_DEVICE if self.run_mode == ACL_HOST else ACL_MEMCPY_DEVICE_TO_DEVICE
        )
        self._check(
            self.acl.rt.memcpy(
                input_pointer,
                input_size,
                source_pointer,
                tensor.nbytes,
                input_copy_kind,
            ),
            "acl.rt.memcpy(input)",
        )

        self._check(
            self.acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset),
            "acl.mdl.execute",
        )
        return self._copy_output(output_index)

    def predict(
        self,
        image_path: str | Path,
        top_k: int = 5,
        input_size: int = 224,
        resize_size: int = 256,
    ) -> list[Prediction]:
        tensor = preprocess_image(image_path, input_size=input_size, resize_size=resize_size)
        logits = self.infer(tensor)
        if logits.size < self.num_classes:
            raise ValueError(
                f"Output contains only {logits.size} values; expected at least {self.num_classes}"
            )

        probabilities = softmax(logits[: self.num_classes])
        top_k = max(1, min(int(top_k), self.num_classes))
        indices = np.argsort(probabilities)[::-1][:top_k]
        return [
            Prediction(
                index=int(index),
                label=STATE_FARM_CLASS_NAMES[int(index)],
                label_zh=STATE_FARM_CLASS_NAMES_ZH[int(index)],
                probability=float(probabilities[int(index)]),
            )
            for index in indices
        ]

    def benchmark(
        self,
        input_tensor: np.ndarray,
        warmup_runs: int = 3,
        measured_runs: int = 20,
    ) -> dict[str, float]:
        for _ in range(max(0, int(warmup_runs))):
            self.infer(input_tensor)

        latencies_ms: list[float] = []
        for _ in range(max(1, int(measured_runs))):
            started_at = time.perf_counter()
            self.infer(input_tensor)
            latencies_ms.append((time.perf_counter() - started_at) * 1000.0)

        values = np.asarray(latencies_ms, dtype=np.float64)
        return {
            "runs": float(len(latencies_ms)),
            "mean_ms": float(np.mean(values)),
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "fps": float(1000.0 / np.mean(values)),
        }

    def _copy_output(self, output_index: int) -> np.ndarray:
        device_pointer, output_size = self.output_device_buffers[output_index]
        host_bytes = np.empty(output_size, dtype=np.uint8)
        host_pointer = self.acl.util.numpy_to_ptr(host_bytes)
        output_copy_kind = (
            ACL_MEMCPY_DEVICE_TO_HOST if self.run_mode == ACL_HOST else ACL_MEMCPY_DEVICE_TO_DEVICE
        )
        self._check(
            self.acl.rt.memcpy(
                host_pointer,
                output_size,
                device_pointer,
                output_size,
                output_copy_kind,
            ),
            f"acl.rt.memcpy(output[{output_index}])",
        )

        dtype = self._resolve_output_dtype(output_size)
        return np.frombuffer(host_bytes, dtype=dtype).astype(np.float32, copy=True)

    def _resolve_output_dtype(self, output_size: int) -> np.dtype:
        if self.output_dtype == "float32":
            return np.dtype(np.float32)
        if self.output_dtype == "float16":
            return np.dtype(np.float16)
        if output_size == self.num_classes * np.dtype(np.float32).itemsize:
            return np.dtype(np.float32)
        if output_size == self.num_classes * np.dtype(np.float16).itemsize:
            return np.dtype(np.float16)
        if output_size % np.dtype(np.float32).itemsize == 0:
            return np.dtype(np.float32)
        if output_size % np.dtype(np.float16).itemsize == 0:
            return np.dtype(np.float16)
        raise ValueError(
            f"Cannot infer output dtype from {output_size} bytes. "
            "Set output_dtype to float32 or float16 explicitly."
        )

    def _set_context(self) -> None:
        if self.context is not None and hasattr(self.acl.rt, "set_context"):
            self._check(self.acl.rt.set_context(self.context), "acl.rt.set_context")

    @staticmethod
    def _check(ret: int | None, operation: str) -> None:
        if ret is not None and int(ret) != ACL_SUCCESS:
            raise AscendAclError(f"{operation} failed with ACL error code {ret}")

    @classmethod
    def _check_result(cls, result, operation: str) -> None:
        if isinstance(result, tuple):
            cls._check(result[-1], operation)
        else:
            cls._check(result, operation)

    @classmethod
    def _value(cls, result, operation: str):
        if not isinstance(result, tuple) or len(result) < 2:
            raise AscendAclError(f"{operation} returned an unexpected value: {result!r}")
        cls._check(result[-1], operation)
        return result[0]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        self._destroy_dataset(self.input_dataset, self.input_data_buffers)
        self.input_dataset = None
        self._destroy_dataset(self.output_dataset, self.output_data_buffers)
        self.output_dataset = None

        for pointer, _ in [*self.input_device_buffers, *self.output_device_buffers]:
            if pointer:
                self._ignore_error(self.acl.rt.free, pointer)
        self.input_device_buffers.clear()
        self.output_device_buffers.clear()

        if self.model_id is not None:
            self._ignore_error(self.acl.mdl.unload, self.model_id)
            self.model_id = None
        if self.model_desc is not None:
            self._ignore_error(self.acl.mdl.destroy_desc, self.model_desc)
            self.model_desc = None
        if self.context is not None:
            self._ignore_error(self.acl.rt.destroy_context, self.context)
            self.context = None
        if self._device_set:
            self._ignore_error(self.acl.rt.reset_device, self.device_id)
            self._device_set = False
        if self._acl_initialized:
            self._ignore_error(self.acl.finalize)
            self._acl_initialized = False

    def _destroy_dataset(self, dataset, data_buffers: list[object]) -> None:
        for data_buffer in data_buffers:
            if data_buffer is not None:
                self._ignore_error(self.acl.destroy_data_buffer, data_buffer)
        data_buffers.clear()
        if dataset is not None:
            self._ignore_error(self.acl.mdl.destroy_dataset, dataset)

    @staticmethod
    def _ignore_error(function, *args) -> None:
        try:
            function(*args)
        except Exception:
            pass

    def __enter__(self) -> "AscendOMClassifier":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
