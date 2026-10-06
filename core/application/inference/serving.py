"""Inference serving adapters for Triton, TensorRT artifacts, and multi-GPU routing.

The module keeps transport and hardware concerns behind small interfaces so the
same scheduling policy is testable without a GPU, Triton server, or network.
"""

from __future__ import annotations

import json
import math
import os
import threading
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


class TritonInferenceError(RuntimeError):
    """Raised when a Triton request fails or returns an invalid payload."""


Transport = Callable[[str, Mapping[str, Any], float], Mapping[str, Any]]

_DATATYPE_BY_KIND: dict[str, str] = {
    "f2": "FP16",
    "f4": "FP32",
    "f8": "FP64",
    "i1": "INT8",
    "i2": "INT16",
    "i4": "INT32",
    "i8": "INT64",
    "u1": "UINT8",
    "u2": "UINT16",
    "u4": "UINT32",
    "u8": "UINT64",
}
_DTYPE_BY_DATATYPE: dict[str, Any] = {
    "FP16": np.float16,
    "FP32": np.float32,
    "FP64": np.float64,
    "INT8": np.int8,
    "INT16": np.int16,
    "INT32": np.int32,
    "INT64": np.int64,
    "UINT8": np.uint8,
    "UINT16": np.uint16,
    "UINT32": np.uint32,
    "UINT64": np.uint64,
}


def triton_datatype(array: np.ndarray) -> str:
    kind = array.dtype.kind + str(array.dtype.itemsize)
    try:
        return _DATATYPE_BY_KIND[kind]
    except KeyError as exc:  # pragma: no cover - defensive for exotic dtypes
        raise TritonInferenceError(f"unsupported Triton dtype: {array.dtype}") from exc


@dataclass(frozen=True, slots=True)
class TritonModelSpec:
    name: str
    version: str = "-1"
    max_batch_size: int = 8
    timeout_seconds: float = 2.0

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name:
            raise ValueError("Triton model name must be non-empty without slashes")
        if not self.version:
            raise ValueError("Triton model version must be non-empty")
        if self.max_batch_size < 1:
            raise ValueError("maximum batch size must be positive")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout must be finite and positive")

    @property
    def infer_path(self) -> str:
        return f"/v2/models/{self.name}/versions/{self.version}/infer"


class TritonHttpClient:
    """Minimal Triton v2 HTTP client with an injectable transport."""

    def __init__(
        self,
        base_url: str,
        model_specs: Mapping[str, TritonModelSpec],
        *,
        transport: Transport | None = None,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("Triton base URL must be HTTP(S)")
        if not model_specs:
            raise ValueError("at least one model specification is required")
        self._base_url = base_url.rstrip("/")
        self._specs = dict(model_specs)
        self._transport = transport or self._urllib_transport

    @staticmethod
    def _urllib_transport(
        url: str, payload: Mapping[str, Any], timeout: float
    ) -> Mapping[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def health(self) -> bool:
        url = f"{self._base_url}/v2/health/ready"
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                return 200 <= response.status < 300
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def infer(
        self, model_name: str, inputs: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        try:
            spec = self._specs[model_name]
        except KeyError as exc:
            raise TritonInferenceError(f"model {model_name!r} is not configured") from exc
        if not inputs:
            raise TritonInferenceError("at least one input tensor is required")

        payload_inputs = []
        for name, value in inputs.items():
            array = np.ascontiguousarray(value)
            if array.ndim == 0:
                raise TritonInferenceError("input tensors must have at least one dimension")
            if array.shape[0] > spec.max_batch_size:
                raise TritonInferenceError(
                    f"batch {array.shape[0]} exceeds configured maximum "
                    f"{spec.max_batch_size} for {model_name!r}"
                )
            payload_inputs.append(
                {
                    "name": name,
                    "shape": list(array.shape),
                    "datatype": triton_datatype(array),
                    "data": array.reshape(-1).tolist(),
                }
            )

        response = self._transport(
            f"{self._base_url}{spec.infer_path}",
            {"inputs": payload_inputs},
            spec.timeout_seconds,
        )
        if "error" in response:
            raise TritonInferenceError(str(response["error"]))
        raw_outputs = response.get("outputs")
        if not isinstance(raw_outputs, list) or not raw_outputs:
            raise TritonInferenceError("Triton response contains no outputs")

        outputs: dict[str, np.ndarray] = {}
        for output in raw_outputs:
            try:
                name = str(output["name"])
                datatype = str(output["datatype"])
                shape = tuple(int(dimension) for dimension in output["shape"])
                data = output["data"]
            except (KeyError, TypeError, ValueError) as exc:
                raise TritonInferenceError("Triton output is malformed") from exc
            if datatype not in _DTYPE_BY_DATATYPE:
                raise TritonInferenceError(f"unsupported Triton output datatype {datatype}")
            if any(dimension < 0 for dimension in shape):
                raise TritonInferenceError("Triton output shape must be non-negative")
            array = np.asarray(data, dtype=_DTYPE_BY_DATATYPE[datatype])
            if array.size != int(np.prod(shape, dtype=np.int64)):
                raise TritonInferenceError("Triton output element count does not match shape")
            outputs[name] = array.reshape(shape)
        return outputs


@dataclass(frozen=True, slots=True)
class TensorRtArtifact:
    engine_path: str
    precision: str = "fp16"
    device_id: int = 0
    max_batch_size: int = 8

    def __post_init__(self) -> None:
        if not self.engine_path:
            raise ValueError("TensorRT engine path is required")
        if self.precision not in {"fp32", "fp16", "int8"}:
            raise ValueError("precision must be fp32, fp16, or int8")
        if self.device_id < 0:
            raise ValueError("device id must be non-negative")
        if self.max_batch_size < 1:
            raise ValueError("maximum batch size must be positive")

    def verify_engine_file(self) -> None:
        if not os.path.isfile(self.engine_path):
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")
        if os.path.getsize(self.engine_path) == 0:
            raise ValueError(f"TensorRT engine is empty: {self.engine_path}")


@dataclass(frozen=True, slots=True)
class GpuDevice:
    device_id: int
    name: str
    memory_mb: int
    max_inflight: int = 1

    def __post_init__(self) -> None:
        if self.device_id < 0:
            raise ValueError("device id must be non-negative")
        if not self.name:
            raise ValueError("device name is required")
        if self.memory_mb < 0:
            raise ValueError("device memory must be non-negative")
        if self.max_inflight < 1:
            raise ValueError("max inflight must be positive")


@dataclass(frozen=True, slots=True)
class GpuLease:
    device: GpuDevice
    lease_id: str


@dataclass(slots=True)
class _DeviceState:
    device: GpuDevice
    inflight: int = 0
    healthy: bool = True


class GpuScheduler:
    """Least-loaded multi-GPU router with health and in-flight accounting."""

    def __init__(self) -> None:
        self._devices: dict[int, _DeviceState] = {}
        self._leases: dict[str, int] = {}
        self._lock = threading.Lock()

    def register(self, device: GpuDevice) -> None:
        with self._lock:
            if device.device_id in self._devices:
                raise ValueError(f"device {device.device_id} is already registered")
            self._devices[device.device_id] = _DeviceState(device)

    def mark_unhealthy(self, device_id: int) -> None:
        with self._lock:
            self._state(device_id).healthy = False

    def mark_healthy(self, device_id: int) -> None:
        with self._lock:
            self._state(device_id).healthy = True

    def acquire(self, preferred_device_id: int | None = None) -> GpuLease | None:
        with self._lock:
            preferred = None
            candidates: list[_DeviceState] = []
            for state in self._devices.values():
                if not state.healthy or state.inflight >= state.device.max_inflight:
                    continue
                if preferred_device_id is not None and state.device.device_id == preferred_device_id:
                    preferred = state
                candidates.append(state)
            selected = preferred or (min(candidates, key=lambda item: (item.inflight, item.device.device_id)) if candidates else None)
            if selected is None:
                return None
            selected.inflight += 1
            lease = GpuLease(selected.device, uuid.uuid4().hex)
            self._leases[lease.lease_id] = selected.device.device_id
            return lease

    def release(self, lease: GpuLease) -> bool:
        with self._lock:
            device_id = self._leases.pop(lease.lease_id, None)
            if device_id is None:
                return False
            state = self._devices.get(device_id)
            if state is None:
                return False
            state.inflight = max(0, state.inflight - 1)
            return True

    def inflight(self, device_id: int) -> int:
        with self._lock:
            return self._state(device_id).inflight

    def devices(self) -> tuple[GpuDevice, ...]:
        with self._lock:
            return tuple(
                sorted(
                    (state.device for state in self._devices.values()),
                    key=lambda device: device.device_id,
                )
            )

    def _state(self, device_id: int) -> _DeviceState:
        try:
            return self._devices[device_id]
        except KeyError as exc:
            raise ValueError(f"device {device_id} is not registered") from exc
