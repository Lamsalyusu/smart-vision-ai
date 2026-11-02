from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence, Callable, Dict, Any, List
from enum import Enum
from abc import ABC, abstractmethod
from collections import deque
import time
import threading
import queue
import uuid

try:
    import tensorrt as trt
except ImportError:
    trt = None

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    cuda = None


class InferenceBackend(Enum):
    TENSORRT = "tensorrt"
    ONNX_RUNTIME = "onnxruntime"
    OPENVINO = "openvino"
    PYTORCH = "pytorch"
    TFLITE = "tflite"
    TRITON = "triton"


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    request_id: str
    inputs: Dict[str, np.ndarray]
    priority: int = 0
    timestamp: float = 0.0
    callback: Optional[Callable] = None
    timeout: float = 30.0


@dataclass(frozen=True, slots=True)
class InferenceResponse:
    request_id: str
    outputs: Dict[str, np.ndarray]
    latency_ms: float
    success: bool = True
    error: Optional[str] = None


class InferenceEngine(ABC):
    @abstractmethod
    def load_model(self, model_path: str, **kwargs) -> bool:
        pass

    @abstractmethod
    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def infer_batch(self, inputs_list: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
        pass

    @abstractmethod
    def get_input_names(self) -> List[str]:
        pass

    @abstractmethod
    def get_output_names(self) -> List[str]:
        pass


class TensorRTEngine(InferenceEngine):
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.engine = None
        self.context = None
        self.input_names = []
        self.output_names = []
        self.bindings = []
        self.stream = None
        self._initialized = False

    def _initialize(self):
        try:
            if trt is None or cuda is None:
                raise ImportError("tensorrt or pycuda not available")

            self.logger = trt.Logger(trt.Logger.WARNING)
            self.runtime = trt.Runtime(self.logger)
            self.stream = cuda.Stream()
            self._initialized = True
        except ImportError:
            self._initialized = False

    def load_model(self, model_path: str, **kwargs) -> bool:
        if not self._initialized:
            self._initialize()
        if not self._initialized:
            return False

        try:
            with open(model_path, 'rb') as f:
                engine_data = f.read()

            self.engine = self.runtime.deserialize_cuda_engine(engine_data)
            self.context = self.engine.create_execution_context()

            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                is_input = self.engine.binding_is_input(i)
                shape = self.engine.get_binding_shape(i)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))

                if is_input:
                    self.input_names.append(name)
                else:
                    self.output_names.append(name)

                binding_ptr = cuda.mem_alloc(np.prod(shape) * np.dtype(dtype).itemsize)
                self.bindings.append(int(binding_ptr))

            return True
        except Exception:
            return False

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for i, name in enumerate(self.input_names):
            if name in inputs:
                cuda.memcpy_htod_async(self.bindings[i], inputs[name].astype(np.float32), self.stream)

        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        outputs = {}
        for i, name in enumerate(self.output_names):
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            output = np.empty(shape, dtype=dtype)
            cuda.memcpy_dtoh_async(output, self.bindings[i], self.stream)
            outputs[name] = output

        self.stream.synchronize()
        return outputs

    def infer_batch(self, inputs_list: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
        return [self.infer(inputs) for inputs in inputs_list]

    def get_input_names(self) -> List[str]:
        return self.input_names

    def get_output_names(self) -> List[str]:
        return self.output_names


class ONNXRuntimeEngine(InferenceEngine):
    def __init__(
        self,
        providers: List[str] = None,
        session_options: Dict = None,
    ):
        self.providers = providers or ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session_options = session_options or {}
        self.session = None
        self._initialized = False

    def _initialize(self):
        try:
            import onnxruntime as ort
            self.ort = ort
            self._initialized = True
        except ImportError:
            self._initialized = False

    def load_model(self, model_path: str, **kwargs) -> bool:
        if not self._initialized:
            self._initialize()
        if not self._initialized:
            return False

        try:
            sess_options = self.ort.SessionOptions()
            for k, v in self.session_options.items():
                setattr(sess_options, k, v)

            self.session = self.ort.InferenceSession(model_path, sess_options, providers=self.providers)
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]
            return True
        except Exception:
            return False

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        outputs = self.session.run(self.output_names, inputs)
        return dict(zip(self.output_names, outputs))

    def infer_batch(self, inputs_list: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
        batch_inputs = {}
        for name in self.input_names:
            batch_inputs[name] = np.stack([inp[name] for inp in inputs_list])

        outputs = self.session.run(self.output_names, batch_inputs)
        batch_outputs = dict(zip(self.output_names, outputs))

        results = []
        batch_size = len(inputs_list)
        for i in range(batch_size):
            result = {name: batch_outputs[name][i] for name in self.output_names}
            results.append(result)
        return results

    def get_input_names(self) -> List[str]:
        return self.input_names

    def get_output_names(self) -> List[str]:
        return self.output_names


class PyTorchEngine(InferenceEngine):
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self._initialized = False

    def _initialize(self):
        try:
            import torch
            self.torch = torch
            self._initialized = True
        except ImportError:
            self._initialized = False

    def load_model(self, model_path: str, **kwargs) -> bool:
        if not self._initialized:
            self._initialize()
        if not self._initialized:
            return False

        try:
            import torch
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()

            dummy_input = kwargs.get('dummy_input')
            if dummy_input is not None:
                with torch.no_grad():
                    _ = self.model(dummy_input)

            return True
        except Exception:
            try:
                self.model = torch.load(model_path, map_location=self.device)
                self.model.eval()
                return True
            except Exception:
                return False

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        import torch

        tensor_inputs = {}
        for name, arr in inputs.items():
            tensor_inputs[name] = torch.from_numpy(arr).to(self.device)

        with torch.no_grad():
            outputs = self.model(**tensor_inputs)

        if isinstance(outputs, tuple):
            outputs = {f"output_{i}": out for i, out in enumerate(outputs)}
        elif isinstance(outputs, torch.Tensor):
            outputs = {"output": outputs}
        elif isinstance(outputs, dict):
            pass
        else:
            outputs = {"output": outputs}

        return {name: out.cpu().numpy() for name, out in outputs.items()}

    def infer_batch(self, inputs_list: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
        import torch

        batch_inputs = {}
        for name in inputs_list[0].keys():
            batch_inputs[name] = torch.from_numpy(np.stack([inp[name] for inp in inputs_list])).to(self.device)

        with torch.no_grad():
            outputs = self.model(**batch_inputs)

        if isinstance(outputs, tuple):
            outputs = {f"output_{i}": out for i, out in enumerate(outputs)}
        elif isinstance(outputs, torch.Tensor):
            outputs = {"output": outputs}

        results = []
        batch_size = len(inputs_list)
        for i in range(batch_size):
            result = {name: out[i].cpu().numpy() for name, out in outputs.items()}
            results.append(result)
        return results

    def get_input_names(self) -> List[str]:
        return list(self.model._forward_signature.parameters.keys()) if hasattr(self.model, '_forward_signature') else []

    def get_output_names(self) -> List[str]:
        return []


class InferenceServer:
    def __init__(
        self,
        engine: InferenceEngine,
        max_batch_size: int = 32,
        max_queue_size: int = 1000,
        num_workers: int = 4,
    ):
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.num_workers = num_workers
        self.request_queue = queue.PriorityQueue(maxsize=max_queue_size)
        self.response_futures: Dict[str, queue.Queue] = {}
        self.workers: List[threading.Thread] = []
        self.running = False
        self.stats = {
            'total_requests': 0,
            'total_latency': 0.0,
            'errors': 0,
        }

    def start(self):
        self.running = True
        for _ in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self.workers.append(worker)

    def stop(self):
        self.running = False
        for _ in range(self.num_workers):
            self.request_queue.put((999, None))
        for worker in self.workers:
            worker.join(timeout=5)

    def infer_async(self, inputs: Dict[str, np.ndarray], priority: int = 0, timeout: float = 30.0) -> str:
        request_id = str(uuid.uuid4())[:8]
        response_queue = queue.Queue(maxsize=1)
        self.response_futures[request_id] = response_queue

        request = InferenceRequest(
            request_id=request_id,
            inputs=inputs,
            priority=priority,
            timestamp=time.time(),
            timeout=timeout,
        )

        self.request_queue.put((-priority, request))
        return request_id

    def get_result(self, request_id: str, timeout: float = 30.0) -> Optional[InferenceResponse]:
        if request_id not in self.response_futures:
            return None

        try:
            return self.response_futures[request_id].get(timeout=timeout)
        except queue.Empty:
            return InferenceResponse(
                request_id=request_id,
                outputs={},
                latency_ms=0,
                success=False,
                error="Timeout",
            )
        finally:
            self.response_futures.pop(request_id, None)

    def _worker_loop(self):
        while self.running:
            try:
                priority, request = self.request_queue.get(timeout=1.0)
                if request is None:
                    break

                start = time.time()
                try:
                    outputs = self.engine.infer(request.inputs)
                    latency = (time.time() - start) * 1000

                    response = InferenceResponse(
                        request_id=request.request_id,
                        outputs=outputs,
                        latency_ms=latency,
                        success=True,
                    )
                    self.stats['total_requests'] += 1
                    self.stats['total_latency'] += latency
                except Exception as e:
                    response = InferenceResponse(
                        request_id=request.request_id,
                        outputs={},
                        latency_ms=(time.time() - start) * 1000,
                        success=False,
                        error=str(e),
                    )
                    self.stats['errors'] += 1

                if request.request_id in self.response_futures:
                    self.response_futures[request.request_id].put(response)

            except queue.Empty:
                continue
            except Exception:
                continue


class BatchInferenceServer:
    def __init__(
        self,
        engine: InferenceEngine,
        max_batch_size: int = 32,
        batch_timeout_ms: float = 10.0,
        num_workers: int = 2,
    ):
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.num_workers = num_workers
        self.request_queue = queue.Queue()
        self.response_futures: Dict[str, queue.Queue] = {}
        self.workers: List[threading.Thread] = []
        self.running = False

    def start(self):
        self.running = True
        for _ in range(self.num_workers):
            worker = threading.Thread(target=self._batch_worker_loop, daemon=True)
            worker.start()
            self.workers.append(worker)

    def stop(self):
        self.running = False
        for _ in range(self.num_workers):
            self.request_queue.put(None)
        for worker in self.workers:
            worker.join(timeout=5)

    def infer_async(self, inputs: Dict[str, np.ndarray], timeout: float = 30.0) -> str:
        request_id = str(uuid.uuid4())[:8]
        response_queue = queue.Queue(maxsize=1)
        self.response_futures[request_id] = response_queue

        request = InferenceRequest(
            request_id=request_id,
            inputs=inputs,
            timestamp=time.time(),
            timeout=timeout,
        )

        self.request_queue.put(request)
        return request_id

    def get_result(self, request_id: str, timeout: float = 30.0) -> Optional[InferenceResponse]:
        if request_id not in self.response_futures:
            return None
        try:
            return self.response_futures[request_id].get(timeout=timeout)
        except queue.Empty:
            return InferenceResponse(
                request_id=request_id,
                outputs={},
                latency_ms=0,
                success=False,
                error="Timeout",
            )
        finally:
            self.response_futures.pop(request_id, None)

    def _batch_worker_loop(self):
        while self.running:
            batch = []
            start_time = time.time()

            while len(batch) < self.max_batch_size:
                try:
                    remaining = self.batch_timeout_ms / 1000 - (time.time() - start_time)
                    if remaining <= 0:
                        break
                    request = self.request_queue.get(timeout=remaining)
                    if request is None:
                        return
                    batch.append(request)
                except queue.Empty:
                    break

            if not batch:
                continue

            start = time.time()
            try:
                inputs_list = [req.inputs for req in batch]
                outputs_list = self.engine.infer_batch(inputs_list)
                latency = (time.time() - start) * 1000

                for req, outputs in zip(batch, outputs_list):
                    response = InferenceResponse(
                        request_id=req.request_id,
                        outputs=outputs,
                        latency_ms=latency,
                        success=True,
                    )
                    if req.request_id in self.response_futures:
                        self.response_futures[req.request_id].put(response)

            except Exception as e:
                for req in batch:
                    response = InferenceResponse(
                        request_id=req.request_id,
                        outputs={},
                        latency_ms=(time.time() - start) * 1000,
                        success=False,
                        error=str(e),
                    )
                    if req.request_id in self.response_futures:
                        self.response_futures[req.request_id].put(response)


class ModelManager:
    def __init__(self):
        self.models: Dict[str, InferenceEngine] = {}
        self.servers: Dict[str, InferenceServer] = {}

    def register_model(
        self,
        name: str,
        engine: InferenceEngine,
        model_path: str,
        **kwargs,
    ) -> bool:
        if engine.load_model(model_path, **kwargs):
            self.models[name] = engine
            return True
        return False

    def create_server(
        self,
        model_name: str,
        server_type: str = "batch",
        **kwargs,
    ) -> Optional[InferenceServer]:
        if model_name not in self.models:
            return None

        engine = self.models[model_name]
        if server_type == "batch":
            server = BatchInferenceServer(engine, **kwargs)
        else:
            server = InferenceServer(engine, **kwargs)

        server.start()
        self.servers[model_name] = server
        return server

    def get_server(self, model_name: str) -> Optional[InferenceServer]:
        return self.servers.get(model_name)

    def shutdown_all(self):
        for server in self.servers.values():
            server.stop()
        self.servers.clear()


class InferencePipeline:
    def __init__(
        self,
        stages: List[InferenceEngine],
        preprocess: Optional[Callable] = None,
        postprocess: Optional[Callable] = None,
    ):
        self.stages = stages
        self.preprocess = preprocess
        self.postprocess = postprocess

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        current = inputs

        if self.preprocess:
            current = self.preprocess(current)

        for stage in self.stages:
            current = stage.infer(current)

        if self.postprocess:
            current = self.postprocess(current)

        return current

    def run_batch(self, inputs_list: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
        current = inputs_list

        if self.preprocess:
            current = [self.preprocess(inp) for inp in current]

        for stage in self.stages:
            current = stage.infer_batch(current)

        if self.postprocess:
            current = [self.postprocess(inp) for inp in current]

        return current


def create_engine(backend: InferenceBackend, **kwargs) -> InferenceEngine:
    if backend == InferenceBackend.TENSORRT:
        return TensorRTEngine(**kwargs)
    elif backend == InferenceBackend.ONNX_RUNTIME:
        return ONNXRuntimeEngine(**kwargs)
    elif backend == InferenceBackend.PYTORCH:
        return PyTorchEngine(**kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
