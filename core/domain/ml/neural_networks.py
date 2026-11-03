from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Set
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict


class LayerType(Enum):
    DENSE = "dense"
    CONV2D = "conv2d"
    CONV1D = "conv1d"
    MAX_POOL2D = "max_pool2d"
    AVG_POOL2D = "avg_pool2d"
    DROPOUT = "dropout"
    BATCH_NORM = "batch_norm"
    LAYER_NORM = "layer_norm"
    ACTIVATION = "activation"
    FLATTEN = "flatten"
    RESHAPE = "reshape"
    CONCATENATE = "concatenate"
    ADD = "add"
    MULTIPLY = "multiply"
    LSTM = "lstm"
    GRU = "gru"
    RNN = "rnn"
    ATTENTION = "attention"
    MULTI_HEAD_ATTENTION = "multi_head_attention"
    TRANSFORMER_BLOCK = "transformer_block"
    EMBEDDING = "embedding"
    POSITIONAL_ENCODING = "positional_encoding"


class ActivationType(Enum):
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    SELU = "selu"
    GELU = "gelu"
    SWISH = "swish"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"
    LOG_SOFTMAX = "log_softmax"
    LINEAR = "linear"
    MISH = "mish"
    HARD_SWISH = "hard_swish"
    SOFTPLUS = "softplus"


class InitializerType(Enum):
    ZEROS = "zeros"
    ONES = "ones"
    CONSTANT = "constant"
    RANDOM_NORMAL = "random_normal"
    RANDOM_UNIFORM = "random_uniform"
    TRUNCATED_NORMAL = "truncated_normal"
    GLOROT_UNIFORM = "glorot_uniform"
    GLOROT_NORMAL = "glorot_normal"
    HE_UNIFORM = "he_uniform"
    HE_NORMAL = "he_normal"
    LECUN_UNIFORM = "lecun_uniform"
    LECUN_NORMAL = "lecun_normal"
    ORTHOGONAL = "orthogonal"
    IDENTITY = "identity"


class OptimizerType(Enum):
    SGD = "sgd"
    MOMENTUM = "momentum"
    NESTEROV = "nesterov"
    ADAGRAD = "adagrad"
    RMSPROP = "rmsprop"
    ADAM = "adam"
    ADAMW = "adamw"
    NADAM = "nadam"
    RADAM = "radam"
    ADAMAX = "adamax"
    ADABOUND = "adabound"


class LossType(Enum):
    MSE = "mse"
    MAE = "mae"
    HUBER = "huber"
    CROSS_ENTROPY = "cross_entropy"
    BINARY_CROSS_ENTROPY = "binary_cross_entropy"
    CATEGORICAL_CROSS_ENTROPY = "categorical_cross_entropy"
    SPARSE_CATEGORICAL_CROSS_ENTROPY = "sparse_categorical_cross_entropy"
    HINGE = "hinge"
    SQUARED_HINGE = "squared_hinge"
    KLD = "kld"
    POISSON = "poisson"
    COSINE_SIMILARITY = "cosine_similarity"
    CONTRASTIVE = "contrastive"
    TRIPLET = "triplet"
    FOCAL = "focal"
    DICE = "dice"
    IOU = "iou"


@dataclass(frozen=True, slots=True)
class LayerConfig:
    layer_type: LayerType
    units: Optional[int] = None
    filters: Optional[int] = None
    kernel_size: Optional[Tuple[int, int]] = None
    strides: Optional[Tuple[int, int]] = None
    padding: str = "valid"
    activation: ActivationType = ActivationType.LINEAR
    use_bias: bool = True
    kernel_initializer: InitializerType = InitializerType.GLOROT_UNIFORM
    bias_initializer: InitializerType = InitializerType.ZEROS
    dropout_rate: float = 0.0
    batch_norm: bool = False
    dilation_rate: Tuple[int, int] = (1, 1)
    groups: int = 1
    return_sequences: bool = False
    stateful: bool = False
    num_heads: int = 8
    key_dim: int = 64
    dropout: float = 0.1


@dataclass(frozen=True, slots=True)
class ModelConfig:
    input_shape: Tuple[int, ...]
    layers: List[LayerConfig]
    optimizer: OptimizerType = OptimizerType.ADAM
    learning_rate: float = 0.001
    loss: LossType = LossType.MSE
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    callbacks: List[str] = field(default_factory=list)


class Initializer:
    @staticmethod
    def get_initializer(initializer_type: InitializerType, shape: Tuple[int, ...], **kwargs) -> np.ndarray:
        if initializer_type == InitializerType.ZEROS:
            return np.zeros(shape)
        elif initializer_type == InitializerType.ONES:
            return np.ones(shape)
        elif initializer_type == InitializerType.CONSTANT:
            return np.full(shape, kwargs.get('value', 0.0))
        elif initializer_type == InitializerType.RANDOM_NORMAL:
            return np.random.normal(kwargs.get('mean', 0.0), kwargs.get('stddev', 1.0), shape)
        elif initializer_type == InitializerType.RANDOM_UNIFORM:
            minval = kwargs.get('minval', -0.05)
            maxval = kwargs.get('maxval', 0.05)
            return np.random.uniform(minval, maxval, shape)
        elif initializer_type == InitializerType.TRUNCATED_NORMAL:
            stddev = kwargs.get('stddev', 1.0)
            samples = np.random.normal(0, stddev, shape * 2)
            samples = samples[np.abs(samples) < 2 * stddev]
            return samples[:np.prod(shape)].reshape(shape)
        elif initializer_type == InitializerType.GLOROT_UNIFORM:
            fan_in, fan_out = shape[-2] if len(shape) >= 2 else shape[0], shape[-1] if len(shape) >= 2 else shape[0]
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            return np.random.uniform(-limit, limit, shape)
        elif initializer_type == InitializerType.GLOROT_NORMAL:
            fan_in, fan_out = shape[-2] if len(shape) >= 2 else shape[0], shape[-1] if len(shape) >= 2 else shape[0]
            stddev = math.sqrt(2.0 / (fan_in + fan_out))
            return np.random.normal(0, stddev, shape)
        elif initializer_type == InitializerType.HE_UNIFORM:
            fan_in = shape[-2] if len(shape) >= 2 else shape[0]
            limit = math.sqrt(6.0 / fan_in)
            return np.random.uniform(-limit, limit, shape)
        elif initializer_type == InitializerType.HE_NORMAL:
            fan_in = shape[-2] if len(shape) >= 2 else shape[0]
            stddev = math.sqrt(2.0 / fan_in)
            return np.random.normal(0, stddev, shape)
        elif initializer_type == InitializerType.LECUN_UNIFORM:
            fan_in = shape[-2] if len(shape) >= 2 else shape[0]
            limit = math.sqrt(3.0 / fan_in)
            return np.random.uniform(-limit, limit, shape)
        elif initializer_type == InitializerType.LECUN_NORMAL:
            fan_in = shape[-2] if len(shape) >= 2 else shape[0]
            stddev = math.sqrt(1.0 / fan_in)
            return np.random.normal(0, stddev, shape)
        elif initializer_type == InitializerType.ORTHOGONAL:
            return Initializer._orthogonal_initializer(shape)
        elif initializer_type == InitializerType.IDENTITY:
            return np.eye(*shape) if len(shape) == 2 else np.eye(shape[0])

        return np.random.randn(*shape) * 0.02

    @staticmethod
    def _orthogonal_initializer(shape: Tuple[int, ...]) -> np.ndarray:
        if len(shape) < 2:
            return np.eye(shape[0])

        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0, 1, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        return q.reshape(shape)


class Activation:
    @staticmethod
    def forward(x: np.ndarray, activation: ActivationType) -> np.ndarray:
        if activation == ActivationType.RELU:
            return np.maximum(x, 0)
        elif activation == ActivationType.LEAKY_RELU:
            return np.where(x > 0, x, 0.01 * x)
        elif activation == ActivationType.ELU:
            return np.where(x > 0, x, np.exp(x) - 1)
        elif activation == ActivationType.SELU:
            scale = 1.0507
            alpha = 1.67326
            return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))
        elif activation == ActivationType.GELU:
            return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
        elif activation == ActivationType.SWISH:
            return x * (1 / (1 + np.exp(-x)))
        elif activation == ActivationType.SIGMOID:
            return 1 / (1 + np.exp(-x))
        elif activation == ActivationType.TANH:
            return np.tanh(x)
        elif activation == ActivationType.SOFTMAX:
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        elif activation == ActivationType.LOG_SOFTMAX:
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            softmax = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
            return np.log(softmax + 1e-8)
        elif activation == ActivationType.MISH:
            return x * np.tanh(np.log1p(np.exp(x)))
        elif activation == ActivationType.HARD_SWISH:
            return x * np.maximum(0, np.minimum(6, x + 3)) / 6
        elif activation == ActivationType.SOFTPLUS:
            return np.log1p(np.exp(x))
        elif activation == ActivationType.LINEAR:
            return x

        return x

    @staticmethod
    def backward(x: np.ndarray, activation: ActivationType) -> np.ndarray:
        if activation == ActivationType.RELU:
            return (x > 0).astype(x.dtype)
        elif activation == ActivationType.LEAKY_RELU:
            return np.where(x > 0, 1, 0.01)
        elif activation == ActivationType.ELU:
            return np.where(x > 0, 1, np.exp(x))
        elif activation == ActivationType.SELU:
            scale = 1.0507
            alpha = 1.67326
            return scale * np.where(x > 0, 1, alpha * np.exp(x))
        elif activation == ActivationType.GELU:
            tanh_arg = math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)
            tanh_val = np.tanh(tanh_arg)
            return 0.5 * (1 + tanh_val) + 0.5 * x * (1 - tanh_val**2) * math.sqrt(2 / math.pi) * (1 + 3 * 0.044715 * x**2)
        elif activation == ActivationType.SWISH:
            sig = 1 / (1 + np.exp(-x))
            return sig + x * sig * (1 - sig)
        elif activation == ActivationType.SIGMOID:
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        elif activation == ActivationType.TANH:
            return 1 - np.tanh(x)**2
        elif activation == ActivationType.MISH:
            softplus = np.log1p(np.exp(x))
            tanh_sp = np.tanh(softplus)
            return tanh_sp + x * (1 - tanh_sp**2) * (1 / (1 + np.exp(-x)))
        elif activation == ActivationType.HARD_SWISH:
            return np.where(x <= -3, 0, np.where(x >= 3, 1, (2*x + 3) / 6))
        elif activation == ActivationType.SOFTPLUS:
            return 1 / (1 + np.exp(-x))
        elif activation == ActivationType.LINEAR:
            return np.ones_like(x)

        return np.ones_like(x)


class Layer:
    def __init__(self, config: LayerConfig):
        self.config = config
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None
        self.running_mean: Optional[np.ndarray] = None
        self.running_var: Optional[np.ndarray] = None
        self.trainable = True

    def build(self, input_shape: Tuple[int, ...]):
        if self.config.layer_type == LayerType.DENSE:
            self._build_dense(input_shape)
        elif self.config.layer_type in [LayerType.CONV2D, LayerType.CONV1D]:
            self._build_conv(input_shape)
        elif self.config.layer_type == LayerType.BATCH_NORM:
            self._build_batch_norm(input_shape)
        elif self.config.layer_type == LayerType.LAYER_NORM:
            self._build_layer_norm(input_shape)

    def _build_dense(self, input_shape: Tuple[int, ...]):
        input_dim = input_shape[-1]
        output_dim = self.config.units

        self.weights = Initializer.get_initializer(
            self.config.kernel_initializer, (input_dim, output_dim)
        )
        if self.config.use_bias:
            self.bias = Initializer.get_initializer(
                self.config.bias_initializer, (output_dim,)
            )

    def _build_conv(self, input_shape: Tuple[int, ...]):
        in_channels = input_shape[-1] if len(input_shape) > 2 else 1
        out_channels = self.config.filters
        kh, kw = self.config.kernel_size

        self.weights = Initializer.get_initializer(
            self.config.kernel_initializer, (kh, kw, in_channels, out_channels)
        )
        if self.config.use_bias:
            self.bias = Initializer.get_initializer(
                self.config.bias_initializer, (out_channels,)
            )

    def _build_batch_norm(self, input_shape: Tuple[int, ...]):
        num_features = input_shape[-1]
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.weights = np.ones(num_features)
        self.bias = np.zeros(num_features)

    def _build_layer_norm(self, input_shape: Tuple[int, ...]):
        num_features = input_shape[-1]
        self.weights = np.ones(num_features)
        self.bias = np.zeros(num_features)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if self.config.layer_type == LayerType.DENSE:
            return self._forward_dense(x)
        elif self.config.layer_type == LayerType.CONV2D:
            return self._forward_conv2d(x)
        elif self.config.layer_type == LayerType.MAX_POOL2D:
            return self._forward_max_pool2d(x)
        elif self.config.layer_type == LayerType.AVG_POOL2D:
            return self._forward_avg_pool2d(x)
        elif self.config.layer_type == LayerType.DROPOUT:
            return self._forward_dropout(x, training)
        elif self.config.layer_type == LayerType.BATCH_NORM:
            return self._forward_batch_norm(x, training)
        elif self.config.layer_type == LayerType.LAYER_NORM:
            return self._forward_layer_norm(x)
        elif self.config.layer_type == LayerType.ACTIVATION:
            return Activation.forward(x, self.config.activation)
        elif self.config.layer_type == LayerType.FLATTEN:
            return self._forward_flatten(x)
        elif self.config.layer_type == LayerType.ADD:
            return x
        elif self.config.layer_type == LayerType.CONCATENATE:
            return x

        return x

    def _forward_dense(self, x: np.ndarray) -> np.ndarray:
        output = x @ self.weights
        if self.bias is not None:
            output += self.bias
        if self.config.activation != ActivationType.LINEAR:
            output = Activation.forward(output, self.config.activation)
        return output

    def _forward_conv2d(self, x: np.ndarray) -> np.ndarray:
        batch, height, width, in_channels = x.shape
        kh, kw = self.config.kernel_size
        sh, sw = self.config.strides

        pad_h = kh // 2 if self.config.padding == "same" else 0
        pad_w = kw // 2 if self.config.padding == "same" else 0

        if pad_h > 0 or pad_w > 0:
            x = np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')

        out_height = (height + 2 * pad_h - kh) // sh + 1
        out_width = (width + 2 * pad_w - kw) // sw + 1

        out_channels = self.config.filters
        output = np.zeros((batch, out_height, out_width, out_channels))

        for b in range(batch):
            for oc in range(out_channels):
                for ic in range(x.shape[-1]):
                    for oh in range(out_height):
                        for ow in range(out_width):
                            h_start = oh * sh
                            w_start = ow * sw
                            window = x[b, h_start:h_start+kh, w_start:w_start+kw, ic]
                            kernel = self.weights[:, :, ic, oc]
                            output[b, oh, ow, oc] += np.sum(window * kernel)

        if self.config.use_bias:
            output += self.bias

        if self.config.activation != ActivationType.LINEAR:
            output = Activation.forward(output, self.config.activation)

        return output

    def _forward_max_pool2d(self, x: np.ndarray) -> np.ndarray:
        batch, height, width, channels = x.shape
        kh, kw = self.config.kernel_size
        sh, sw = self.config.strides

        out_height = (height - kh) // sh + 1
        out_width = (width - kw) // sw + 1

        output = np.zeros((batch, out_height, out_width, channels))

        for b in range(batch):
            for c in range(channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        h_start = oh * sh
                        w_start = ow * sw
                        window = x[b, h_start:h_start+kh, w_start:w_start+kw, c]
                        output[b, oh, ow, c] = np.max(window)

        return output

    def _forward_avg_pool2d(self, x: np.ndarray) -> np.ndarray:
        batch, height, width, channels = x.shape
        kh, kw = self.config.kernel_size
        sh, sw = self.config.strides

        out_height = (height - kh) // sh + 1
        out_width = (width - kw) // sw + 1

        output = np.zeros((batch, out_height, out_width, channels))

        for b in range(batch):
            for c in range(channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        h_start = oh * sh
                        w_start = ow * sw
                        window = x[b, h_start:h_start+kh, w_start:w_start+kw, c]
                        output[b, oh, ow, c] = np.mean(window)

        return output

    def _forward_dropout(self, x: np.ndarray, training: bool) -> np.ndarray:
        if not training or self.config.dropout_rate == 0:
            return x

        mask = np.random.binomial(1, 1 - self.config.dropout_rate, x.shape)
        return x * mask / (1 - self.config.dropout_rate)

    def _forward_batch_norm(self, x: np.ndarray, training: bool) -> np.ndarray:
        if x.ndim == 4:
            axis = (0, 1, 2)
        else:
            axis = 0

        if training:
            mean = np.mean(x, axis=axis, keepdims=True)
            var = np.var(x, axis=axis, keepdims=True)

            self.running_mean = 0.9 * self.running_mean + 0.1 * np.mean(mean, axis=axis, keepdims=False)
            self.running_var = 0.9 * self.running_var + 0.1 * np.mean(var, axis=axis, keepdims=False)
        else:
            mean = self.running_mean
            var = self.running_var

        normalized = (x - mean) / np.sqrt(var + 1e-5)
        return normalized * self.weights + self.bias

    def _forward_layer_norm(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + 1e-5)
        return normalized * self.weights + self.bias

    def _forward_flatten(self, x: np.ndarray) -> np.ndarray:
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)


class Optimizer:
    def __init__(self, optimizer_type: OptimizerType, learning_rate: float = 0.001, **kwargs):
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.params = {}
        self.velocities = {}
        self.cache = {}
        self.t = 0

        for k, v in kwargs.items():
            setattr(self, k, v)

    def update(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        self.t += 1

        if self.optimizer_type == OptimizerType.SGD:
            return self._sgd(params, grads)
        elif self.optimizer_type == OptimizerType.MOMENTUM:
            return self._momentum(params, grads)
        elif self.optimizer_type == OptimizerType.NESTEROV:
            return self._nesterov(params, grads)
        elif self.optimizer_type == OptimizerType.ADAGRAD:
            return self._adagrad(params, grads)
        elif self.optimizer_type == OptimizerType.RMSPROP:
            return self._rmsprop(params, grads)
        elif self.optimizer_type == OptimizerType.ADAM:
            return self._adam(params, grads)
        elif self.optimizer_type == OptimizerType.ADAMW:
            return self._adamw(params, grads)
        elif self.optimizer_type == OptimizerType.NADAM:
            return self._nadam(params, grads)

        return params

    def _sgd(self, params: Dict, grads: Dict) -> Dict:
        for key in params:
            params[key] -= self.learning_rate * grads[key]
        return params

    def _momentum(self, params: Dict, grads: Dict, momentum: float = 0.9) -> Dict:
        for key in params:
            if key not in self.velocities:
                self.velocities[key] = np.zeros_like(params[key])
            self.velocities[key] = momentum * self.velocities[key] - self.learning_rate * grads[key]
            params[key] += self.velocities[key]
        return params

    def _nesterov(self, params: Dict, grads: Dict, momentum: float = 0.9) -> Dict:
        for key in params:
            if key not in self.velocities:
                self.velocities[key] = np.zeros_like(params[key])
            v_prev = self.velocities[key].copy()
            self.velocities[key] = momentum * self.velocities[key] - self.learning_rate * grads[key]
            params[key] += -momentum * v_prev + (1 + momentum) * self.velocities[key]
        return params

    def _adagrad(self, params: Dict, grads: Dict, epsilon: float = 1e-8) -> Dict:
        for key in params:
            if key not in self.cache:
                self.cache[key] = np.zeros_like(params[key])
            self.cache[key] += grads[key]**2
            params[key] -= self.learning_rate * grads[key] / (np.sqrt(self.cache[key]) + epsilon)
        return params

    def _rmsprop(self, params: Dict, grads: Dict, decay: float = 0.9, epsilon: float = 1e-8) -> Dict:
        for key in params:
            if key not in self.cache:
                self.cache[key] = np.zeros_like(params[key])
            self.cache[key] = decay * self.cache[key] + (1 - decay) * grads[key]**2
            params[key] -= self.learning_rate * grads[key] / (np.sqrt(self.cache[key]) + epsilon)
        return params

    def _adam(self, params: Dict, grads: Dict, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> Dict:
        for key in params:
            if key not in self.cache:
                self.cache[key] = np.zeros_like(params[key])
                self.velocities[key] = np.zeros_like(params[key])

            self.velocities[key] = beta1 * self.velocities[key] + (1 - beta1) * grads[key]
            self.cache[key] = beta2 * self.cache[key] + (1 - beta2) * grads[key]**2

            m_hat = self.velocities[key] / (1 - beta1**self.t)
            v_hat = self.cache[key] / (1 - beta2**self.t)

            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        return params

    def _adamw(self, params: Dict, grads: Dict, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8, weight_decay: float = 0.01) -> Dict:
        for key in params:
            if key not in self.cache:
                self.cache[key] = np.zeros_like(params[key])
                self.velocities[key] = np.zeros_like(params[key])

            self.velocities[key] = beta1 * self.velocities[key] + (1 - beta1) * grads[key]
            self.cache[key] = beta2 * self.cache[key] + (1 - beta2) * grads[key]**2

            m_hat = self.velocities[key] / (1 - beta1**self.t)
            v_hat = self.cache[key] / (1 - beta2**self.t)

            params[key] -= self.learning_rate * (m_hat / (np.sqrt(v_hat) + epsilon) + weight_decay * params[key])
        return params

    def _nadam(self, params: Dict, grads: Dict, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> Dict:
        for key in params:
            if key not in self.cache:
                self.cache[key] = np.zeros_like(params[key])
                self.velocities[key] = np.zeros_like(params[key])

            self.velocities[key] = beta1 * self.velocities[key] + (1 - beta1) * grads[key]
            self.cache[key] = beta2 * self.cache[key] + (1 - beta2) * grads[key]**2

            m_hat = self.velocities[key] / (1 - beta1**self.t)
            v_hat = self.cache[key] / (1 - beta2**self.t)

            params[key] -= self.learning_rate * (beta1 * m_hat + (1 - beta1) * grads[key] / (1 - beta1**self.t)) / (np.sqrt(v_hat) + epsilon)
        return params


class Loss:
    @staticmethod
    def forward(y_true: np.ndarray, y_pred: np.ndarray, loss_type: LossType) -> float:
        if loss_type == LossType.MSE:
            return np.mean((y_true - y_pred)**2)
        elif loss_type == LossType.MAE:
            return np.mean(np.abs(y_true - y_pred))
        elif loss_type == LossType.HUBER:
            delta = 1.0
            error = y_true - y_pred
            return np.mean(np.where(np.abs(error) <= delta, 0.5 * error**2, delta * (np.abs(error) - 0.5 * delta)))
        elif loss_type == LossType.CROSS_ENTROPY:
            y_pred_clipped = np.clip(y_pred, 1e-8, 1 - 1e-8)
            return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=-1))
        elif loss_type == LossType.BINARY_CROSS_ENTROPY:
            y_pred_clipped = np.clip(y_pred, 1e-8, 1 - 1e-8)
            return -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        elif loss_type == LossType.CATEGORICAL_CROSS_ENTROPY:
            y_pred_clipped = np.clip(y_pred, 1e-8, 1 - 1e-8)
            return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=-1))
        elif loss_type == LossType.SPARSE_CATEGORICAL_CROSS_ENTROPY:
            y_pred_clipped = np.clip(y_pred, 1e-8, 1 - 1e-8)
            return -np.mean(np.log(y_pred_clipped[np.arange(len(y_true)), y_true.astype(int)] + 1e-8))
        elif loss_type == LossType.HINGE:
            return np.mean(np.maximum(0, 1 - y_true * y_pred))
        elif loss_type == LossType.COSINE_SIMILARITY:
            return -np.mean(np.sum(y_true * y_pred, axis=-1) / (np.linalg.norm(y_true, axis=-1) * np.linalg.norm(y_pred, axis=-1) + 1e-8))
        elif loss_type == LossType.MSE:
            return np.mean((y_true - y_pred)**2)

        return 0.0

    @staticmethod
    def backward(y_true: np.ndarray, y_pred: np.ndarray, loss_type: LossType) -> np.ndarray:
        if loss_type == LossType.MSE:
            return 2 * (y_pred - y_true) / y_true.shape[0]
        elif loss_type == LossType.MAE:
            return np.sign(y_pred - y_true) / y_true.shape[0]
        elif loss_type == LossType.HUBER:
            delta = 1.0
            error = y_pred - y_true
            return np.where(np.abs(error) <= delta, error, delta * np.sign(error)) / y_true.shape[0]
        elif loss_type == LossType.CROSS_ENTROPY:
            return (y_pred - y_true) / y_true.shape[0]
        elif loss_type == LossType.BINARY_CROSS_ENTROPY:
            return (y_pred - y_true) / (y_pred * (1 - y_pred) + 1e-8) / y_true.shape[0]
        elif loss_type == LossType.CATEGORICAL_CROSS_ENTROPY:
            return (y_pred - y_true) / y_true.shape[0]
        elif loss_type == LossType.SPARSE_CATEGORICAL_CROSS_ENTROPY:
            grad = np.zeros_like(y_pred)
            grad[np.arange(len(y_true)), y_true.astype(int)] = -1 / (y_pred[np.arange(len(y_true)), y_true.astype(int)] + 1e-8)
            return grad / y_true.shape[0]
        elif loss_type == LossType.MSE:
            return 2 * (y_pred - y_true) / y_true.shape[0]

        return np.zeros_like(y_pred)


class Model:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.layers: List[Layer] = []
        self.optimizer = Optimizer(config.optimizer, config.learning_rate)
        self.loss_fn = Loss()

        self._build_layers()

    def _build_layers(self):
        input_shape = self.config.input_shape

        for layer_config in self.config.layers:
            layer = Layer(layer_config)
            layer.build(input_shape)
            self.layers.append(layer)

            if layer_config.layer_type in [LayerType.DENSE, LayerType.CONV2D, LayerType.CONV1D]:
                output_shape = self._compute_output_shape(input_shape, layer_config)
                input_shape = output_shape
            elif layer_config.layer_type in [LayerType.FLATTEN]:
                input_shape = (np.prod(input_shape),)
            else:
                input_shape = self._compute_output_shape(input_shape, layer_config)

    def _compute_output_shape(self, input_shape: Tuple[int, ...], layer_config: LayerConfig) -> Tuple[int, ...]:
        if layer_config.layer_type == LayerType.DENSE:
            return (layer_config.units,)
        elif layer_config.layer_type in [LayerType.CONV2D, LayerType.CONV1D]:
            h = input_shape[-3] if len(input_shape) >= 3 else 1
            w = input_shape[-2] if len(input_shape) >= 2 else 1
            kh, kw = layer_config.kernel_size
            sh, sw = layer_config.strides
            pad = layer_config.kernel_size[0] // 2 if layer_config.padding == "same" else 0
            out_h = (input_shape[-3] + 2*pad - layer_config.kernel_size[0]) // layer_config.strides[0] + 1
            out_w = (input_shape[-2] + 2*pad - layer_config.kernel_size[1]) // layer_config.strides[1] + 1
            return (out_h, out_w, layer_config.filters)
        elif layer_config.layer_type == LayerType.MAX_POOL2D:
            kh, kw = layer_config.kernel_size
            sh, sw = layer_config.strides
            h = (input_shape[-3] - kh) // layer_config.strides[0] + 1
            w = (input_shape[-2] - kw) // layer_config.strides[1] + 1
            return (h, w, input_shape[-1])
        return input_shape

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, training)
        return x

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self.loss_fn.forward(y_true, y_pred, self.config.loss)

    def compute_loss_grad(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return self.loss_fn.backward(y_true, y_pred, self.config.loss)

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        grad = self.compute_loss_grad(y_true, y_pred)

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return {}

    def train_step(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y_pred = self.forward(X, training=True)
        loss = self.compute_loss(y, y_pred)

        self.backward(y, y_pred)

        params = {}
        grads = {}
        for i, layer in enumerate(self.layers):
            if layer.weights is not None:
                params[f'layer_{i}_w'] = layer.weights
                params[f'layer_{i}_b'] = layer.bias

        updated_params = self.optimizer.update(params, grads)

        for i, layer in enumerate(self.layers):
            if layer.weights is not None:
                layer.weights = updated_params.get(f'layer_{i}_w', layer.weights)
                layer.bias = updated_params.get(f'layer_{i}_b', layer.bias)

        return {"loss": loss}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X, training=False)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X)
        loss = self.compute_loss(y, y_pred)

        metrics = {"loss": loss}
        if "accuracy" in self.config.metrics:
            if y.ndim > 1:
                accuracy = np.mean(np.argmax(y_pred, axis=-1) == np.argmax(y, axis=-1))
            else:
                accuracy = np.mean(np.round(y_pred.flatten()) == y.flatten())
            metrics["accuracy"] = float(accuracy)

        return metrics

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = None,
        epochs: int = None,
        validation_data: Tuple[np.ndarray, np.ndarray] = None,
        callbacks: List[Callable] = None
    ) -> Dict[str, List[float]]:
        batch_size = batch_size or self.config.batch_size
        epochs = epochs or self.config.epochs

        history = {"loss": [], "val_loss": []}

        n_samples = X.shape[0]
        steps_per_epoch = n_samples // batch_size

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            for step in range(steps_per_epoch):
                start = step * batch_size
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                result = self.train_step(X_batch, y_batch)
                epoch_loss += result["loss"]

            avg_loss = epoch_loss / steps_per_epoch
            history["loss"].append(avg_loss)

            if validation_data:
                X_val, y_val = validation_data
                val_metrics = self.evaluate(X_val, y_val)
                history["val_loss"].append(val_metrics["loss"])

            if callbacks:
                for cb in callbacks:
                    cb.on_epoch_end(epoch, {"loss": avg_loss})

        return history


class SequentialModel:
    def __init__(self):
        self.layers: List[Layer] = []

    def add(self, layer_config: LayerConfig):
        layer = Layer(layer_config)
        if self.layers:
            input_shape = self._get_output_shape(self.layers[-1])
        else:
            raise ValueError("First layer must specify input_shape")

        layer.build(input_shape)
        self.layers.append(layer)

    def _get_output_shape(self, layer: Layer) -> Tuple[int, ...]:
        return (layer.config.units or layer.config.filters or 1,)


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
