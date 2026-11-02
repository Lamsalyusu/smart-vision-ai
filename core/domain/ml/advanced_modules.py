from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Sequence, Callable, Any
from enum import Enum
from abc import ABC, abstractmethod
import math


class ActivationType(Enum):
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    GELU = "gelu"
    SWISH = "swish"
    MISH = "mish"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"
    SOFTPLUS = "softplus"
    SELU = "selu"
    PRELU = "prelu"


class NormalizationType(Enum):
    BATCH_NORM = "batch_norm"
    LAYER_NORM = "layer_norm"
    INSTANCE_NORM = "instance_norm"
    GROUP_NORM = "group_norm"
    WEIGHT_NORM = "weight_norm"
    SPECTRAL_NORM = "spectral_norm"


@dataclass(frozen=True, slots=True)
class Tensor:
    data: np.ndarray
    requires_grad: bool = False
    grad: Optional[np.ndarray] = None
    _prev: tuple = ()
    _op: str = ""

    def __post_init__(self):
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data)

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    def backward(self, grad: Optional[np.ndarray] = None):
        if grad is None:
            grad = np.ones_like(self.data)
        self.grad = grad
        self._backward()

    def _backward(self):
        if self._op == "add":
            self._prev[0].backward(self.grad)
            self._prev[1].backward(self.grad)
        elif self._op == "mul":
            self._prev[0].backward(self.grad * self._prev[1].data)
            self._prev[1].backward(self.grad * self._prev[0].data)
        elif self._op == "matmul":
            self._prev[0].backward(self.grad @ self._prev[1].data.T)
            self._prev[1].backward(self._prev[0].data.T @ self.grad)
        elif self._op == "relu":
            self._prev[0].backward(self.grad * (self._prev[0].data > 0))
        elif self._op == "sigmoid":
            s = 1 / (1 + np.exp(-self._prev[0].data))
            self._prev[0].backward(self.grad * s * (1 - s))
        elif self._op == "tanh":
            t = np.tanh(self._prev[0].data)
            self._prev[0].backward(self.grad * (1 - t**2))
        elif self._op == "sum":
            self._prev[0].backward(np.ones_like(self._prev[0].data) * self.grad.sum())

    def __add__(self, other: "Tensor") -> "Tensor":
        if not isinstance(other, Tensor):
            other = Tensor(np.array(other))
        out = Tensor(self.data + other.data, _prev=(self, other), _op="add")
        return out

    def __mul__(self, other: "Tensor") -> "Tensor":
        if not isinstance(other, Tensor):
            other = Tensor(np.array(other))
        out = Tensor(self.data * other.data, _prev=(self, other), _op="mul")
        return out

    def matmul(self, other: "Tensor") -> "Tensor":
        if not isinstance(other, Tensor):
            other = Tensor(np.array(other))
        out = Tensor(self.data @ other.data, _prev=(self, other), _op="matmul")
        return out

    def sum(self, axis: Optional[int] = None) -> "Tensor":
        out = Tensor(self.data.sum(axis=axis), _prev=(self,), _op="sum")
        return out

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(self.data, 0), _prev=(self,), _op="relu")
        return out

    def sigmoid(self) -> "Tensor":
        out = Tensor(1 / (1 + np.exp(-self.data)), _prev=(self,), _op="sigmoid")
        return out

    def tanh(self) -> "Tensor":
        out = Tensor(np.tanh(self.data), _prev=(self,), _op="tanh")
        return out

    def gelu(self) -> "Tensor":
        x = self.data
        out = Tensor(0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))), _prev=(self,), _op="gelu")
        return out

    def swish(self) -> "Tensor":
        s = 1 / (1 + np.exp(-self.data))
        out = Tensor(self.data * s, _prev=(self,), _op="swish")
        return out

    def mish(self) -> "Tensor":
        s = np.log(1 + np.exp(self.data))
        t = np.tanh(s)
        out = Tensor(self.data * t, _prev=(self,), _op="mish")
        return out

    def softmax(self, axis: int = -1) -> "Tensor":
        x = self.data - self.data.max(axis=axis, keepdims=True)
        exp_x = np.exp(x)
        out = Tensor(exp_x / exp_x.sum(axis=axis, keepdims=True), _prev=(self,), _op="softmax")
        return out

    def layer_norm(self, weight: "Tensor", bias: "Tensor", eps: float = 1e-5) -> "Tensor":
        mean = self.data.mean(axis=-1, keepdims=True)
        var = self.data.var(axis=-1, keepdims=True)
        normalized = (self.data - mean) / np.sqrt(var + eps)
        out = Tensor(normalized * weight.data + bias.data, _prev=(self,), _op="layer_norm")
        return out

    def dropout(self, p: float = 0.5, training: bool = True) -> "Tensor":
        if not training or p == 0:
            return self
        mask = np.random.binomial(1, 1 - p, self.data.shape) / (1 - p)
        out = Tensor(self.data * mask, _prev=(self,), _op="dropout")
        return out


class Module(ABC):
    def __init__(self):
        self._parameters: dict[str, Tensor] = {}
        self._modules: dict[str, Module] = {}
        self.training = True

    def parameters(self) -> list[Tensor]:
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def train(self):
        self.training = True
        for m in self._modules.values():
            m.train()

    def eval(self):
        self.training = False
        for m in self._modules.values():
            m.eval()

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    def _register_parameter(self, name: str, param: Tensor):
        self._parameters[name] = param

    def _register_module(self, name: str, module: "Module"):
        self._modules[name] = module


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        weight = np.random.randn(out_features, in_features) * np.sqrt(2 / in_features)
        self._register_parameter("weight", Tensor(weight.astype(np.float32)))
        if bias:
            bias_data = np.zeros(out_features, dtype=np.float32)
            self._register_parameter("bias", Tensor(bias_data))

    def forward(self, x: Tensor) -> Tensor:
        out = x.matmul(self._parameters["weight"])
        if "bias" in self._parameters:
            out = out + self._parameters["bias"]
        return out


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        fan_in = in_channels * kernel_size * kernel_size // groups
        weight = np.random.randn(out_channels, in_channels // groups, kernel_size, kernel_size) * np.sqrt(2 / fan_in)
        self._register_parameter("weight", Tensor(weight.astype(np.float32)))

        if bias:
            self._register_parameter("bias", Tensor(np.zeros(out_channels, dtype=np.float32)))

    def forward(self, x: Tensor) -> Tensor:
        batch, in_c, h, w = x.data.shape
        kh, kw = self.kernel_size, self.kernel_size

        if self.padding > 0:
            x_padded = np.pad(x.data, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            x_padded = x.data

        out_h = (h + 2 * self.padding - self.dilation * (kh - 1) - 1) // self.stride + 1
        out_w = (w + 2 * self.padding - self.dilation * (kw - 1) - 1) // self.stride + 1

        weight = self._parameters["weight"].data
        out_data = np.zeros((batch, self.out_channels, out_h, out_w), dtype=np.float32)

        for b in range(batch):
            for oc in range(self.out_channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        h_start = oh * self.stride
                        w_start = ow * self.stride
                        h_end = h_start + kh * self.dilation
                        w_end = w_start + kw * self.dilation

                        patch = x_padded[b, :, h_start:h_end:self.dilation, w_start:w_end:self.dilation]
                        out_data[b, oc, oh, ow] = np.sum(patch * weight[oc])

        if "bias" in self._parameters:
            out_data += self._parameters["bias"].data.reshape(1, -1, 1, 1)

        return Tensor(out_data)


class BatchNorm2d(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self._register_parameter("weight", Tensor(np.ones(num_features, dtype=np.float32)))
        self._register_parameter("bias", Tensor(np.zeros(num_features, dtype=np.float32)))
        self.register_buffer("running_mean", np.zeros(num_features, dtype=np.float32))
        self.register_buffer("running_var", np.ones(num_features, dtype=np.float32))

    def register_buffer(self, name: str, tensor: np.ndarray):
        setattr(self, name, tensor)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mean = x.data.mean(axis=(0, 2, 3))
            var = x.data.var(axis=(0, 2, 3))
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var

        normalized = (x.data - mean.reshape(1, -1, 1, 1)) / np.sqrt(var.reshape(1, -1, 1, 1) + self.eps)
        out = normalized * self._parameters["weight"].data.reshape(1, -1, 1, 1) + self._parameters["bias"].data.reshape(1, -1, 1, 1)
        return Tensor(out)


class LayerNorm(Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self._register_parameter("weight", Tensor(np.ones(normalized_shape, dtype=np.float32)))
        self._register_parameter("bias", Tensor(np.zeros(normalized_shape, dtype=np.float32)))

    def forward(self, x: Tensor) -> Tensor:
        mean = x.data.mean(axis=-1, keepdims=True)
        var = x.data.var(axis=-1, keepdims=True)
        normalized = (x.data - mean) / np.sqrt(var + self.eps)
        out = normalized * self._parameters["weight"].data + self._parameters["bias"].data
        return Tensor(out)


class Activation(Module):
    def __init__(self, activation_type: ActivationType):
        super().__init__()
        self.activation_type = activation_type

    def forward(self, x: Tensor) -> Tensor:
        if self.activation_type == ActivationType.RELU:
            return x.relu()
        elif self.activation_type == ActivationType.GELU:
            return x.gelu()
        elif self.activation_type == ActivationType.SWISH:
            return x.swish()
        elif self.activation_type == ActivationType.MISH:
            return x.mish()
        elif self.activation_type == ActivationType.SIGMOID:
            return x.sigmoid()
        elif self.activation_type == ActivationType.TANH:
            return x.tanh()
        elif self.activation_type == ActivationType.SOFTPLUS:
            return Tensor(np.log(1 + np.exp(x.data)))
        elif self.activation_type == ActivationType.SELU:
            scale = 1.0507
            alpha = 1.67326
            return Tensor(scale * np.where(x.data > 0, x.data, alpha * (np.exp(x.data) - 1)))
        else:
            return x.relu()


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return x.dropout(self.p, self.training)


class Sequential(Module):
    def __init__(self, *modules: Module):
        super().__init__()
        for i, module in enumerate(modules):
            self._register_module(str(i), module)

    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)
        return x


class MultiHeadAttention(Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        self.q_proj = Linear(embed_dim, embed_dim, bias)
        self.k_proj = Linear(embed_dim, embed_dim, bias)
        self.v_proj = Linear(embed_dim, embed_dim, bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias)
        self.dropout = Dropout(dropout)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch, seq_len, embed_dim = query.data.shape

        q = self.q_proj(query).data.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(key).data.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(value).data.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)

        if mask is not None:
            scores = scores + mask.data * -1e9

        attn_weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn_weights = attn_weights / attn_weights.sum(axis=-1, keepdims=True)

        if self.training:
            attn_weights = self.dropout(Tensor(attn_weights)).data

        attn_output = np.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch, seq_len, embed_dim)

        out = self.out_proj(Tensor(attn_output))
        return out


class TransformerBlock(Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = LayerNorm(embed_dim)
        self.mlp = Sequential(
            Linear(embed_dim, int(embed_dim * mlp_ratio)),
            Activation(ActivationType.GELU),
            Dropout(dropout),
            Linear(int(embed_dim * mlp_ratio), embed_dim),
            Dropout(dropout),
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = []
        for i in range(num_layers):
            block = TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            self._register_module(f"layer_{i}", block)
            self.layers.append(block)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerDecoder(Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = []
        for i in range(num_layers):
            block = TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            self._register_module(f"layer_{i}", block)
            self.layers.append(block)

    def forward(self, x: Tensor, memory: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Optimizer(ABC):
    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def zero_grad(self):
        pass


class SGD(Optimizer):
    def __init__(self, parameters: list[Tensor], lr: float = 0.01, momentum: float = 0.0, weight_decay: float = 0.0):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [np.zeros_like(p.data) for p in parameters]

    def step(self):
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            grad = p.grad + self.weight_decay * p.data
            self.velocity[i] = self.momentum * self.velocity[i] + grad
            p.data -= self.lr * self.velocity[i]

    def zero_grad(self):
        for p in self.parameters:
            p.grad = None


class Adam(Optimizer):
    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.parameters = parameters
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p.data) for p in parameters]
        self.v = [np.zeros_like(p.data) for p in parameters]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            grad = p.grad + self.weight_decay * p.data

            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * grad
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (grad ** 2)

            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            v_hat = self.v[i] / (1 - self.betas[1] ** self.t)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.parameters:
            p.grad = None


class AdamW(Optimizer):
    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.parameters = parameters
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p.data) for p in parameters]
        self.v = [np.zeros_like(p.data) for p in parameters]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            grad = p.grad

            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * grad
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (grad ** 2)

            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            v_hat = self.v[i] / (1 - self.betas[1] ** self.t)

            p.data -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * p.data)

    def zero_grad(self):
        for p in self.parameters:
            p.grad = None


class Loss(ABC):
    @abstractmethod
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pass


class MSELoss(Loss):
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred.data - target.data
        loss = Tensor((diff ** 2).mean())
        loss._prev = (pred, target)
        loss._op = "mse"
        return loss


class CrossEntropyLoss(Loss):
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        log_probs = np.log(pred.data + 1e-8)
        loss = Tensor(-(target.data * log_probs).sum(axis=-1).mean())
        loss._prev = (pred, target)
        loss._op = "cross_entropy"
        return loss


class BCELoss(Loss):
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        eps = 1e-8
        pred_clipped = np.clip(pred.data, eps, 1 - eps)
        loss = Tensor(-(target.data * np.log(pred_clipped) + (1 - target.data) * np.log(1 - pred_clipped)).mean())
        loss._prev = (pred, target)
        loss._op = "bce"
        return loss


class HuberLoss(Loss):
    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred.data - target.data
        abs_diff = np.abs(diff)
        loss = np.where(abs_diff < self.delta, 0.5 * diff ** 2, self.delta * (abs_diff - 0.5 * self.delta))
        loss_tensor = Tensor(loss.mean())
        loss_tensor._prev = (pred, target)
        loss_tensor._op = "huber"
        return loss_tensor


class DataLoader:
    def __init__(
        self,
        dataset: list[tuple[np.ndarray, np.ndarray]],
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_x = np.stack([self.dataset[idx][0] for idx in batch_indices])
            batch_y = np.stack([self.dataset[idx][1] for idx in batch_indices])
            yield Tensor(batch_x), Tensor(batch_y)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size