from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple, Union
from enum import Enum
from abc import ABC, abstractmethod
import math
from collections import defaultdict, deque

class AttentionType(Enum):
    SELF = "self"
    CROSS = "cross"
    MULTI_HEAD = "multi_head"
    SPARSE = "sparse"
    LINEAR = "linear"
    FLASH = "flash"
    MEMORY_EFFICIENT = "memory_efficient"

class PositionalEncodingType(Enum):
    SINUSOIDAL = "sinusoidal"
    LEARNED = "learned"
    RELATIVE = "relative"
    ROTARY = "rotary"
    ALIBI = "alibi"
    T5_RELATIVE = "t5_relative"

class NormalizationType(Enum):
    LAYER_NORM = "layer_norm"
    RMS_NORM = "rms_norm"
    BATCH_NORM = "batch_norm"
    INSTANCE_NORM = "instance_norm"
    GROUP_NORM = "group_norm"

class ActivationType(Enum):
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    SWISH = "swish"
    GLU = "glu"
    REGULATED_GELU = "regulated_gelu"

@dataclass(frozen=True, slots=True)
class TransformerConfig:
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 2048
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    activation: ActivationType = ActivationType.GELU
    normalization: NormalizationType = NormalizationType.LAYER_NORM
    positional_encoding: PositionalEncodingType = PositionalEncodingType.SINUSOIDAL
    use_bias: bool = True
    tie_weights: bool = True
    rope_theta: float = 10000.0
    alibi_max_bias: float = 8.0
    attention_type: AttentionType = AttentionType.MULTI_HEAD
    window_size: int = 256
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-6
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

@dataclass(frozen=True, slots=True)
class AttentionOutput:
    output: np.ndarray
    attention_weights: Optional[np.ndarray] = None
    key_value_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None

@dataclass(frozen=True, slots=True)
class TransformerOutput:
    last_hidden_state: np.ndarray
    hidden_states: Optional[List[np.ndarray]] = None
    attentions: Optional[List[np.ndarray]] = None
    key_value_caches: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None

class Tensor:
    def __init__(self, data: np.ndarray, requires_grad: bool = False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        self._backward = lambda: None
        self._prev: List['Tensor'] = []

    def __add__(self, other: 'Tensor') -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data)
        out._prev = [self, other]
        def _backward():
            if self.requires_grad:
                self.grad = (self.grad if self.grad is not None else 0) + out.grad
            if other.requires_grad:
                other.grad = (other.grad if other.grad is not None else 0) + out.grad
        out._backward = _backward
        return out

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data)
        out._prev = [self, other]
        def _backward():
            if self.requires_grad:
                self.grad = (self.grad if self.grad is not None else 0) + out.grad @ other.data.T
            if other.requires_grad:
                other.grad = (other.grad if other.grad is not None else 0) + self.data.T @ out.grad
        out._backward = _backward
        return out

    def transpose(self, *axes) -> 'Tensor':
        out = Tensor(self.data.transpose(*axes))
        out._prev = [self]
        def _backward():
            if self.requires_grad:
                self.grad = (self.grad if self.grad is not None else 0) + out.grad.transpose(*axes)
        out._backward = _backward
        return out

    def reshape(self, *shape) -> 'Tensor':
        out = Tensor(self.data.reshape(*shape))
        out._prev = [self]
        def _backward():
            if self.requires_grad:
                self.grad = (self.grad if self.grad is not None else 0) + out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for u in v._prev:
                    build_topo(u)
                topo.append(v)
        build_topo(self)
        for v in reversed(topo):
            v._backward()

    def numpy(self) -> np.ndarray:
        return self.data

class Parameter(Tensor):
    def __init__(self, data: np.ndarray):
        super().__init__(data, requires_grad=True)

class Module(ABC):
    def __init__(self):
        self.training = True
        self._parameters: Dict[str, Parameter] = {}
        self._modules: Dict[str, 'Module'] = {}

    def __setattr__(self, name: str, value: Any):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def parameters(self) -> List[Parameter]:
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def named_parameters(self) -> Dict[str, Parameter]:
        result = {}
        for name, param in self._parameters.items():
            result[name] = param
        for mod_name, module in self._modules.items():
            for name, param in module.named_parameters().items():
                result[f"{mod_name}.{name}"] = param
        return result

    def train(self):
        self.training = True
        for module in self._modules.values():
            module.train()

    def eval(self):
        self.training = False
        for module in self._modules.values():
            module.eval()

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(np.random.randn(out_features, in_features) * math.sqrt(2.0 / in_features))
        if bias:
            self.bias = Parameter(np.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.transpose()
        if self.bias is not None:
            out = out + self.bias
        return out

class LayerNorm(Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        if elementwise_affine:
            self.weight = Parameter(np.ones(normalized_shape))
            self.bias = Parameter(np.zeros(normalized_shape))
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        mean = np.mean(x.data, axis=-1, keepdims=True)
        var = np.var(x.data, axis=-1, keepdims=True)
        normalized = (x.data - mean) / np.sqrt(var + self.eps)
        if self.weight is not None:
            normalized = normalized * self.weight.data + self.bias.data
        return Tensor(normalized)

class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(np.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = np.sqrt(np.mean(x.data**2, axis=-1, keepdims=True) + self.eps)
        normalized = x.data / rms * self.weight.data
        return Tensor(normalized)

class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = Parameter(np.random.randn(num_embeddings, embedding_dim) * 0.02)

    def forward(self, x: Tensor) -> Tensor:
        indices = x.data.astype(int)
        out_data = self.weight.data[indices]
        return Tensor(out_data)

class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x
        mask = np.random.binomial(1, 1 - self.p, x.data.shape) / (1 - self.p)
        return Tensor(x.data * mask)

class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        data = x.data
        gelu = 0.5 * data * (1 + np.tanh(math.sqrt(2 / math.pi) * (data + 0.044715 * data**3)))
        return Tensor(gelu)

class SiLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        data = x.data
        silu = data / (1 + np.exp(-data))
        return Tensor(silu)

class GLU(Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        a, b = np.split(x.data, 2, axis=self.dim)
        return Tensor(a * (1 / (1 + np.exp(-b))))

class RotaryEmbedding:
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0

    def _compute_cos_sin(self, seq_len: int):
        if seq_len <= self._seq_len_cached and self._cos_cached is not None:
            return self._cos_cached[:seq_len], self._sin_cached[:seq_len]

        t = np.arange(seq_len)
        freqs = 1.0 / (self.theta ** (np.arange(0, self.dim, 2) / self.dim))
        freqs = np.outer(t, freqs)
        cos = np.cos(freqs)
        sin = np.sin(freqs)

        self._cos_cached = cos
        self._sin_cached = sin
        self._seq_len_cached = seq_len

        return cos, sin

    def apply_rotary(self, x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        cos = cos[:, :x1.shape[-1]]
        sin = sin[:, :x1.shape[-1]]

        rotated = np.empty_like(x)
        rotated[..., ::2] = x1 * cos - x2 * sin
        rotated[..., 1::2] = x1 * sin + x2 * cos

        return rotated

class AlibiBias:
    def __init__(self, num_heads: int, max_bias: float = 8.0):
        self.num_heads = num_heads
        self.max_bias = max_bias
        self._slopes = self._get_slopes(num_heads)

    def _get_slopes(self, num_heads: int) -> np.ndarray:
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(num_heads).is_integer():
            return np.array(get_slopes_power_of_2(num_heads))
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            slopes_power_of_2 = get_slopes_power_of_2(closest_power_of_2)
            slopes = np.concatenate([slopes_power_of_2, get_slopes_power_of_2(2 * closest_power_of_2)[1::2]])
            return slopes[:num_heads]

    def get_bias(self, seq_len: int) -> np.ndarray:
        bias = np.zeros((self.num_heads, seq_len, seq_len))
        for h in range(self.num_heads):
            for i in range(seq_len):
                for j in range(seq_len):
                    bias[h, i, j] = -self._slopes[h] * abs(i - j)
        return bias

class MultiHeadAttention(Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_key_value_heads = config.num_key_value_heads or config.num_heads
        self.head_dim = config.head_dim or config.hidden_size // config.num_heads

        self.q_proj = Linear(config.hidden_size, config.num_heads * self.head_dim, bias=config.use_bias)
        self.k_proj = Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.use_bias)
        self.v_proj = Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.use_bias)
        self.o_proj = Linear(config.num_heads * self.head_dim, config.hidden_size, bias=config.use_bias)

        self.attention_dropout = Dropout(config.attention_dropout)

        self.rotary = RotaryEmbedding(self.head_dim, config.max_position_embeddings, config.rope_theta)
        self.alibi = AlibiBias(config.num_heads, config.alibi_max_bias)

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None,
                position_ids: Optional[Tensor] = None,
                past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
                use_cache: bool = False) -> AttentionOutput:
        batch_size, seq_len, _ = hidden_states.data.shape

        q = self.q_proj(hidden_states).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(hidden_states).reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(hidden_states).reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)

        if position_ids is not None:
            cos, sin = self.rotary._compute_cos_sin(seq_len)
            q = Tensor(self.rotary.apply_rotary(q.data, cos, sin))
            k = Tensor(self.rotary.apply_rotary(k.data, cos, sin))

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = Tensor(np.concatenate([past_k.data, k.data], axis=2))
            v = Tensor(np.concatenate([past_v.data, v.data], axis=2))

        new_key_value = (k, v) if use_cache else None

        if self.num_key_value_heads != self.num_heads:
            k = self._repeat_kv(k, self.num_heads // self.num_key_value_heads)
            v = self._repeat_kv(v, self.num_heads // self.num_key_value_heads)

        attn_weights = q @ k.transpose(0, 1, 3, 2) / math.sqrt(self.head_dim)

        if self.config.positional_encoding == PositionalEncodingType.ALIBI:
            alibi_bias = self.alibi.get_bias(seq_len)
            attn_weights = Tensor(attn_weights.data + alibi_bias)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = self._softmax(attn_weights)
        attn_weights = self.attention_dropout(attn_weights)

        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        output = self.o_proj(attn_output)

        return AttentionOutput(
            output=output,
            attention_weights=attn_weights if not self.training else None,
            key_value_cache=new_key_value
        )

    def _repeat_kv(self, x: Tensor, n_rep: int) -> Tensor:
        if n_rep == 1:
            return x
        batch_size, num_kv_heads, seq_len, head_dim = x.data.shape
        x = Tensor(np.repeat(x.data, n_rep, axis=1))
        return x

    def _softmax(self, x: Tensor) -> Tensor:
        max_val = np.max(x.data, axis=-1, keepdims=True)
        exp_x = np.exp(x.data - max_val)
        sum_exp = np.sum(exp_x, axis=-1, keepdims=True)
        return Tensor(exp_x / sum_exp)

class FeedForward(Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation = self._get_activation(config.activation)

    def _get_activation(self, act_type: ActivationType) -> Module:
        if act_type == ActivationType.GELU:
            return GELU()
        elif act_type == ActivationType.SILU:
            return SiLU()
        elif act_type == ActivationType.RELU:
            return ReLU()
        elif act_type == ActivationType.GLU:
            return GLU()
        else:
            return GELU()

    def forward(self, x: Tensor) -> Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        activated = self.activation(gate) * up
        return self.down_proj(activated)

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(np.maximum(x.data, 0))

class TransformerBlock(Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

        if config.normalization == NormalizationType.LAYER_NORM:
            self.norm1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.norm2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        elif config.normalization == NormalizationType.RMS_NORM:
            self.norm1 = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.norm2 = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.norm1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.norm2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.dropout = Dropout(config.hidden_dropout)

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None,
                position_ids: Optional[Tensor] = None,
                past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
                use_cache: bool = False) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        attn_output = self.attention(hidden_states, attention_mask, position_ids, past_key_value, use_cache)
        hidden_states = residual + self.dropout(attn_output.output)

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        ff_output = self.feed_forward(hidden_states)
        hidden_states = residual + self.dropout(ff_output)

        return hidden_states, attn_output.key_value_cache

class TransformerModel(Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embeddings = Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.position_embeddings = self._create_positional_embeddings()
        self.dropout = Dropout(config.hidden_dropout)
        self.layers = [TransformerBlock(config) for _ in range(config.num_layers)]

        for i, layer in enumerate(self.layers):
            setattr(self, f"layer_{i}", layer)

        if config.normalization == NormalizationType.LAYER_NORM:
            self.final_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        elif config.normalization == NormalizationType.RMS_NORM:
            self.final_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.final_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if config.tie_weights:
            self.lm_head_weight = self.embeddings.weight
        else:
            self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def _create_positional_embeddings(self) -> Optional[Embedding]:
        if self.config.positional_encoding == PositionalEncodingType.LEARNED:
            return Embedding(self.config.max_position_embeddings, self.config.hidden_size)
        return None

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None,
                position_ids: Optional[Tensor] = None,
                past_key_values: Optional[List[Tuple[Tensor, Tensor]]] = None,
                use_cache: bool = False,
                output_hidden_states: bool = False,
                output_attentions: bool = False) -> TransformerOutput:
        batch_size, seq_len = input_ids.data.shape

        if position_ids is None:
            position_ids = Tensor(np.arange(seq_len).reshape(1, -1).repeat(batch_size, axis=0))

        hidden_states = self.embeddings(input_ids)

        if self.position_embeddings is not None:
            pos_emb = self.position_embeddings(position_ids)
            hidden_states = hidden_states + pos_emb

        hidden_states = self.dropout(hidden_states)

        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        new_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states.data.copy())

            past_kv = past_key_values[i] if past_key_values else None
            hidden_states, new_kv = layer(hidden_states, attention_mask, position_ids, past_kv, use_cache)

            if use_cache:
                new_key_values.append(new_kv)
            if output_attentions:
                all_attentions.append(None)

        hidden_states = self.final_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states.data.copy())

        return TransformerOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            key_value_caches=new_key_values
        )

    def generate(self, input_ids: Tensor, max_length: int = 100,
                 temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9,
                 do_sample: bool = True) -> Tensor:
        self.eval()
        generated = input_ids.data.copy()
        past_key_values = None

        for _ in range(max_length - input_ids.data.shape[1]):
            input_tensor = Tensor(generated[:, -1:] if past_key_values else generated)
            position_ids = Tensor(np.array([[generated.shape[1] - 1]] * generated.shape[0]))

            output = self.forward(input_tensor, position_ids=position_ids,
                                  past_key_values=past_key_values, use_cache=True)

            logits = self._get_logits(output.last_hidden_state)
            logits = logits[:, -1, :] / temperature

            if do_sample:
                next_token = self._sample(logits, top_k, top_p)
            else:
                next_token = np.argmax(logits.data, axis=-1, keepdims=True)

            generated = np.concatenate([generated, next_token], axis=1)
            past_key_values = output.key_value_caches

            if np.any(next_token == self.config.eos_token_id):
                break

        return Tensor(generated)

    def _get_logits(self, hidden_states: Tensor) -> Tensor:
        if self.config.tie_weights:
            return hidden_states @ self.lm_head_weight.transpose()
        else:
            return self.lm_head(hidden_states)

    def _sample(self, logits: Tensor, top_k: int, top_p: float) -> np.ndarray:
        probs = self._softmax(logits).data

        if top_k > 0:
            indices_to_remove = probs < np.partition(probs, -top_k, axis=-1)[:, -top_k:]
            probs[indices_to_remove] = 0
            probs = probs / probs.sum(axis=-1, keepdims=True)

        if top_p < 1.0:
            sorted_indices = np.argsort(probs, axis=-1)[:, ::-1]
            sorted_probs = np.take_along_axis(probs, sorted_indices, axis=-1)
            cumsum_probs = np.cumsum(sorted_probs, axis=-1)
            sorted_indices_to_remove = cumsum_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1]
            sorted_indices_to_remove[:, 0] = False
            indices_to_remove = np.take_along_axis(sorted_indices_to_remove, np.argsort(sorted_indices, axis=-1), axis=-1)
            probs[indices_to_remove] = 0
            probs = probs / probs.sum(axis=-1, keepdims=True)

        return np.array([np.random.choice(len(p), p=p) for p in probs]).reshape(-1, 1)

    def _softmax(self, x: Tensor) -> Tensor:
        max_val = np.max(x.data, axis=-1, keepdims=True)
        exp_x = np.exp(x.data - max_val)
        sum_exp = np.sum(exp_x, axis=-1, keepdims=True)
        return Tensor(exp_x / sum_exp)

class CrossAttention(Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim or config.hidden_size // config.num_heads

        self.q_proj = Linear(config.hidden_size, config.num_heads * self.head_dim, bias=config.use_bias)
        self.k_proj = Linear(config.hidden_size, config.num_heads * self.head_dim, bias=config.use_bias)
        self.v_proj = Linear(config.hidden_size, config.num_heads * self.head_dim, bias=config.use_bias)
        self.o_proj = Linear(config.num_heads * self.head_dim, config.hidden_size, bias=config.use_bias)
        self.dropout = Dropout(config.attention_dropout)

    def forward(self, query: Tensor, key_value: Tensor,
                attention_mask: Optional[Tensor] = None) -> AttentionOutput:
        batch_size, q_len, _ = query.data.shape
        _, kv_len, _ = key_value.data.shape

        q = self.q_proj(query).reshape(batch_size, q_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(key_value).reshape(batch_size, kv_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(key_value).reshape(batch_size, kv_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        attn_weights = q @ k.transpose(0, 1, 3, 2) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = self._softmax(attn_weights)
        attn_weights = self.dropout(attn_weights)

        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, q_len, self.num_heads * self.head_dim)
        output = self.o_proj(attn_output)

        return AttentionOutput(
            output=output,
            attention_weights=attn_weights
        )

    def _softmax(self, x: Tensor) -> Tensor:
        max_val = np.max(x.data, axis=-1, keepdims=True)
        exp_x = np.exp(x.data - max_val)
        sum_exp = np.sum(exp_x, axis=-1, keepdims=True)
        return Tensor(exp_x / sum_exp)

class VisionTransformer(Module):
    def __init__(self, config: TransformerConfig, image_size: int = 224,
                 patch_size: int = 16, num_channels: int = 3):
        super().__init__()
        self.config = config
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embedding = Linear(num_channels * patch_size * patch_size, config.hidden_size)
        self.cls_token = Parameter(np.random.randn(1, 1, config.hidden_size) * 0.02)
        self.position_embeddings = Parameter(np.random.randn(1, self.num_patches + 1, config.hidden_size) * 0.02)
        self.dropout = Dropout(config.hidden_dropout)

        self.layers = [TransformerBlock(config) for _ in range(config.num_layers)]
        for i, layer in enumerate(self.layers):
            setattr(self, f"layer_{i}", layer)

        if config.normalization == NormalizationType.LAYER_NORM:
            self.norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: Tensor) -> TransformerOutput:
        batch_size = pixel_values.data.shape[0]

        patches = self._patchify(pixel_values)
        hidden_states = self.patch_embedding(patches)

        cls_tokens = Tensor(np.repeat(self.cls_token.data, batch_size, axis=0))
        hidden_states = Tensor(np.concatenate([cls_tokens.data, hidden_states.data], axis=1))

        hidden_states = hidden_states + Tensor(self.position_embeddings.data)
        hidden_states = self.dropout(hidden_states)

        all_hidden_states = []
        for layer in self.layers:
            all_hidden_states.append(hidden_states.data.copy())
            hidden_states, _ = layer(hidden_states)

        hidden_states = self.norm(hidden_states)
        all_hidden_states.append(hidden_states.data.copy())

        return TransformerOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states
        )

    def _patchify(self, images: Tensor) -> Tensor:
        batch_size, channels, height, width = images.data.shape
        assert height == self.image_size and width == self.image_size

        patches = images.data.reshape(batch_size, channels,
                                       height // self.patch_size, self.patch_size,
                                       width // self.patch_size, self.patch_size)
        patches = patches.transpose(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(batch_size, self.num_patches, channels * self.patch_size * self.patch_size)
        return Tensor(patches)

class TransformerEncoder(Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.layers = [TransformerBlock(config) for _ in range(config.num_layers)]
        for i, layer in enumerate(self.layers):
            setattr(self, f"layer_{i}", layer)

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None,
                output_hidden_states: bool = False,
                output_attentions: bool = False) -> TransformerOutput:
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states.data.copy())
            hidden_states, _ = layer(hidden_states, attention_mask)
            if output_attentions:
                all_attentions.append(None)

        if output_hidden_states:
            all_hidden_states.append(hidden_states.data.copy())

        return TransformerOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions
        )

class TransformerDecoder(Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.layers = [TransformerDecoderLayer(config) for _ in range(config.num_layers)]
        for i, layer in enumerate(self.layers):
            setattr(self, f"layer_{i}", layer)

class TransformerDecoderLayer(Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attention = MultiHeadAttention(config)
        self.cross_attention = CrossAttention(config)
        self.feed_forward = FeedForward(config)

        if config.normalization == NormalizationType.LAYER_NORM:
            self.norm1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.norm2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.norm3 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.norm1 = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.norm2 = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.norm3 = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.dropout = Dropout(config.hidden_dropout)

    def forward(self, hidden_states: Tensor, encoder_hidden_states: Tensor,
                self_attention_mask: Optional[Tensor] = None,
                cross_attention_mask: Optional[Tensor] = None,
                past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
                use_cache: bool = False) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        self_attn = self.self_attention(hidden_states, self_attention_mask, past_key_value=past_key_value, use_cache=use_cache)
        hidden_states = residual + self.dropout(self_attn.output)

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        cross_attn = self.cross_attention(hidden_states, encoder_hidden_states, cross_attention_mask)
        hidden_states = residual + self.dropout(cross_attn.output)

        residual = hidden_states
        hidden_states = self.norm3(hidden_states)
        ff_output = self.feed_forward(hidden_states)
        hidden_states = residual + self.dropout(ff_output)

        return hidden_states, self_attn.key_value_cache

class EncoderDecoderModel(Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self.embeddings = Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor, decoder_input_ids: Tensor,
                attention_mask: Optional[Tensor] = None,
                decoder_attention_mask: Optional[Tensor] = None) -> TransformerOutput:
        encoder_output = self.encoder(self.embeddings(input_ids), attention_mask)
        decoder_output = self.decoder(self.embeddings(decoder_input_ids), encoder_output.last_hidden_state,
                                       decoder_attention_mask)
        return TransformerOutput(last_hidden_state=decoder_output.last_hidden_state)

def create_transformer_config(model_type: str = "base") -> TransformerConfig:
    configs = {
        "tiny": TransformerConfig(hidden_size=128, num_layers=2, num_heads=2, intermediate_size=512),
        "small": TransformerConfig(hidden_size=256, num_layers=4, num_heads=4, intermediate_size=1024),
        "base": TransformerConfig(hidden_size=768, num_layers=12, num_heads=12, intermediate_size=3072),
        "large": TransformerConfig(hidden_size=1024, num_layers=24, num_heads=16, intermediate_size=4096),
        "xl": TransformerConfig(hidden_size=1280, num_layers=36, num_heads=20, intermediate_size=5120),
        "xxl": TransformerConfig(hidden_size=1600, num_layers=48, num_heads=25, intermediate_size=6400),
    }
    return configs.get(model_type, configs["base"])
