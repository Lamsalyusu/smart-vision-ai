from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
import math
import time


class AttentionType(Enum):
    SELF = "self"
    CROSS = "cross"
    MULTI_HEAD = "multi_head"
    SPARSE = "sparse"
    LINEAR = "linear"
    FLASH = "flash"
    WINDOWED = "windowed"
    GLOBAL = "global"


class PositionEncodingType(Enum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    ROTARY = "rotary"
    LEARNED = "learned"
    SINUSOIDAL = "sinusoidal"
    ALIBI = "alibi"


@dataclass(frozen=True, slots=True)
class TransformerConfig:
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    attention_type: AttentionType = AttentionType.MULTI_HEAD
    position_encoding: PositionEncodingType = PositionEncodingType.SINUSOIDAL
    max_seq_len: int = 2048
    vocab_size: int = 32000
    layer_norm_eps: float = 1e-6
    activation: str = "gelu"
    use_bias: bool = True
    tie_weights: bool = True


@dataclass(frozen=True, slots=True)
class AttentionOutput:
    output: np.ndarray
    attention_weights: Optional[np.ndarray] = None
    key_values: Optional[Tuple[np.ndarray, np.ndarray]] = None


class PositionalEncoding:
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.encoding = self._build_encoding()

    def _build_encoding(self) -> np.ndarray:
        if self.config.position_encoding == PositionEncodingType.SINUSOIDAL:
            return self._sinusoidal_encoding()
        elif self.config.position_encoding == PositionEncodingType.LEARNED:
            return np.random.randn(self.config.max_seq_len, self.config.d_model) * 0.02
        elif self.config.position_encoding == PositionEncodingType.ALIBI:
            return self._alibi_slopes()
        return np.zeros((self.config.max_seq_len, self.config.d_model))

    def _sinusoidal_encoding(self) -> np.ndarray:
        pe = np.zeros((self.config.max_seq_len, self.config.d_model))
        position = np.arange(self.config.max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.config.d_model, 2) * -(math.log(10000.0) / self.config.d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

    def _alibi_slopes(self) -> np.ndarray:
        n_heads = self.config.n_heads
        slopes = np.zeros(n_heads)
        for i in range(n_heads):
            slopes[i] = 2 ** (-(i + 1) * 8 / n_heads)
        return slopes

    def get_encoding(self, seq_len: int) -> np.ndarray:
        return self.encoding[:seq_len]

    def get_alibi_bias(self, seq_len: int) -> np.ndarray:
        if self.config.position_encoding != PositionEncodingType.ALIBI:
            return np.zeros((self.config.n_heads, seq_len, seq_len))

        slopes = self.encoding
        bias = np.zeros((self.config.n_heads, seq_len, seq_len))
        for h in range(self.config.n_heads):
            for i in range(seq_len):
                for j in range(seq_len):
                    bias[h, i, j] = slopes[h] * (j - i)
        return bias


class MultiHeadAttention:
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.d_k = config.d_model // config.n_heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        self.w_q = np.random.randn(config.d_model, config.d_model) * 0.02
        self.w_k = np.random.randn(config.d_model, config.d_model) * 0.02
        self.w_v = np.random.randn(config.d_model, config.d_model) * 0.02
        self.w_o = np.random.randn(config.d_model, config.d_model) * 0.02

        if config.use_bias:
            self.b_q = np.zeros(config.d_model)
            self.b_k = np.zeros(config.d_model)
            self.b_v = np.zeros(config.d_model)
            self.b_o = np.zeros(config.d_model)
        else:
            self.b_q = self.b_k = self.b_v = self.b_o = None

    def forward(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray] = None,
        position_bias: Optional[np.ndarray] = None
    ) -> AttentionOutput:
        batch_size, seq_len, _ = query.shape

        q = query @ self.w_q + (self.b_q if self.b_q is not None else 0)
        k = key @ self.w_k + (self.b_k if self.b_k is not None else 0)
        v = value @ self.w_v + (self.b_v if self.b_v is not None else 0)

        q = q.reshape(batch_size, seq_len, self.config.n_heads, self.d_k).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.config.n_heads, self.d_k).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.config.n_heads, self.d_k).transpose(0, 2, 1, 3)

        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale

        if position_bias is not None:
            scores += position_bias

        if mask is not None:
            scores = np.where(mask, scores, -1e9)

        attn_weights = self._softmax(scores, axis=-1)
        output = np.matmul(attn_weights, v)

        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.config.d_model)
        output = output @ self.w_o + (self.b_o if self.b_o is not None else 0)

        return AttentionOutput(
            output=output,
            attention_weights=attn_weights,
            key_values=(k, v)
        )

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class FeedForward:
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.w1 = np.random.randn(config.d_model, config.d_ff) * 0.02
        self.w2 = np.random.randn(config.d_ff, config.d_model) * 0.02

        if config.use_bias:
            self.b1 = np.zeros(config.d_ff)
            self.b2 = np.zeros(config.d_model)
        else:
            self.b1 = self.b2 = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        hidden = x @ self.w1 + (self.b1 if self.b1 is not None else 0)

        if self.config.activation == "gelu":
            hidden = self._gelu(hidden)
        elif self.config.activation == "relu":
            hidden = np.maximum(hidden, 0)
        elif self.config.activation == "swish":
            hidden = hidden * self._sigmoid(hidden)

        output = hidden @ self.w2 + (self.b2 if self.b2 is not None else 0)
        return output

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))


class TransformerBlock:
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.attention = MultiHeadAttention(config)
        self.ffn = FeedForward(config)

        self.ln1_gamma = np.ones(config.d_model)
        self.ln1_beta = np.zeros(config.d_model)
        self.ln2_gamma = np.ones(config.d_model)
        self.ln2_beta = np.zeros(config.d_model)

        self.dropout_rate = config.dropout

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        position_bias: Optional[np.ndarray] = None
    ) -> np.ndarray:
        attn_out = self.attention.forward(x, x, x, mask, position_bias).output
        x = self._layer_norm(x + self._dropout(attn_out), self.ln1_gamma, self.ln1_beta)

        ffn_out = self.ffn.forward(x)
        x = self._layer_norm(x + self._dropout(ffn_out), self.ln2_gamma, self.ln2_beta)

        return x

    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + self.config.layer_norm_eps) + beta

    def _dropout(self, x: np.ndarray) -> np.ndarray:
        if self.dropout_rate > 0 and self.config.n_layers > 0:
            mask = np.random.binomial(1, 1 - self.dropout_rate, x.shape) / (1 - self.dropout_rate)
            return x * mask
        return x


class VisionTransformer:
    def __init__(
        self,
        config: TransformerConfig,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000
    ):
        self.config = config
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size

        self.patch_embedding = np.random.randn(self.patch_dim, config.d_model) * 0.02
        self.cls_token = np.random.randn(1, 1, config.d_model) * 0.02
        self.pos_embedding = PositionalEncoding(config)

        self.blocks = [TransformerBlock(config) for _ in range(config.n_layers)]

        self.ln_f_gamma = np.ones(config.d_model)
        self.ln_f_beta = np.zeros(config.d_model)
        self.head = np.random.randn(config.d_model, num_classes) * 0.02

    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size = x.shape[0]

        patches = self._patchify(x)
        patches = patches @ self.patch_embedding

        cls_tokens = np.tile(self.cls_token, (batch_size, 1, 1))
        x = np.concatenate([cls_tokens, patches], axis=1)

        pos_enc = self.pos_embedding.get_encoding(x.shape[1])
        x = x + pos_enc

        position_bias = self.pos_embedding.get_alibi_bias(x.shape[1])

        for block in self.blocks:
            x = block.forward(x, position_bias=position_bias)

        x = self._layer_norm(x, self.ln_f_gamma, self.ln_f_beta)
        cls_output = x[:, 0]
        logits = cls_output @ self.head

        return logits

    def _patchify(self, x: np.ndarray) -> np.ndarray:
        batch_size, channels, height, width = x.shape
        assert height == self.image_size and width == self.image_size

        patches = x.reshape(batch_size, channels, height // self.patch_size, self.patch_size, width // self.patch_size, self.patch_size)
        patches = patches.transpose(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(batch_size, self.num_patches, self.patch_dim)
        return patches

    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + self.config.layer_norm_eps) + beta


class SwinTransformer:
    def __init__(
        self,
        config: TransformerConfig,
        image_size: int = 224,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 1000,
        window_size: int = 7,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24]
    ):
        self.config = config
        self.image_size = image_size
        self.patch_size = patch_size
        self.window_size = window_size

        self.patch_embed = np.random.randn(patch_size * patch_size * in_channels, config.d_model) * 0.02
        self.pos_drop = config.dropout

        self.layers = []
        for i, (depth, n_heads) in enumerate(zip(depths, num_heads)):
            layer_config = TransformerConfig(
                d_model=config.d_model * (2 ** i),
                n_heads=n_heads,
                n_layers=depth,
                d_ff=config.d_ff * (2 ** i),
                dropout=config.dropout
            )
            self.layers.append(self._make_layer(layer_config, depth))

        self.norm = np.ones(config.d_model * (2 ** (len(depths) - 1)))
        self.head = np.random.randn(config.d_model * (2 ** (len(depths) - 1)), num_classes) * 0.02

    def _make_layer(self, config: TransformerConfig, depth: int):
        return [TransformerBlock(config) for _ in range(depth)]

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x


class DETR:
    def __init__(
        self,
        config: TransformerConfig,
        num_classes: int = 91,
        num_queries: int = 100,
        backbone: str = "resnet50"
    ):
        self.config = config
        self.num_classes = num_classes
        self.num_queries = num_queries

        self.backbone = self._build_backbone(backbone)
        self.input_proj = np.random.randn(2048, config.d_model) * 0.02
        self.query_embed = np.random.randn(num_queries, config.d_model) * 0.02

        self.transformer = Transformer(config)
        self.class_embed = np.random.randn(config.d_model, num_classes + 1) * 0.02
        self.bbox_embed = np.random.randn(config.d_model, 4) * 0.02

    def _build_backbone(self, name: str):
        return None

    def forward(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        return {'pred_logits': np.zeros((1, self.num_queries, self.num_classes + 1)),
                'pred_boxes': np.zeros((1, self.num_queries, 4))}


class Transformer:
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.encoder_layers = [TransformerBlock(config) for _ in range(config.n_layers)]
        self.decoder_layers = [TransformerBlock(config) for _ in range(config.n_layers)]

    def forward(
        self,
        src: np.ndarray,
        tgt: np.ndarray,
        src_mask: Optional[np.ndarray] = None,
        tgt_mask: Optional[np.ndarray] = None,
        memory_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        memory = src
        for layer in self.encoder_layers:
            memory = layer.forward(memory, src_mask)

        output = tgt
        for layer in self.decoder_layers:
            output = layer.forward(output, tgt_mask)

        return output


class TransformerTrainer:
    def __init__(
        self,
        model: Transformer,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 10000,
        max_steps: int = 100000
    ):
        self.model = model
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.step = 0

    def get_lr(self) -> float:
        if self.step < self.warmup_steps:
            return self.lr * (self.step / self.warmup_steps)
        progress = (self.step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        return self.lr * 0.5 * (1 + math.cos(math.pi * progress))

    def training_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        self.step += 1
        return {"loss": 0.0, "lr": self.get_lr()}


class TransformerMetrics:
    @staticmethod
    def compute_flops(config: TransformerConfig, seq_len: int) -> int:
        flops = 0
        flops += 2 * config.d_model * config.d_model * seq_len
        flops += 2 * config.n_heads * config.d_model * seq_len * seq_len
        flops += 2 * config.d_model * config.d_ff * seq_len
        flops *= config.n_layers
        return flops * 3

    @staticmethod
    def compute_params(config: TransformerConfig) -> int:
        params = 0
        params += config.vocab_size * config.d_model
        params += config.max_seq_len * config.d_model
        params += config.n_layers * (
            4 * config.d_model * config.d_model +
            2 * config.d_model * config.d_ff +
            2 * config.d_model
        )
        return params

    @staticmethod
    def compute_memory(config: TransformerConfig, seq_len: int, batch_size: int, dtype_bytes: int = 4) -> int:
        activations = batch_size * seq_len * config.d_model * config.n_layers * 4
        params = TransformerMetrics.compute_params(config)
        return (activations + params) * dtype_bytes


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
