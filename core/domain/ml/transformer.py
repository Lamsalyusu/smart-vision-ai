from __future__ import annotations

import math

from .activations import Activations
from .layers import LayerNorm, Linear, Module
from .tensor import Tensor


class MultiHeadAttention(Module):
    """Scaled Dot-Product Multi-Head Attention layer."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        # Simplified forward pass for attention mechanism
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Scaled dot-product: Q @ K^T / sqrt(d_k)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = (q.data @ k.data.T) * scale
        attn_weights = Activations.softmax(attn_scores, axis=-1)

        attended = attn_weights @ v.data
        return self.out_proj(Tensor(attended, requires_grad=x.requires_grad))


class TransformerEncoderBlock(Module):
    """Transformer Encoder layer with residual connections and layer normalization."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        self.ln1 = LayerNorm(embed_dim)
        self.ffn1 = Linear(embed_dim, ffn_dim)
        self.ffn2 = Linear(ffn_dim, embed_dim)
        self.ln2 = LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        # Pre-LN architecture
        norm1 = self.ln1(x)
        attn_out = self.mha.forward(norm1)
        x2 = x + attn_out

        norm2 = self.ln2(x2)
        ffn_out = self.ffn2(self.ffn1(norm2).relu())
        return x2 + ffn_out
