from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class Tensor:
    """N-dimensional array representation supporting automatic differentiation and broadcasting."""

    def __init__(
        self,
        data: float | Sequence | np.ndarray,
        requires_grad: bool = False,
        creator: tuple[str, list[Tensor]] | None = None,
    ) -> None:
        if isinstance(data, np.ndarray):
            self.data = data.astype(np.float32)
        elif isinstance(data, (float, int)):
            self.data = np.array([data], dtype=np.float32)
        else:
            self.data = np.array(data, dtype=np.float32)

        self.requires_grad = requires_grad
        self.grad: np.ndarray | None = None
        self.creator = creator

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def zero_grad(self) -> None:
        if self.grad is not None:
            self.grad.fill(0.0)

    def backward(self, grad_output: np.ndarray | None = None) -> None:
        """Propagate gradients backwards through the computation graph."""
        if not self.requires_grad:
            return

        if grad_output is None:
            grad_output = np.ones_like(self.data)

        if self.grad is None:
            self.grad = grad_output.copy()
        else:
            self.grad += grad_output

        if self.creator is None:
            return

        op_name, parents = self.creator

        if op_name == "add":
            p1, p2 = parents
            if p1.requires_grad:
                p1.backward(self._unbroadcast(grad_output, p1.shape))
            if p2.requires_grad:
                p2.backward(self._unbroadcast(grad_output, p2.shape))

        elif op_name == "mul":
            p1, p2 = parents
            if p1.requires_grad:
                p1.backward(self._unbroadcast(grad_output * p2.data, p1.shape))
            if p2.requires_grad:
                p2.backward(self._unbroadcast(grad_output * p1.data, p2.shape))

        elif op_name == "matmul":
            p1, p2 = parents
            if p1.requires_grad:
                p1.backward(grad_output @ p2.data.T)
            if p2.requires_grad:
                p2.backward(p1.data.T @ grad_output)

        elif op_name == "relu":
            p1 = parents[0]
            if p1.requires_grad:
                grad = grad_output * (p1.data > 0).astype(np.float32)
                p1.backward(grad)

    @staticmethod
    def _unbroadcast(grad: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
        """Sum gradient along broadcasted axes to match target tensor shape."""
        grad_ndim = grad.ndim
        target_ndim = len(target_shape)

        for _ in range(grad_ndim - target_ndim):
            grad = grad.sum(axis=0)

        for i, dim in enumerate(target_shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)

        return grad

    def __add__(self, other: Tensor | float) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other)
        req_grad = self.requires_grad or other.requires_grad
        return Tensor(self.data + other.data, requires_grad=req_grad, creator=("add", [self, other]))

    def __mul__(self, other: Tensor | float) -> Tensor:
        if not isinstance(other, Tensor):
            other = Tensor(other)
        req_grad = self.requires_grad or other.requires_grad
        return Tensor(self.data * other.data, requires_grad=req_grad, creator=("mul", [self, other]))

    def __matmul__(self, other: Tensor) -> Tensor:
        req_grad = self.requires_grad or other.requires_grad
        return Tensor(self.data @ other.data, requires_grad=req_grad, creator=("matmul", [self, other]))

    def relu(self) -> Tensor:
        out_data = np.maximum(0.0, self.data)
        return Tensor(out_data, requires_grad=self.requires_grad, creator=("relu", [self]))
