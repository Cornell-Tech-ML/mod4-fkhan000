from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData

# Comment these out if not yet implemented
from .tensor_functions import (
    EQ,
    LT,
    Add,
    All,
    Copy,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
    tensor,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

    import numpy.typing as npt

    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


@dataclass
class History:
    """`History` stores the history of `Function` operations that were
    used to construct the current Variable.
    """

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


class Tensor:
    """Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    """

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        """Initializes a tensor with data, history, name, and backend."""
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = backend

    def requires_grad_(self, x: bool) -> None:
        """Sets the tensor to require gradients for backpropagation."""
        self.history = History()

    def requires_grad(self) -> bool:
        """Checks if the tensor requires gradients for backpropagation."""
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Converts the tensor to a numpy array."""
        return self.contiguous()._tensor._storage.reshape(self.shape)

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Ensures the input is a tensor. Converts numbers to tensors if needed."""
        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            c = b
        return c

    def item(self) -> float:
        """Converts a 1-element tensor to a float."""
        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def contiguous(self) -> Tensor:
        """Returns a contiguous tensor with the same data."""
        return Copy.apply(self)

    def __repr__(self) -> str:
        """Returns a string representation of the tensor."""
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        """Gets an item from the tensor based on the index."""
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        """Sets an item in the tensor at the given index."""
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    def _type_(self, backend: TensorBackend) -> None:
        """Sets the backend type for the tensor."""
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        """Creates a new tensor with the provided tensor data."""
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Creates a new tensor from the provided storage, shape, and optional strides."""
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """Expands the tensor to match the shape of the other tensor for broadcasting."""
        if self.shape == other.shape:
            return other
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(other, buf)
        if self.shape == true_shape:
            return buf
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        """Creates a tensor filled with zeros of the given shape."""

        def zero(shape: UserShape) -> Tensor:
            return Tensor.make(
                [0.0] * int(operators.prod(shape)), shape, backend=self.backend
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Returns the tensor's storage, shape, and strides as a tuple."""
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        """Returns a new tensor detached from the computation graph."""
        return Tensor(self._tensor, backend=self.backend)

    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative on this tensor (only for leaf variables)."""
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(operators.prod(self.shape)),
                self.shape,
                backend=self.backend,
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """Checks if the tensor is a leaf (created by the user)."""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Checks if the tensor is constant (no history)."""
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Returns the parent tensors involved in the computation."""
        assert self.history is not None
        return self.history.inputs

    @property
    def size(self) -> int:
        """Returns the size of the tensor."""
        return self._tensor.size

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Implements the chain rule for backpropagation."""
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        return [
            (inp, inp.expand(self._ensure_tensor(d_in)))
            for inp, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        """Performs backpropagation to compute the gradient."""
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
        backpropagate(self, grad_output)

    def __truediv__(self, b: TensorLike) -> Tensor:
        """Implements the division of two tensors."""
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        """Implements reverse division (scalar / tensor)."""
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Implements matrix multiplication."""
        return MatMul.apply(self, b)

    @property
    def shape(self) -> UserShape:
        """Returns the shape of the tensor."""
        return self._tensor.shape

    def zero_grad_(self) -> None:
        """Resets the gradient of the tensor to None."""
        self.grad = None

    def __add__(self, b: TensorLike) -> Tensor:
        """Implements addition of two tensors."""
        return Add.apply(self, self._ensure_tensor(b))

    def __sub__(self, b: TensorLike) -> Tensor:
        """Implements subtraction of two tensors."""
        return Add.apply(self, -self._ensure_tensor(b))

    def __mul__(self, b: TensorLike) -> Tensor:
        """Implements multiplication of two tensors."""
        return Mul.apply(self, self._ensure_tensor(b))

    def __lt__(self, b: TensorLike) -> Tensor:
        """Implements the less-than comparison of two tensors."""
        return LT.apply(self, self._ensure_tensor(b))

    def __eq__(self, b: TensorLike) -> Tensor:  # type: ignore[override]
        """Implements equality comparison between two tensors."""
        return EQ.apply(self, self._ensure_tensor(b))

    def __gt__(self, b: TensorLike) -> Tensor:
        """Implements the greater-than comparison of two tensors."""
        return LT.apply(self._ensure_tensor(b), self)

    def __neg__(self) -> Tensor:
        """Implements negation of a tensor."""
        return Neg.apply(self)

    def __radd__(self, b: TensorLike) -> Tensor:
        """Implements reverse addition (scalar + tensor)."""
        return self + b

    def __rmul__(self, b: TensorLike) -> Tensor:
        """Implements reverse multiplication (scalar * tensor)."""
        return self * b

    def all(self, dim: Optional[int] = None) -> Tensor:
        """Checks if all elements along the specified dimension evaluate to True."""
        if dim is None:
            return All.apply(self.view(self.size), self._ensure_tensor(0))
        else:
            return All.apply(self, self._ensure_tensor(dim))

    def is_close(self, y: Tensor) -> Tensor:
        """Checks if two tensors are element-wise close within a tolerance."""
        return IsClose.apply(self, y)

    def sigmoid(self) -> Tensor:
        """Applies the sigmoid activation function to the tensor."""
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        """Applies the ReLU activation function to the tensor."""
        return ReLU.apply(self)

    def log(self) -> Tensor:
        """Applies the natural logarithm element-wise to the tensor."""
        return Log.apply(self)

    def exp(self) -> Tensor:
        """Applies the exponential function element-wise to the tensor."""
        return Exp.apply(self)

    def sum(self, dim: Optional[int] = None) -> Tensor:
        """Computes the sum over the specified dimension."""
        if dim is None:
            return Sum.apply(self.contiguous().view(self.size), self._ensure_tensor(0))
        else:
            return Sum.apply(self, self._ensure_tensor(dim))

    def mean(self, dim: Optional[int] = None) -> Tensor:
        """Computes the mean over the specified dimension."""
        if dim is not None:
            return self.sum(dim) / self.shape[dim]
        else:
            return self.sum() / self.size

    def permute(self, *order: int) -> Tensor:
        """Permutes the dimensions of the tensor according to the specified order."""
        return Permute.apply(self, tensor(list(order)))

    def view(self, *shape: int) -> Tensor:
        """Reshapes the tensor to the specified shape while keeping the same data."""
        return View.apply(self, tensor(list(shape)))
