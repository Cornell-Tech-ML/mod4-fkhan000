from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba
import numba.cuda
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias


MAX_DIMS = 32


class IndexingError(RuntimeError):
    """Exception raised for indexing errors."""

    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """Convert a multi-dimensional index to a single position based on strides."""
    pos = 0
    for a, b in zip(index, strides):
        pos += a * b
    return pos


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """Convert an ordinal value to a multi-dimensional index."""
    cur_ord = ordinal + 0
    for i in range(len(shape) - 1, -1, -1):
        sh = shape[i]
        out_index[i] = int(cur_ord % sh)
        cur_ord = cur_ord // sh


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """Map a large tensor index to a smaller one following broadcasting rules."""
    # If the smaller shape has fewer dimensions then the extra ones to the left that
    # the bigger shape has can be ignored since by broadcasting we would just repeat the smaller
    # along those extra dimensions to get the dimensions to match
    start = big_shape.size - shape.size

    # for each dimension in the smaller shape
    for i in range(shape.size):
        # we fill in out_index with big_index[i+start] when the dims between shape and big_shape agree
        # but if shape[i] is 1 then we just use 0 since broadcasting had made the shape repeat along that
        # dimension so we just access the first element in that dimension
        out_index[i] = big_index[i + start] if shape[i] != 1 else 0


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """Broadcast two shapes to a compatible shape."""
    if len(shape2) > len(shape1):
        shape1, shape2 = shape2, shape1
    shape = list(shape1)
    for i in range(1, len(shape2) + 1):
        if shape1[-i] == 1 or shape2[-i] == 1:
            shape[-i] = max(shape1[-i], shape2[-i])
        elif shape1[-i] != shape2[-i]:
            raise IndexingError(f"Cannot broadcast shape {shape1} with shape {shape2}")
    return tuple(shape)


def strides_from_shape(shape: UserShape) -> UserStrides:
    """Return the strides for a given shape."""
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        """Initialize tensor with storage, shape, and optional strides."""
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.shape = shape
        size = 1
        for i in shape:
            size *= i
        self.size = size
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        """Convert tensor storage to CUDA."""
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """Check if the tensor is stored contiguously in memory."""
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        """Broadcast two shapes to a compatible shape."""
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        """Convert a multi-dimensional index to a single-dimensional index."""
        if isinstance(index, int):
            aindex: Index = array([index])
        else:
            aindex = array(index)

        shape = self.shape
        if len(shape) == 0 and len(aindex) != 0:
            shape = (1,)

        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        """Yield all possible indices for the tensor."""
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        """Return a random valid index."""
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        """Get the value at a specific index."""
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        """Set a value at a specific index."""
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Return the core tensor data as a tuple."""
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """Permute the tensor dimensions."""
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        shape = tuple([self.shape[i] for i in order])
        strides = tuple([self.strides[i] for i in order])
        return TensorData(self._storage, shape, strides)

    def to_string(self) -> str:
        """Return a string representation of the tensor."""
        s = ""
        for index in self.indices():
            m = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    m = "\n%s[" % ("\t" * i) + m
                else:
                    break
            s += m
            v = self.get(index)
            s += f"{v:3.2f}"
            m = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    m += "]"
                else:
                    break
            if m:
                s += m
            else:
                s += " "
        return s
