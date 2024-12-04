from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Applies a non-JIT (non-Just-In-Time) compilation to the given function, with the option
    to always inline the function.

    Args:
    ----
        fn (Fn): The function to be inlined and processed without JIT compilation.
        **kwargs (Any): Additional options for non-JIT compilation.

    Returns:
    -------
        Fn: The processed function with inlining and other specified options applied.

    """
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # if output and input tensors have the same shapes and strides
        if np.array_equal(out_strides, in_strides) and np.array_equal(
            out_shape, in_shape
        ):
            # we avoid indexing and simply apply f to each value in in_storage
            # and assign it to the corresponding value in out
            for i in prange(len(out)):
                out[i] = fn(in_storage[i])
        else:
            # for each index in the output
            for ordinal_pos in prange(len(out)):
                out_index = np.empty_like(out_shape, dtype=np.int32)
                in_index = np.empty_like(in_shape, dtype=np.int32)
                # we get the multidim index
                to_index(ordinal_pos, out_shape, out_index)
                # broadcast it to get the corresponding index in the input
                broadcast_index(out_index, out_shape, in_shape, in_index)
                # get the value of the input
                x = in_storage[index_to_position(in_index, in_strides)]
                # and then fill output with fn(x)
                out[ordinal_pos] = fn(x)

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # if a, b, and out are of the same shape and stride
        if (
            np.array_equal(out_strides, a_strides)
            and np.array_equal(out_strides, b_strides)
            and np.array_equal(out_shape, a_shape)
            and np.array_equal(out_shape, b_shape)
        ):
            # we avoid indexing and just functional zip the corresponding values in a and b
            # and add them to out
            for i in prange(len(out)):
                out[i] = fn(a_storage[i], b_storage[i])
        else:
            # for each index in the output
            for i in prange(len(out)):
                out_index = np.empty_like(out_shape, dtype=np.int32)
                a_index = np.empty_like(a_shape, dtype=np.int32)
                b_index = np.empty_like(b_shape, dtype=np.int32)
                # we obtain the multidim index
                to_index(i, out_shape, out_index)
                # broadcast it to the corresponding indices in a and b
                broadcast_index(out_index, out_shape, a_shape, a_index)
                broadcast_index(out_index, out_shape, b_shape, b_index)
                # and then get the values of a and b at those indices
                x_a = a_storage[index_to_position(a_index, a_strides)]
                x_b = b_storage[index_to_position(b_index, b_strides)]
                # finally we fill in the output index with fn(x_a, x_b)
                out[i] = fn(x_a, x_b)

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        for i in prange(len(out)):
            out_index = np.empty_like(out_shape, dtype=np.int32)
            # convert the ordinal position to the multi dim index
            to_index(i, out_shape, out_index)
            a_index = out_index.copy()
            # get initial starting point in a for reduction
            a_ordinal = index_to_position(a_index, a_strides)
            start = out[i]
            for j in prange(a_shape[reduce_dim]):
                # to reduce need for indexing, we simply increment using reduce_dim stride
                start = fn(a_storage[int(a_ordinal + j * a_strides[reduce_dim])], start)
            out[i] = start

    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    # Let b_a, b_b be the batch 'dimension' of a and b respectively
    # We have that:
    # (b_a, n, k) x (b_b, k, m) => (out_shape[0], n, m)

    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    N, K, M = a_shape[-2], a_shape[-1], b_shape[-1]

    for i in prange(out_shape[0]):
        for n in prange(N):
            for m in prange(M):
                # (i, n, m) is our multidim index for out so we can multiply that by strides
                # to get the ordinal position
                out_ordinal = (
                    i * out_strides[0] + n * out_strides[1] + m * out_strides[2]
                )
                tmp = 0
                for k in prange(K):
                    # and then same idea for a and b ordinal positions
                    a_ordinal = i * a_batch_stride + n * a_strides[1] + k * a_strides[2]
                    b_ordinal = i * b_batch_stride + k * b_strides[1] + m * b_strides[2]

                    # then update the value of out[out_ordinal]
                    # in this loop we are calculating the dot product between the nth "row" of
                    # a and the mth column of b
                    tmp += a_storage[a_ordinal] * b_storage[b_ordinal]
                out[out_ordinal] = tmp


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
