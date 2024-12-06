# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: dict[str, Any]) -> Fn:
    """JIT-compile a function for device execution (e.g., GPU).

    Args:
    ----
        fn (Fn): The function to compile.
        **kwargs (dict[str, Any]): Additional options for JIT compilation.

    Returns:
    -------
        Fn: Compiled function optimized for device execution.

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: dict[str, Any]) -> FakeCUDAKernel:
    """JIT-compile a function for general execution.

    Args:
    ----
        fn (Fn): The function to compile.
        **kwargs (dict[str, Any]): Additional options for JIT compilation.

    Returns:
    -------
        FakeCUDAKernel: Compiled function.

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Applies a binary function element-wise to two tensors using JIT-compiled execution.

        This method takes a function that operates on two scalar values and applies it to two tensors,
        broadcasting their shapes as needed. The function is JIT-compiled for device execution to improve performance.

        Args:
        ----
            fn (Callable[[float, float], float]): A binary function that takes two floats as input
                and returns a float as output.

        Returns:
        -------
            Callable[[Tensor, Tensor], Tensor]: A function that takes two tensors as input, applies the binary function
                element-wise, and returns the resulting tensor.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Reduces a tensor along a specified dimension using a binary function.

        Args:
        ----
            fn (Callable[[float, float], float]): Binary function for reduction (e.g., sum, max).
            start (float, optional): Initial value for the reduction. Defaults to 0.0.

        Returns:
        -------
            Callable[[Tensor, int], Tensor]: Function to reduce a tensor along a dimension.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Performs matrix multiplication on two tensors, supporting broadcasting and 2D/3D tensors.

        Args:
        ----
            a (Tensor): The first input tensor.
            b (Tensor): The second input tensor. Its second-to-last dimension must match
                the last dimension of `a`.

        Returns:
        -------
            Tensor: The resulting tensor from the matrix multiplication. If both inputs
            are 2D, the output will also be 2D.

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

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

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
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # we allocate arrays to store the multidim indices for in and out
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        # we get the thread's position in the grid
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # and use it to parametrize our parallel operation

        # we set up our guard
        if i < out_size:
            # and then perform the map with the corresponding index in out
            # we get the multidim index in out
            to_index(i, out_shape, out_index)
            # broadcast it to get the corresponding index in in
            broadcast_index(out_index, out_shape, in_shape, in_index)
            # access it
            x = in_storage[index_to_position(in_index, in_strides)]
            # and apply fn at this value and feed it into out
            out[i] = fn(x)

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # allocate arrays to store out and in multidim index
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        # our thread's position in grid
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # guard
        if i < out_size:
            # we obtain the multidim index
            to_index(i, out_shape, out_index)
            # broadcast it to the corresponding indices in a and b
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            # and then get the values of a and b at those indices
            x_a = a_storage[index_to_position(a_index, a_strides)]
            x_b = b_storage[index_to_position(b_index, b_strides)]
            out_ordinal = index_to_position(out_index, out_strides)
            # finally we fill in the output index with fn(x_a, x_b)
            out[out_ordinal] = fn(x_a, x_b)

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """A practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // blockDIM
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    # we set our shared cache (shared memory within a block)
    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    # and set out thread's position in the grid and block
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # if i is less than size
    if i < size:
        # we copy in a[i] into our cache
        cache[pos] = a[i]
    else:
        # the last block might not have 32 entries so we'll set this value to 0
        cache[pos] = 0.0
    cuda.syncthreads()
    # at this point each block has a shared array containing a 32 sized chunk
    # of the array a

    # to parallelize the sum, we will have every even thread in the block
    # sum the corresponding value in cache and the value in the slot to the right of it
    # after this, we will have every thread divisible by four sum up itself and the
    # value in the slot that's 2 spaces away from them and so on.
    # In the end we will have the summed up value in cache[0].

    # for each round of summing
    n = 1
    while n < BLOCK_DIM:
        # if your position in the block is divisible by 2**n, sum
        # the person that's n slots away from you
        if pos % (2 * n) == 0:
            cache[pos] += cache[pos + n]
            # we then update the shared array for all threads
            cuda.syncthreads()
        n *= 2
    # finally we have one of the threads in each block assign cache[0]
    # to out
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Computes the sum of a tensor's elements using a GPU-accelerated approach.

    Args:
    ----
        a (Tensor): The input tensor to be summed.

    Returns:
    -------
        TensorData: A TensorData object containing the result of the sum operation.

    """
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024

        # to parallelize this, we have each block work on
        # reducing into a single position in out
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        # we by default set cache[pos] to the initial value given to us
        # (in sum this was 0)
        cache[pos] = reduce_value

        if out_pos < out_size:
            # we get the multidim index for out_pos
            to_index(out_pos, out_shape, out_index)

            # we then map each thread to an ordinal position in a

            out_index[reduce_dim] = out_index[reduce_dim] * BLOCK_DIM + cuda.threadIdx.x
            a_ordinal = index_to_position(out_index, a_strides)

            # we add a guard to make sure we don't cause an index out of bounds error
            if out_index[reduce_dim] < a_shape[reduce_dim]:
                # store the corresponding value in cache
                cache[pos] = a_storage[a_ordinal]
                cuda.syncthreads()
                # and now we have mapped a chunk of a into shared memory and we want to reduce
                # this to a single value
                # just as with sum we have half of the threads in the block apply the function
                # to themselves and their neighbor and then have a quarter of the threads in the block
                # apply it to themselves and the values corresponding to the threads that went through
                # the first round
                n = 1

                while n < BLOCK_DIM:
                    if pos % (2 * n) == 0:
                        cache[pos] = fn(cache[pos], cache[pos + n])
                        cuda.syncthreads()
                    n *= 2
            # we then have one thread in our block
            # assign the reduced value to the corresponding value in out
            if pos == 0:
                out[out_pos] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """A practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32
    # since we have enough shared memory to store two 32x32 matrices
    # and our input matrices take up less space than that
    # we can copy in a and b entirely into shared memory and have each
    # thread individually compute the i,j entry of the output matrix

    a_cache = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_cache = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # position of thread in block
    local_i = cuda.threadIdx.x
    local_j = cuda.threadIdx.y

    if local_i < size and local_j < size:
        # copy in matrix into cache
        a_cache[local_i, local_j] = a[local_i * size + local_j]
        b_cache[local_i, local_j] = b[local_i * size + local_j]
        cuda.syncthreads()
        tmp = 0
        # each thread now calculates a single dot product between
        # local_ith row and the local_jth column or the local_ith
        # local_jth entry in the output matrix
        for k in range(size):
            tmp += a_cache[local_i, k] * b_cache[k, local_j]
        out[local_i * size + local_j] = tmp


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Performs matrix multiplication on two tensors using a GPU-accelerated approach.

    Args:
    ----
        a (Tensor): The first input tensor.
        b (Tensor): The second input tensor. Both tensors must have compatible shapes
            for matrix multiplication (i.e., `a.shape[-1]` must equal `b.shape[-2]`).

    Returns:
    -------
        TensorData: A TensorData object containing the result of the matrix multiplication.

    """
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    # each block is responsible for performing matrix multiply
    # between corresponding submatrices of a and b in the batch
    # and each thread computes a single value in each of these output matrices

    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    # we initialize the shared memory to store copies of the corresponding
    # pair of submatrices of a and b which we will call A and B
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j] calculated by each thread
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Let a_b and b_b be the batch dimension of a and b respectively. Then we have that:
    # (a_b, N, K) x (b_b, K, M) = (out_shape[0], N, M)

    N, M, K = a_shape[-2], b_shape[-1], a_shape[-1]

    tmp = 0.0
    # since we can't hold the entire matrix in shared memory
    # we will perform mini matrix multiplies with the submatrices
    # that we can load into memory and then add them up in the end
    # we break up matrices A and B  into submatrices of size
    # BLOCK_DIM by BLOCK_DIM
    for l in range(0, K, BLOCK_DIM):
        # we get the column index of A by offsetting pj, the y position of the thread
        # in the block by l, the pj column in the current submatrix that we're in.
        a_k = l + pj

        # we set our guards
        if l < N and a_k < K:
            # and then we convert our multidim index (batch, i, a_k) to its ordinal position
            # and then copy the value of a at this point into shared memory.
            # we use i as the row index since each thread is being used to compute
            # C[i, j], mean thread (i, j) is solely responsible for copying the ith
            # row and the jth column of A and B respectively into memory
            a_shared[pi, pj] = a_storage[
                batch * a_batch_stride + i * a_strides[1] + a_k * a_strides[2]
            ]
        # we then do the same for B
        b_k = l + pi

        if b_k < K and j < M:
            b_shared[pi, pj] = b_storage[
                batch * b_batch_stride + b_k * b_strides[1] + j * b_strides[2]
            ]
        # finally we then call syncthreads to update the shared memory so that all threads in
        # the block now have access to the copied submatrices of A and B
        cuda.syncthreads()

        # the [pi, pj] entry in our matrix product is
        # sum a_shared[pi, k] * b_shared[k, pj]
        # here we clculate the partial product
        # and store it in tmp. When we will have iterate through
        # all of the submatrices, tmp will hold c[i, j]
        for k in range(BLOCK_DIM):
            # since it's possible that K isn't a multiple of BLOCK_DIM
            # we should perform a boundary check to ensure we don't
            # cause an index out of bounds error
            if l + k < K:
                tmp += a_shared[pi, k] * b_shared[k, pj]
    # finally we again apply our guard
    if i < N and j < M:
        # and write to out the value of tmp which is c[i, j]
        # we convert (batch, i, j) to its ordinal position in out and
        # then write to out.
        out[batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]] = tmp

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
