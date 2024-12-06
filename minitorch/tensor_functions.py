from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a value into a tuple if it is not already a tuple.

    Args:
    ----
        x: Any input value.

    Returns:
    -------
        Tuple: A tuple version of the input value.

    """
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        """Perform the backward pass and return the gradient.

        Args:
        ----
            ctx: Context storing intermediate values for backpropagation.
            grad_out: Tensor representing the gradient from the forward pass.

        Returns:
        -------
            Tuple[Tensor, ...]: Gradients with respect to each input.

        """
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        """Perform the forward pass.

        Args:
        ----
            ctx: Context storing intermediate values for backpropagation.
            inps: Inputs to the forward function.

        Returns:
        -------
            Tensor: The result of the forward computation.

        """
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Apply the forward function and track history for autodiff.

        Args:
        ----
            vals: Tensors to apply the function to.

        Returns:
        -------
            Tensor: Result of the forward computation with history tracking.

        """
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        ctx = Context(not need_grad)

        c = cls._forward(ctx, *raw_vals)

        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Perform the forward pass for negation."""
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Perform the backward pass for negation."""
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Perform the forward pass for inversion."""
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Perform the backward pass for inversion."""
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Perform the forward pass for addition."""
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Perform the backward pass for addition."""
        return grad_output, grad_output


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Return 1 if all elements are true in the tensor along the dimension."""
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Perform the forward pass for multiplication."""
        ctx.save_for_backward(a, b)
        return a.f.mul_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Perform the backward pass for multiplication."""
        (a, b) = ctx.saved_values
        # just as with scalar multiplication we multiply b and a with grad output
        return grad_output.f.mul_zip(b, grad_output), grad_output.f.mul_zip(
            a, grad_output
        )


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Perform the forward pass for the sigmoid activation function."""
        sigmoid_t1 = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(sigmoid_t1)
        return sigmoid_t1

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Perform the backward pass for the sigmoid function."""
        (sigmoid_t1,) = ctx.saved_values
        deriv_sigmoid = sigmoid_t1.f.neg_map(sigmoid_t1)
        deriv_sigmoid = deriv_sigmoid.f.add_zip(deriv_sigmoid, tensor([1]))
        return grad_output.f.mul_zip(
            grad_output, grad_output.f.mul_zip(sigmoid_t1, deriv_sigmoid)
        )


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Perform the forward pass for the ReLU activation function."""
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Perform the backward pass for the ReLU function."""
        (t1,) = ctx.saved_values
        return grad_output.f.relu_back_zip(t1, grad_output)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Perform the forward pass for the logarithm function."""
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Perform the backward pass for the logarithm function."""
        (t1,) = ctx.saved_values
        return grad_output.f.log_back_zip(t1, grad_output)


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Perform the forward pass for summing along a dimension."""
        return a.f.add_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Perform the backward pass for summing."""
        return grad_output, 0.0


class LT(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Perform the forward pass for less-than comparison."""
        return a.f.lt_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Perform the backward pass for less-than comparison."""
        return grad_output.zeros(), grad_output.zeros()


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Perform the forward pass for equality comparison."""
        return a.f.eq_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Perform the backward pass for equality comparison."""
        return grad_output.zeros(), grad_output.zeros()


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Perform the forward pass to check if tensors are close element-wise."""
        return a.f.is_close_zip(a, b)


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        """Perform the forward pass for permuting the tensor's dimensions."""
        ord = order.to_numpy().astype(int).tolist()
        ctx.save_for_backward(ord)
        return a._new(a._tensor.permute(*ord))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Perform the backward pass for permuting the tensor's dimensions."""
        (ord,) = ctx.saved_values
        ord = sorted(ord, key=lambda x: ord[x])
        return grad_output._new(grad_output._tensor.permute(*ord)), 0.0


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Perform the forward pass for the exponential function."""
        exp_t1 = t1.f.exp_map(t1)
        ctx.save_for_backward(exp_t1)
        return exp_t1

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Perform the backward pass for the exponential function."""
        (exp_t1,) = ctx.saved_values
        return grad_output.f.mul_zip(grad_output, exp_t1)


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Perform the forward pass for reshaping the tensor."""
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape._tensor.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Perform the backward pass for reshaping the tensor."""
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Perform the forward pass for copying the tensor."""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Perform the backward pass for copying the tensor."""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Perform the forward pass for matrix multiplication."""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Perform the backward pass for matrix multiplication."""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a._tensor.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Create a tensor filled with zeros of the specified shape.

    Args:
    ----
        shape: Shape of the tensor.
        backend: Backend to be used for the tensor (optional).

    Returns:
    -------
        Tensor: A tensor filled with zeros.

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Create a tensor filled with random values of the specified shape.

    Args:
    ----
        shape: Shape of the tensor.
        backend: Backend to be used for the tensor (optional).
        requires_grad: Whether the tensor requires gradient computation (optional).

    Returns:
    -------
        Tensor: A tensor filled with random values.

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Create a tensor with specified data and shape.

    Args:
    ----
        ls: Data to be used for the tensor.
        shape: Shape of the tensor.
        backend: Backend to be used for the tensor (optional).
        requires_grad: Whether the tensor requires gradient computation (optional).

    Returns:
    -------
        Tensor: A tensor with the specified data and shape.

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Create a tensor with data and automatically inferred shape.

    Args:
    ----
        ls: Data to be used for the tensor.
        backend: Backend to be used for the tensor (optional).
        requires_grad: Whether the tensor requires gradient computation (optional).

    Returns:
    -------
        Tensor: A tensor with the specified data and shape.

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Compute the gradient using central difference.

    Args:
    ----
        f: Function to compute the gradient for.
        vals: Input tensors.
        arg: Index of the argument to differentiate with respect to (optional).
        epsilon: Small value for numerical differentiation (optional).
        ind: Index at which to compute the gradient.

    Returns:
    -------
        float: Central difference gradient at the specified index.

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether the autodiff gradients match central difference gradients.

    Args:
    ----
        f: Function to check the gradients for.
        vals: Input tensors.

    """
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
