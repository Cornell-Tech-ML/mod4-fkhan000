from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        """Computes the backward pass by calling the backward function and wrapping the result in a tuple.

        Args:
        ----
            ctx (Context): The context object that holds information saved during the forward pass.
            d_out (float): The derivative of the output with respect to the function.

        Returns:
        -------
            Tuple[float, ...]: The gradients of the input variables with respect to the output.

        """
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        """Computes the forward pass by calling the forward function with the given inputs.

        Args:
        ----
            ctx (Context): The context object to save values for the backward pass.
            *inps (float): The input scalar values for the forward function.

        Returns:
        -------
            float: The result of the forward computation.

        """
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Applies the function to the given scalar-like input values and returns a Scalar.

        This method extracts the raw float values from the Scalar objects, calls the forward function,
        and creates a new Scalar object with a history which is used for backpropagation.

        Args:
        ----
            *vals (ScalarLike): The scalar-like input values, which can be either Scalar objects or floats.

        Returns:
        -------
            Scalar: A new Scalar object representing the result of the function with backpropagation history.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


class Add(ScalarFunction):
    """Addition function that performs element-wise addition: f(x, y) = x + y.

    This function adds two scalars `a` and `b`.
    """

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass of the addition operation."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the backward pass of the addition operation, returning gradients."""
        return d_output, d_output


class Log(ScalarFunction):
    """Logarithm function: f(x) = log(x).

    This function computes the natural logarithm of a scalar `a`.
    """

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of the logarithm operation."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass of the logarithm operation, returning the gradient."""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function: f(x, y) = x * y.

    This function multiplies two scalars `a` and `b`.
    """

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass of the multiplication operation."""
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the backward pass of the multiplication operation, returning gradients."""
        a, b = ctx.saved_values
        return d_output * b, d_output * a


class Exp(ScalarFunction):
    """Exponential function: f(x) = exp(x).

    This function computes the exponential of a scalar `a`.
    """

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of the exponential operation."""
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass of the exponential operation, returning the gradient."""
        (a,) = ctx.saved_values
        return d_output * operators.exp(a)


class Neg(ScalarFunction):
    """Negation function: f(x) = -x.

    This function negates the scalar `a`.
    """

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of the negation operation."""
        return -1 * a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass of the negation operation, returning the gradient."""
        return d_output * -1


class ReLU(ScalarFunction):
    """ReLU (Rectified Linear Unit) function: f(x) = max(0, x).

    This function computes the ReLU of a scalar `a`.
    """

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of the ReLU operation."""
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass of the ReLU operation, returning the gradient."""
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Sigmoid(ScalarFunction):
    """Sigmoid function: f(x) = 1 / (1 + exp(-x)).

    This function computes the sigmoid of a scalar `a`.
    """

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of the sigmoid operation."""
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass of the sigmoid operation, returning the gradient."""
        (a,) = ctx.saved_values
        return operators.sigmoid(a) * (1 - operators.sigmoid(a)) * d_output


class Inv(ScalarFunction):
    """Inverse function: f(x) = 1 / x.

    This function computes the inverse of a scalar `a`.
    """

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of the inverse operation."""
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass of the inverse operation, returning the gradient."""
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class LT(ScalarFunction):
    """Less-than function: f(x, y) = 1 if x < y else 0.

    This function compares two scalars `a` and `b`.
    """

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass of the less-than operation."""
        return 1.0 if operators.lt(a, b) else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the backward pass of the less-than operation. Always returns zero as gradients."""
        return 0, 0


class EQ(ScalarFunction):
    """Equality function: f(x, y) = 1 if x == y else 0.

    This function checks if two scalars `a` and `b` are equal.
    """

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass of the equality operation."""
        return 1.0 if operators.eq(a, b) else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the backward pass of the equality operation. Always returns zero as gradients."""
        return 0, 0
