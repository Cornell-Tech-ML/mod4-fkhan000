from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

from dataclasses import field
from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    EQ,
    LT,
    Add,
    Exp,
    Inv,
    Log,
    Mul,
    Neg,
    ReLU,
    ScalarFunction,
    Sigmoid,
)

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """`ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes
    ----------
        last_fn : The last Function that was called.
        ctx : The context for that Function.
        inputs : The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()


# ## Task 1.2 and 1.4
# Scalar Forward and Backward

_var_count = 0


@dataclass
class Scalar:
    """A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.
    """

    data: float
    history: Optional[ScalarHistory] = field(default_factory=ScalarHistory)
    derivative: Optional[float] = None
    name: str = field(default="")
    unique_id: int = field(default=0)

    def __post_init__(self):
        global _var_count
        _var_count += 1
        object.__setattr__(self, "unique_id", _var_count)
        object.__setattr__(self, "name", str(self.unique_id))
        object.__setattr__(self, "data", float(self.data))

    def __repr__(self) -> str:
        """String representation of the Scalar object."""
        return f"Scalar({self.data})"

    def __mul__(self, b: ScalarLike) -> Scalar:
        """Performs scalar multiplication."""
        return Mul.apply(self, b)

    def __add__(self, b: ScalarLike) -> Scalar:
        """Performs scalar addition."""
        return Add.apply(self, b)

    def __truediv__(self, b: ScalarLike) -> Scalar:
        """Performs scalar division."""
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        """Performs scalar reverse division."""
        return Mul.apply(b, Inv.apply(self))

    def __bool__(self) -> bool:
        """Returns the boolean value of the scalar."""
        return bool(self.data)

    def __radd__(self, b: ScalarLike) -> Scalar:
        """Performs reverse scalar addition."""
        return self + b

    def __rmul__(self, b: ScalarLike) -> Scalar:
        """Performs reverse scalar multiplication."""
        return self * b

    def __neg__(self) -> Scalar:
        """Returns the negation of the scalar."""
        return Neg.apply(self)

    def __sub__(self, b: ScalarLike) -> Scalar:
        """Performs scalar subtraction."""
        return self + (-b)

    def __eq__(self, b: ScalarLike) -> Scalar:
        """Performs scalar equality check."""
        return EQ.apply(self, b)

    def __lt__(self, b: ScalarLike) -> Scalar:
        """Performs scalar less-than comparison."""
        return LT.apply(self, b)

    def __gt__(self, b: ScalarLike) -> Scalar:
        """Performs scalar greater-than comparison."""
        return LT.apply(b, self)

    def relu(self) -> Scalar:
        """Applies the ReLU function to the scalar."""
        return ReLU.apply(self)

    def exp(self) -> Scalar:
        """Applies the exponential function to the scalar."""
        return Exp.apply(self)

    def log(self) -> Scalar:
        """Applies the logarithm function to the scalar."""
        return Log.apply(self)

    def sigmoid(self) -> Scalar:
        """Applies the sigmoid function to the scalar."""
        return Sigmoid.apply(self)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x: value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.__setattr__("derivative", 0.0)
        self.__setattr__("derivative", self.derivative + x)

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Checks if the Scalar is constant; in other words whether or not it is a function
        of some other input.
        """
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Returns the parent variables of the current variable.
        This function retrieves the inputs that were used to create this variable,
        which are stored in its history.
        """
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to propagate the gradient through the computational graph.
        This function retrieves the function, context, and inputs from the variable's history
        and propagates the derivative `d_output` through the function using the chain rule.
        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        derivatives = h.last_fn._backward(h.ctx, d_output)
        propagate = []
        for index, input in enumerate(h.inputs):
            if not input.is_constant():
                propagate.append((input, derivatives[index]))

        return propagate

    def backward(self, d_output: Optional[float] = None) -> None:
        """Calls autodiff to fill in the derivatives for the history of this object.

        Args:
        ----
            d_output (number, opt): starting derivative to backpropagate through the model
                                   (typically left out, and assumed to be 1.0).

        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """Checks that autodiff works on a python function.
    Asserts False if derivative is incorrect.

    Args:
    ----
        f : function from n-scalars to 1-scalar.
        *scalars  : n input scalar values.

    """
    out = f(*scalars)
    out.backward()

    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        print(str([x.data for x in scalars]), x.derivative, i, check)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )
