from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    return (
        f(*vals[:arg], vals[arg] + epsilon, *vals[arg + 1 :])
        - f(*vals[:arg], vals[arg] - epsilon, *vals[arg + 1 :])
    ) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative value `x` for this variable."""
        ...

    @property
    def unique_id(self) -> int:
        """Returns the unique identifier for this variable."""
        ...

    def is_leaf(self) -> bool:
        """Returns whether this variable is a leaf node (i.e., no parents)."""
        ...

    def is_constant(self) -> bool:
        """Returns whether this variable represents a constant value."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns an iterable of the parent variables of this variable in the computation graph."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to propagate the derivative through the computational graph.

        Returns the gradient of the inputs with respect to the output.
        """
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    li = []
    visited = set()

    def visit(var: Variable) -> None:
        """Helper function that performs a depth-first search (DFS)
        on the computational graph to achieve topological sorting.

        Args:
        ----
        var (Variable): A variable in the grpah to be visited.

        """
        # if we have already visited var or var is constant return
        if var.unique_id in visited or var.is_constant():
            return
        # else we visit each of the var's parents
        for par_var in var.parents:
            if not par_var.is_constant():
                visit(par_var)
        # After all of the parents of var have been added, we then add var
        visited.add(var.unique_id)
        li.append(var)

    visit(variable)

    # However since in backwards pass the graph's directed edges are reversed, we actually
    # want to reverse the order of this list
    return li[::-1]


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable (Variable): The right-most variable
        deriv (Any): Its derivative that we want to propagate backward to the leaves.

    Returns:
    -------
        No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    li = topological_sort(variable)
    derivatives = {}
    derivatives[variable.unique_id] = deriv

    for var in li:
        d_output = derivatives[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(d_output)
        else:
            for v_in, d_in in var.chain_rule(d_output):
                if v_in.unique_id not in derivatives:
                    derivatives[v_in.unique_id] = 0
                derivatives[v_in.unique_id] = derivatives[v_in.unique_id] + d_in


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved values (tensors) stored during the forward pass."""
        return self.saved_values
