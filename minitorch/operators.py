"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable


def mul(x: float, y: float) -> float:
    """Multiplies two numbers and returns their product.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The product of x and y.

    """
    return x * y


def id(x: float) -> float:
    """Returns the input unchanged.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The same input number.

    """
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers and returns their sum.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The sum of x and y.

    """
    return x + y


def neg(x: float) -> float:
    """Returns the negation of the input number.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The negation of x.

    """
    return -1 * x


def lt(x: float, y: float) -> bool:
    """Checks if the first number is less than the second number.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        bool: True if x is less than y, False otherwise.

    """
    return x < y


def eq(x: float, y: float) -> bool:
    """Checks if two numbers are equal.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        bool: True if x is equal to y, False otherwise.

    """
    return x == y


def max(x: float, y: float) -> float:
    """Returns the maximum of two numbers.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        float: The maximum of x and y.

    """
    if x >= y:
        return x
    return y


def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close to each other within a small tolerance.

    Args:
    ----
        x (float): The first number.
        y (float): The second number.

    Returns:
    -------
        bool: True if the absolute difference between x and y is less than 1e-2, False otherwise.

    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Computes the sigmoid function for the input number.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The sigmoid of x.

    """
    if x >= 0:
        return 1 / (1 + math.exp(-1 * x))
    else:
        a = math.exp(x)
        return a / (1 + a)


def relu(x: float) -> float:
    """Applies the ReLU (Rectified Linear Unit) function.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: x if x is greater than 0, otherwise 0.

    """
    if x < 0:
        return 0.0
    return x


def log(x: float) -> float:
    """Computes the natural logarithm of the input number.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The natural logarithm of x.

    """
    return math.log(x)


def exp(x: float) -> float:
    """Computes the exponential of the input number.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The exponential of x.

    """
    return math.exp(x)


def inv(x: float) -> float:
    """Computes the multiplicative inverse of the input number.

    Args:
    ----
        x (float): The input number.

    Returns:
    -------
        float: The multiplicative inverse of x.

    """
    return 1 / x


def log_back(x: float, y: float) -> float:
    """Computes the backward gradient of the logarithm function.

    Args:
    ----
        x (float): The point at which to compute the gradient.
        y (float): The gradient with respect to the output.

    Returns:
    -------
        float: The gradient with respect to the input.

    """
    return y / x


def inv_back(x: float, y: float) -> float:
    """Computes the backward gradient of the inverse function.

    Args:
    ----
        x (float): The point at which to compute the gradient.
        y (float): The gradient with respect to the output.

    Returns:
    -------
        float: The gradient with respect to the input.

    """
    return (-1 / x**2) * y


def relu_back(x: float, y: float) -> float:
    """Computes the backward gradient of the ReLU function.

    Args:
    ----
        x (float): The input number.
        y (float): The gradient with respect to the output.

    Returns:
    -------
        float: The gradient with respect to the input if x > 0, otherwise 0.

    """
    if x > 0:
        return y
    else:
        return 0


def map(func: Callable, li: Iterable) -> Iterable:
    """Applies a given function to each element in a list and returns a new list with the results.

    Args:
    ----
        func (Callable): A function that takes a single argument and returns a value.
        li (Iterable): An iterable of elements to which the function will be applied.

    Returns:
    -------
        Iterable: A new iterable containing the results of applying `func` to each element in `li`.

    """
    res = []
    for x in li:
        res.append(func(x))
    return res


def zipWith(
    li_1: Iterable[float], li_2: Iterable[float]
) -> Iterable[tuple[float, float]]:
    """Combines two iterables element-wise into a list of tuples.

    Args:
    ----
        li_1 (Iterable[float]): The first iterable.
        li_2 (Iterable[float]): The second iterable.

    Returns:
    -------
        list[tuple[float, float]]: An iterable of tuples, where each tuple contains elements from the corresponding positions
        of li_1 and li_2.

    """
    ls_1 = [el for el in li_1]
    res = []
    for index, el in enumerate(li_2):
        res.append((ls_1[index], el))

    return res


def reduce(func: Callable, li: Iterable[float], start: float) -> float:
    """Reduces an iterable to a single value by applying a binary function cumulatively to its elements.

    Args:
    ----
        func (Callable): A function that takes two arguments and returns a single value.
        li (Iterable[float]): An iterable to be reduced.
        start (float): The initial value to be used with the first element in the iterable.

    Returns:
    -------
        float: A single value obtained by cumulatively applying `func` to the elements of `li`.

    """
    pending = start
    for el in li:
        pending = func(pending, el)
    return pending


def negList(li: Iterable[float]) -> Iterable[float]:
    """Applies negation to each element in an Iterable of floats.

    Args:
    ----
        li (Iterable[float]): A list of float numbers.

    Returns:
    -------
        Iterable[float]: A new Iterable where each element is the negation of the corresponding element in `li`.

    """
    return map(lambda x: neg(x), li)


def addLists(li_1: Iterable[float], li_2: Iterable[float]) -> Iterable[float]:
    """Adds corresponding elements of two Iterables together.

    Args:
    ----
        li_1 (Iterable[float]): The first Iterable of float numbers.
        li_2 (Iterable[float]): The second Iterable of float numbers.

    Returns:
    -------
        Iterable[float]: A new Iterable where each element is the sum of the corresponding elements in `li_1` and `li_2`.

    """
    zipped_lists = zipWith(li_1, li_2)
    return map(lambda tup: tup[0] + tup[1], zipped_lists)


def sum(li: Iterable[float]) -> float:
    """Calculates the sum of all elements in a list of floats.

    Args:
    ----
        li (Iterable[float]): An Iterable of float numbers.

    Returns:
    -------
        float: The sum of all elements in `li`.

    """
    return reduce(lambda x, y: x + y, li, 0)


def prod(li: Iterable[float]) -> float:
    """Calculates the product of all elements in an Iterable of floats.

    Args:
    ----
        li (Iterable[float]): An Iterable of float numbers.

    Returns:
    -------
        float: The product of all elements in `li`.

    """
    product = 1
    for i in li:
        product *= i
    return product
