import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generates a list of N random 2D points within the range [0, 1].

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
    List[Tuple[float, float]]: A list of tuples, each representing a 2D point (x_1, x_2) with random coordinates.

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    """A dataclass representing a graph with a set of points and corresponding labels.

    Attributes
    ----------
        N (int): The number of points in the graph.
        X (List[Tuple[float, float]]): A list of tuples representing 2D points.
    y (List[int]): A list of integer labels associated with each point.

    """

    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Generates a graph where the label is 1 if the x-coordinate of the point is less than 0.5, else 0.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
    Graph: A graph containing N points and their corresponding labels based on the simple rule.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Generates a graph where the label is 1 if the sum of the x and y coordinates of a point is less than 0.5, else 0.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A graph containing N points and their corresponding labels based on the diagonal rule.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Generates a graph where the label is 1 if the x-coordinate of the point is less than 0.2 or greater than 0.8, else 0.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A graph containing N points and their corresponding labels based on the split rule.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Generates a graph where the label is 1 if the point falls in opposite quadrants, simulating an XOR pattern, else 0.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A graph containing N points and their corresponding labels based on the XOR rule.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if (x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5) else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Generates a graph where the label is 1 if the point is outside a circle of radius sqrt(0.1) centered at (0.5, 0.5), else 0.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A graph containing N points and their corresponding labels based on the circle rule.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = x_1 - 0.5, x_2 - 0.5
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Generates a graph of points arranged in a spiral pattern with alternating labels.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
    Graph: A graph containing N points and their corresponding labels based on the spiral pattern.

    """

    def x(t: float) -> float:
        """Computes the x-coordinate for a point in a spiral based on the parameter t."""
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        """Computes the y-coordinate for a point in a spiral based on the parameter t."""
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
