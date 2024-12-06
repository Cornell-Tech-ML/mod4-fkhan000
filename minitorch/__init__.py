"""Neural Network Framework

This package provides a modular framework for building and testing neural networks, with core components for modules, operators, datasets, and testing utilities.

Modules
-------

- `autodiff`: Handles forward and backward passes for automatic differentiation.
- `datasets`: Utilities for loading and managing datasets.
- `module`: Base class for defining and managing model parameters.
- `operators`: Core mathematical operations for scalar computations.
- `optim`: Optimization algorithms like Stochastic Gradient Descent (SGD).
- `tensor`: Defines the core Tensor object and its operations for autodifferentiation.
- `tensor_ops`: Provides the core tensor operations, including mapping, zipping, and reducing functions.
- `tensor_data`: Handles the underlying data storage, indexing, and broadcasting for tensors.
- `tensor_functions`: Implements various mathematical functions used in tensor operations and autodifferentiation.
- `scalar`: Defines scalar variables and their roles in the computation graph.
- `scalar_functions`: Differentiable functions (e.g., Add, Mul, ReLU, Exp) used in computations.
- `testing`: Utilities for testing models and operations.
- `cuda_ops`: Provides the core tensor operations using a GPU.
- `fast_ops`: Provides the core tensor operations using numba (only CPU).
- `fast_conv`: Provides implementations for 1D and 2D convolution layers using numba (only CPU)
- `nn`: Implements a variety of nn layers such as max pool and dropout
"""

from .testing import MathTest, MathTestVariable  # type: ignore # noqa: F401,F403
from .datasets import *  # noqa: F401,F403
from .optim import *  # noqa: F401,F403
from .tensor import *  # noqa: F401,F403
from .testing import *  # noqa: F401,F403
from .nn import *  # noqa: F401,F403
from .fast_conv import *  # noqa: F401,F403
from .tensor_data import *  # noqa: F401,F403
from .tensor_functions import *  # noqa: F401,F403
from .tensor_ops import *  # noqa: F401,F403
from .scalar import *  # noqa: F401,F403
from .scalar_functions import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .autodiff import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .module import *  # noqa: F401,F403
from .autodiff import *  # noqa: F401,F403
from .tensor import *  # noqa: F401,F403
from .datasets import *  # noqa: F401,F403
from .testing import *  # noqa: F401,F403
from .optim import *  # noqa: F401,F403
from .tensor_ops import *  # noqa: F401,F403
from .fast_ops import *  # noqa: F401,F403
from .cuda_ops import *  # noqa: F401,F403
from . import fast_ops, cuda_ops  # noqa: F401,F403
