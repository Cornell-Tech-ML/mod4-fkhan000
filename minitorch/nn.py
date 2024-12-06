from typing import Tuple
from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    # We calculate the new dimensions after tiling
    new_height = height // kh
    new_width = width // kw

    # And then reshape the input to create tiled blocks along the width
    output = input.contiguous().view(batch, channel, height, new_width, kw)
    # We then swap new_width with height so that the height is grouped with the kernel width
    # to give the tiles that we want
    output = output.permute(0, 1, 3, 2, 4)
    # We then pack the khxkw tiles into a single dimension so that we can reduce over them
    output = output.contiguous().view(batch, channel, new_width, new_height, kh * kw)
    # We then swap back to match the original shape
    output = output.permute(0, 1, 3, 2, 4)
    # Finally, we return the tiled tensor as well as the new height and width
    return output, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Performs 2D average pooling across the tensor.

    Args:
    ----
        input (Tensor): input tensor
        kernel (Tuple[int, int]): The dimensions of the sliding window over which values are averaged.

    Returns:
    -------
        :class:`Tensor` : The resulting tensor after performing average pooling

    """
    batch, channel = input.shape[0], input.shape[1]

    # We use the tile function so that the input is reordered with the last dimension
    # containing the tiles that we want to average over
    input, new_height, new_width = tile(input, kernel)
    # We calculate the mean along the last dimension (pooled regions)
    pooled = input.mean(dim=-1)
    # After reducing the last dimension is reduced to 1 so we use view to get rid of it
    return pooled.view(batch, channel, new_height, new_width)


max_reduce = FastOps.reduce(operators.max, float("-inf"))


def argmax(input: Tensor, dim: int) -> Tensor:
    """Computes the argmax of a tensor along a specified dimension.

    Args:
    ----
        input (Tensor): The input tensor.
        dim (int): The dimension along which to compute the argmax.

    Returns:
    -------
        Tensor: A tensor with 1s at the indices of the maximum values and 0 elsewhere.

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Performs the forward pass of Max which applies max along an axis in the given tensor."""
        d = int(dim[0])
        ctx.save_for_backward(input, d)
        return max_reduce(input, d)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """For the backward pass, the only values that influenced the output would be the arguments
        that acheived max so the backwards pass for max is argmax
        """
        (input, dim) = ctx.saved_values
        return argmax(input, dim) * grad_output, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Wraps our Max function so that it can be used by the functions below"""
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Computes the softmax of a tensor along a specified dimension.

    Args:
    ----
        input (Tensor): The input tensor.
        dim (int): The dimension along which to apply the softmax operation.

    Returns:
    -------
        Tensor: The tensor after applying the softmax operation, with values normalized along the specified dimension.

    """
    inp_exp = input.exp()
    return inp_exp / inp_exp.sum(dim=dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax of a tensor along a given axis using the trick
    provided here: https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
    ----
        input : input tensor
        dim (int): The dimension along which to apply the log softmax

    Returns:
    -------
        Tensor: The tensor after applying log softmax

    """
    in_max = max(input, dim)
    logsumexp = ((input - in_max).exp().sum(dim=dim)).log()
    return input - in_max - logsumexp


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Performs 2D max pooling on a tensor using the specified kernel size.

    Args:
    ----
        input (Tensor): The input tensor of shape (batch, channel, height, width).
        kernel (Tuple[int, int]): The height and width of the pooling window.

    Returns:
    -------
        Tensor: The tensor after applying max pooling, with reduced spatial dimensions.

    """
    batch, channel, _, _ = input.shape
    input, new_height, new_width = tile(input, kernel)
    out = max(input, dim=4).view(batch, channel, new_height, new_width)
    return out


def dropout(input: Tensor, drop_rate: float, ignore: bool = False) -> Tensor:
    """Applies dropout to the input tensor by randomly setting values to zero.

    Args:
    ----
        input (Tensor): The input tensor.
        drop_rate (float): The probability of setting a value to zero (between 0 and 1).
        ignore (bool): If True, skips the dropout operation (e.g., during inference).

    Returns:
    -------
        Tensor: The tensor after applying dropout, or the original tensor if `ignore` is True.

    """
    if ignore:
        return input
    else:
        return input * (rand(input.shape) > drop_rate)
