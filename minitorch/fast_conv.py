from typing import Tuple, TypeVar, Any

from numba import prange
from numba import njit as _njit

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    Shape,
    Strides,
    Storage,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

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


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


def _tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right

    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    s1 = input_strides
    s2 = weight_strides
    # for each batch
    # for each output channel
    # for each element in width
    # (batch, output_channel, width) -> out_pos
    # (batch, in_channels, width) -> in_pos
    # sum up (batch, output_channel, width + j) * (output_channel, in_channels, width)
    # For each batch
    for b in prange(batch):
        # For each output channel
        for o in prange(out_channels):
            # For each position in the output width
            for w in prange(out_width):
                # Get the corresponding position in output storage
                out_ordinal = (
                    b * out_strides[0] + o * out_strides[1] + w * out_strides[2]
                )
                # For each channel in the input
                for c_ in prange(in_channels):
                    # Determine the range of input positions to consider
                    # `start` and `end` depend on whether weights are reversed
                    start = max(w - kw + 1, 0) if reverse else min(w, width - 1)
                    end = min(w + 1, width) if reverse else min(w + kw, width)
                    # Loop over the input positions in the range
                    for i_ in prange(start, end):
                        # Compute the index in the flattened input storage
                        in_ordinal = b * s1[0] + c_ * s1[1] + i_ * s1[2]
                        # Compute the index in the flattened weight storage
                        weight_ordinal = o * s2[0] + c_ * s2[1] + (i_ - start) * s2[2]
                        # Accumulate the convolution result into the output
                        out[out_ordinal] += input[in_ordinal] * weight[weight_ordinal]


tensor_conv1d = njit(_tensor_conv1d, parallel=True)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 1D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
        -------
            batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Performs the backwards pass for 1D convolution"""
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right

    """
    batch_, out_channels, _, _ = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    # For each batch
    for b in prange(batch):
        # For each output channel
        for c in prange(out_channels):
            # For each position in output height
            for h in prange(out_shape[2]):
                # For each position in output width
                for w in prange(out_shape[3]):
                    # Calculate the position in output storage
                    out_ordinal = (
                        b * out_strides[0]
                        + c * out_strides[1]
                        + h * out_strides[2]
                        + w * out_strides[3]
                    )
                    temp = 0.0
                    # Loop over input channels
                    for ic in range(in_channels):
                        # Loop over the kernel height
                        for kh_ in range(kh):
                            # Loop over the kernel width
                            for kw_ in range(kw):
                                # Calculate the corresponding input height and width indices
                                ih = h + (-kh_ if reverse else kh_)
                                iw = w + (-kw_ if reverse else kw_)

                                # Check if the input indices are within bounds (essentially pads input with zeros)
                                if 0 <= ih < height and 0 <= iw < width:
                                    # Compute the index in the flattened input storage
                                    in_ordinal = (
                                        b * input_strides[0]
                                        + ic * input_strides[1]
                                        + ih * input_strides[2]
                                        + iw * input_strides[3]
                                    )
                                    # Compute the index in the flattened weight storage
                                    weight_ordinal = (
                                        c * weight_strides[0]
                                        + ic * weight_strides[1]
                                        + kh_ * weight_strides[2]
                                        + kw_ * weight_strides[3]
                                    )
                                    # Add their product to temp
                                    temp += input[in_ordinal] * weight[weight_ordinal]
                    # Write out to output
                    out[out_ordinal] = temp


tensor_conv2d = njit(_tensor_conv2d, parallel=True, fastmath=True)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 2D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
        -------
            (:class:`Tensor`) : batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Performs the backwards pass for 2D convolution"""
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
