# -*- encoding: utf-8 -*-
# pylint: disable=E0203,E1101,C0111
"""
@file
@brief Helpers for operators Conv, ConvTranspose.
"""
import numpy
from .op_conv_helper_ import (  # pylint: disable=E0611
    im2col_1d_inplace_float,
    tch_im2col_2d_float, tch_col2im_2d_float,
    new_array as _new_array)


def im2col_nn(res):
    """
    Functions @see fn nn_im2col_2d and @see fn im2col returns the
    same results but with different shapes. This function
    converts a result from @see fn nn_im2col_2d into the same
    shape as a return from @see fn nn_im2col_2d.
    """
    if len(res.shape) % 2 != 0:
        raise ValueError(  # pragma: no cover
            "Number of dimensions should be even.")
    m = len(res.shape) // 2
    data = numpy.prod(res.shape[:m])
    ker = numpy.prod(res.shape[m:])
    resh = res.reshape((data, ker))
    tr = numpy.transpose(resh, [1, 0])
    return tr[numpy.newaxis, ...]


def new_array(shape, dtype=numpy.float32):
    """
    Creates a new empty array.

    :param shape: shape
    :param dtype: dtype
    :return: new array
    """
    if dtype == numpy.float32:
        dtype = numpy.dtype('float32')
    return _new_array(list(shape), dtype)


def nn_im2col_2d(data, kernel_shape, dilations, padding, fill_value=0):
    """
    C++ implementation for `im2col` or :func:`torch.nn.Unfold`.

    :param data: image (float), 2 dimensions.
    :param kernel_shape: kernel shape
    :param dilations: dilations
    :param padding: padding
    :param fill_value: fill value
    :return: result
    """
    strides = (1, 1)
    ext_shape = (
        (data.shape[0] + 2 * padding[0] - dilations[0] * (
            kernel_shape[0] - 1) - 1) // strides[0] + 1,
        (data.shape[1] + 2 * padding[1] - dilations[1] * (
            kernel_shape[1] - 1) - 1) // strides[1] + 1)
    kernel_size = kernel_shape[0] * kernel_shape[1]
    shape = (kernel_size, ext_shape[0] * ext_shape[1])
    result = numpy.empty(shape, dtype=data.dtype)
    if data.dtype == numpy.float32:
        tch_im2col_2d_float(result, data,
                            numpy.array(kernel_shape, dtype=numpy.int64),
                            numpy.array(dilations, dtype=numpy.int64),
                            numpy.array(padding, dtype=numpy.int64),
                            fill_value)
    else:
        raise NotImplementedError(  # pragma: no cover
            f"Unexpected dtype {data.dtype!r} for data.")
    return result


def nn_col2im_2d(data, output_shape, kernel_shape, dilations, padding):
    """
    C++ implementation for `col2im` or :func:`torch.nn.Fold`.

    :param data: image (float), 2 dimensions.
    :param output_shape: output size
    :param kernel_shape: kernel shape
    :param dilations: dilations
    :param padding: padding
    :return: result
    """
    result = numpy.empty(output_shape, dtype=data.dtype)
    if data.dtype == numpy.float32:
        tch_col2im_2d_float(result, data,
                            numpy.array(output_shape, dtype=numpy.int64),
                            numpy.array(kernel_shape, dtype=numpy.int64),
                            numpy.array(dilations, dtype=numpy.int64),
                            numpy.array(padding, dtype=numpy.int64))
    else:
        raise NotImplementedError(  # pragma: no cover
            f"Unexpected dtype {data.dtype!r} for data.")
    return result


def _get_indices(i, shape):
    res = numpy.empty((len(shape), ), dtype=numpy.int64)
    k = len(shape) - 1
    while k > 0:
        m = i % shape[k]
        res[k] = m
        i -= m
        i /= shape[k]
        k -= 1
    res[0] = i
    return res


def _is_out(ind, shape):
    for i, s in zip(ind, shape):
        if i < 0:
            return True
        if i >= s:
            return True
    return False


def im2col_naive_implementation(data, kernel_shape, fill_value=0):
    """
    Naive implementation for `im2col` or
    :func:`torch.nn.Unfold` (but with `padding=1`).

    :param image: image (float)
    :param kernel_shape: kernel shape
    :param fill_value: fill value
    :return: result
    """
    if not isinstance(kernel_shape, tuple):
        raise TypeError(
            f"Unexpected type {type(kernel_shape)!r} for kernel_shape.")
    if len(data.shape) != len(kernel_shape):
        raise ValueError(
            f"Shape mismatch {data.shape!r} and {kernel_shape!r}.")
    output_shape = data.shape + kernel_shape
    res = numpy.empty(output_shape, dtype=data.dtype)
    middle = numpy.array([-m / 2 for m in kernel_shape], dtype=numpy.int64)
    kernel_size = numpy.prod(kernel_shape)
    data_size = numpy.prod(data.shape)
    for i in range(data_size):
        for j in range(kernel_size):
            i_data = _get_indices(i, data.shape)
            i_kernel = _get_indices(j, kernel_shape)
            ind = i_data + i_kernel + middle
            t_data = tuple(i_data)
            t_kernel = tuple(i_kernel)
            i_out = t_data + t_kernel
            res[i_out] = fill_value if _is_out(
                ind, data.shape) else data[tuple(ind)]
    return res


def im2col_recursive(data, kernel_shape, fill_value=0, fall_back_dim=2):
    """
    Recursive implementation, falls back to
    @see fn im2col_naive_implementation for dimension `<= fall_back_dim`.
    The function is equivalent to
    :func:`torch.nn.Unfold` (but with `padding=1` on all dimensions).

    :param image: image (float)
    :param kernel_shape: kernel shape
    :param fill_value: fill value
    :param fall_back_dim: below that threshold,
        switches to @see fn im2col_naive_implementation.
    :return: result
    """
    if len(data.shape) <= fall_back_dim:
        return im2col_naive_implementation(data, kernel_shape, fill_value)

    perm = numpy.arange(len(data.shape) * 2).tolist()
    del perm[1:2]
    perm.insert(len(data.shape), 1)

    res = []
    N0 = data.shape[0]
    k0 = kernel_shape[0]
    mini_kernel = kernel_shape[1:]
    mini_shape = data.shape[1:] + mini_kernel
    for i in range(N0):
        for k in range(k0):
            ii = k - k0 // 2 + i
            if ii < 0 or ii >= N0:
                cc = numpy.full(mini_shape, dtype=data.dtype,
                                fill_value=fill_value)
            else:
                # many computation are already done, results should be cached.
                cc = im2col_recursive(data[ii], mini_kernel, fill_value)
            cc2 = cc[numpy.newaxis, ...]
            res.append(cc2)

    final = numpy.vstack(res)
    new_shape = (N0, k0) + cc.shape
    resh = final.reshape(new_shape)
    return numpy.transpose(resh, tuple(perm))


def im2col(data, kernel_shape=None, fill_value=0):
    """
    Returns the result of `im2col` on a image `NHCW` where N is 1.
    The function is equivalent to
    :func:`torch.nn.Unfold` (but with `padding=1` on all dimensions).

    :param image: image (float)
    :param kernel_shape: kernel shape
    :param fill_value: fill value
    :return: result

    This function is equivalent to function
    :func:`torch.nn.Unfold` with `padding=kernel_shape / 2`
    followed by a reshape and a transpose.

    ::

        import numpy
        from numpy.testing import assert_almost_equal
        import torch

        data = (numpy.arange(20).astype(numpy.float64) + 10).reshape((4, 5))
        expected = im2col_recursive(data, (3, 3), fill_value=0)
        unfold = torch.nn.Unfold(kernel_size=(3, 3), padding=1)
        input = torch.from_numpy(data.reshape((1, 1) + data.shape))
        output = unfold(input)
        mat = output.numpy()
        tr = numpy.transpose(mat, [0, 2, 1])
        resh = tr.reshape(expected.shape)
        assert_almost_equal(expected, resh)
    """
    if len(data.shape) == 1:
        if kernel_shape is None:
            kernel_shape = (3, )
        elif len(kernel_shape) != 1:
            raise ValueError(
                f"Unexpected kernel_shape {kernel_shape!r}, should be 1d.")
        if data.dtype == numpy.float32:
            result = numpy.empty(
                (data.shape[0], kernel_shape[0]), dtype=data.dtype)
            im2col_1d_inplace_float(
                result, data,
                kernel_shape if isinstance(kernel_shape, numpy.ndarray)
                else numpy.array(kernel_shape, dtype=numpy.int64),
                numpy.float32(fill_value))
            return result
    return im2col_naive_implementation(data, kernel_shape, fill_value)


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    """
    Source `im2col.py <https://github.com/huyouare/CS231n/
    blob/master/assignment2/cs231n/im2col.py>`_.
    """
    # First figure out what the size of the output should be
    _, C, H, W = x_shape
    if (H + 2 * padding - field_height) % stride != 0:
        raise RuntimeError(
            "Unexpected value: %d != %d." % (
                H + 2 * padding - field_height, stride))
    if (W + 2 * padding - field_height) % stride != 0:
        raise RuntimeError(
            "Unexpected value: %d != %d." % (
                W + 2 * padding - field_height, stride))
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1

    i0 = numpy.repeat(numpy.arange(field_height), field_width)
    i0 = numpy.tile(i0, C)
    i1 = stride * numpy.repeat(numpy.arange(out_height), out_width)
    j0 = numpy.tile(numpy.arange(field_width), field_height * C)
    j1 = stride * numpy.tile(numpy.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = numpy.repeat(numpy.arange(C), field_height *
                     field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=0, stride=1):
    """
    Source `im2col.py <https://github.com/huyouare/CS231n/
    blob/master/assignment2/cs231n/im2col.py>`_.
    """
    if padding > 0:
        p = padding
        x_padded = numpy.pad(
            x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    else:
        x_padded = x
    k, i, j = get_im2col_indices(
        x.shape, field_height, field_width, padding, stride)
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=0,
                   stride=1):
    """
    Source `im2col.py <https://github.com/huyouare/CS231n/
    blob/master/assignment2/cs231n/im2col.py>`_.
    """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = numpy.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                                 stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    numpy.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]
