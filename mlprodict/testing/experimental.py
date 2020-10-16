"""
@file
@brief Experimental implementation.
"""
import numpy


def custom_pad(arr, paddings, constant=0, debug=False):
    """
    Implements function
    `pad <https://numpy.org/doc/stable/reference/
    generated/numpy.pad.html>`_ in python,
    only the constant version.

    :param arr: array
    :param paddings: paddings
    :param constant: constant
    :return: padded array
    """
    if paddings.shape[0] != len(arr.shape):
        raise ValueError(
            "Input shape {} and paddings {} are inconsistent.".format(
                arr.shape, paddings))
    if min(paddings.ravel()) < 0:
        raise NotImplementedError("Negative paddings is not implemented yet.")
    if not arr.flags['C_CONTIGUOUS']:
        arr = numpy.ascontiguousarray(arr)

    new_shape = tuple(
        a + s for a, s in zip(arr.shape, numpy.sum(paddings, axis=1, keepdims=0)))

    cumulative_copy = [1]
    for a in reversed(new_shape):
        cumulative_copy.insert(0, a * cumulative_copy[0])
    cumulative_input = [1]
    for a in reversed(arr.shape):
        cumulative_input.insert(0, a * cumulative_input[0])

    input_arr = arr.ravel()
    if debug:
        res = numpy.zeros(cumulative_copy[0], dtype=arr.dtype) - 1
    else:
        res = numpy.empty(cumulative_copy[0], dtype=arr.dtype)

    # preparation
    first_index = sum(
        p * c for p, c in zip(paddings[:, 0], cumulative_copy[1:]))
    dh_input = arr.shape[-1]
    dh_copy = new_shape[-1]

    # constance
    no_constant = 1 if constant == 0 else 0
    res[first_index:cumulative_copy[0]:dh_copy] = no_constant

    # padding
    for i, sh in enumerate(new_shape):
        upper_number = cumulative_copy[0] // cumulative_copy[i]
        contiguous = cumulative_copy[i + 1]
        big_index = 0
        p_left = paddings[i, 0] * contiguous
        p_right = paddings[i, 1] * contiguous
        dp = sh * contiguous - p_right
        for _ in range(upper_number):
            if p_left > 0:
                res[big_index:big_index + p_left] = constant
            if p_right > 0:
                index = big_index + dp
                res[index:index + p_right] = constant
            big_index += cumulative_copy[i]

    # copy
    index_input = 0
    index_copy = first_index
    while index_copy < cumulative_copy[0]:
        if res[index_copy] == no_constant:
            res[index_copy:index_copy + dh_input] = \
                input_arr[index_input:index_input + dh_input]
            index_input += dh_input
        index_copy += dh_copy

    # final
    return res.reshape(new_shape)
