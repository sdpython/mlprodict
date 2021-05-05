"""
@file
@brief Experimental implementation.
"""
from collections import OrderedDict
import numpy


def custom_pad(arr, paddings, constant=0, verbose=False):
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
        raise ValueError(  # pragma: no cover
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
    if verbose:
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


def custom_einsum(equation, x, y, verbose=False):
    """
    Experimental implementation of operator Einsum
    when it does a matrix multiplication.
    Case: ``bsnh,btnh->bnts`` with shapes
    `(1,512,12,64)` and `(1,512,12,64)`.

    :param equation: equation
    :param x: first matrix
    :param y: second matrix
    :param verbose: display internal information
    :return: result of *einsum*

    This implementation does not any transpose,
    it does a direct computation of the final result.
    It does not implementation diagonal summation (square product).
    """
    def _check_eq(eq, sh):
        if len(eq) != len(sh):
            raise ValueError(
                "Unable to map equation %r to shape %r." % (eq, sh))

    def _split(eq, sh):
        dx = OrderedDict((e, (v, i)) for i, (e, v) in enumerate(zip(eq, sh)))
        return dx

    def _interpret(dx, dy, eqr):
        c_uni = []
        c_trp = []
        c_sum = []
        for r in eqr:
            if r in dx:
                if r in dy:
                    if dx[r][0] != dy[r][0]:
                        raise ValueError(
                            "Dimension mismatch for letter "
                            "%r dx=%r dy=%r." % (r, dx, dy))
                    c_trp.append(r)
                else:
                    c_uni.append((r, None))
            elif r in dy:
                c_uni.append((None, r))
            else:
                raise ValueError(  # pragma: no cover
                    "Unexpected letter %r in result %r." % (r, eqr))
        for c in dx:
            if c not in eqr:
                if c not in dy:
                    raise ValueError(  # pragma: no cover
                        "Unable to guess what to do with column %r (left side)" % c)
                if dx[c][0] != dy[c][0]:
                    raise ValueError(  # pragma: no cover
                        "Dimension mismatch for letter "
                        "%r dx=%r dy=%r." % (c, dx, dy))
                c_sum.append(c)
        for c in dy:
            if c not in eqr and c not in dx:
                raise ValueError(  # pragma: no cover
                    "Unable to guess what to do with column %r (right side)" % c)
        shape = OrderedDict()
        for i, r in enumerate(eqr):
            if r in c_trp:
                shape[r] = (dx[r][0], i)
            else:
                for a, b in c_uni:
                    if a == r:
                        shape[r] = (dx[r][0], i)
                        break
                    if b == r:
                        shape[r] = (dy[r][0], i)
                        break
        if len(shape) != len(eqr):
            raise RuntimeError(  # pragma: no cover
                "Unable to compute the output shape "
                "dx=%r dy=%r eqr=%r got shape=%r." % (dx, dy, eqr, shape))
        return shape, c_trp, c_uni, c_sum

    def _inc(d):
        t = 1
        drev = list(reversed(d.items()))
        res = []
        for c, (sh, p) in drev:
            res.append((c, (t, p)))
            t *= sh
        return OrderedDict(reversed(res))

    def prod(seq):
        p = 1
        for s in seq:
            p *= s
        return p

    def get_index(cd, shape, index, col_sum):
        ind = 0
        for c, i in zip(shape, index):
            if c in cd:
                inc = cd[c][0]
                ind += inc * i
        return ind, cd[col_sum][0]

    def get_incs(cd, shape):
        incs = []
        for c in shape:
            inc = cd[c][0] if c in cd else 0
            incs.append(inc)
        return incs

    if x.dtype != y.dtype:
        raise RuntimeError("x and y must have the same dtype.")
    eqx = equation.split(',')[0]
    eqy = equation.split(',')[-1].split('->')[0]
    eqr = equation.split('->')[-1]
    _check_eq(eqx, x.shape)
    _check_eq(eqy, y.shape)
    dx = _split(eqx, x.shape)
    dy = _split(eqy, y.shape)
    shape, __, _, c_sum = _interpret(dx, dy, eqr)
    cdx = _inc(dx)
    cdy = _inc(dy)
    xrav = x.ravel()
    yrav = y.ravel()
    full_size = prod(v[0] for v in shape.values())
    zrav = numpy.empty((full_size, ), dtype=x.dtype)

    # loop
    if len(c_sum) != 1:
        raise NotImplementedError(
            "More than one summation indices %r in equation %r." % (
                c_sum, equation))
    zeros = numpy.zeros((1, ), dtype=x.dtype)
    shape_dims = [v[0] for v in shape.values()]
    index = [0 for s in shape]
    len_index = len(index)
    loop_size = dx[c_sum[0]][0]

    i_left_loop, inc_left = get_index(cdx, shape, index, c_sum[0])
    i_right_loop, inc_right = get_index(cdy, shape, index, c_sum[0])
    left_incs = get_incs(cdx, shape)
    right_incs = get_incs(cdy, shape)

    if verbose:
        def MakeString(*args):
            return "".join(map(str, args))

        print(MakeString("equation=", equation))
        print(MakeString("c_sum=", c_sum))
        print(MakeString("full_size=", full_size))
        print(MakeString("loop_size=", loop_size))
        print(MakeString("i_left_loop=", i_left_loop))
        print(MakeString("i_right_loop=", i_right_loop))
        print(MakeString("inc_left=", inc_left))
        print(MakeString("inc_right=", inc_right))
        print(MakeString("left_incs=", left_incs))
        print(MakeString("right_incs=", right_incs))
        print(MakeString("shape=", shape))
        print(MakeString("cdx=", cdx))
        print(MakeString("cdy=", cdy))

    for i in range(0, full_size):

        i_left = i_left_loop
        i_right = i_right_loop

        # summation
        add = zeros[0]
        for _ in range(loop_size):
            add += xrav[i_left] * yrav[i_right]
            i_left += inc_left
            i_right += inc_right
        zrav[i] = add

        if verbose:
            print(MakeString(
                "  -- index=", index, " ii=", i,
                " i_left_loop=", i_left_loop, " i_right_loop=", i_right_loop,
                " add=", add))

        # increment
        pos = len_index - 1
        index[pos] += 1
        i_left_loop += left_incs[pos]
        i_right_loop += right_incs[pos]
        while pos > 0 and index[pos] >= shape_dims[pos]:
            i_left_loop -= left_incs[pos] * index[pos]
            i_right_loop -= right_incs[pos] * index[pos]
            index[pos] = 0
            pos -= 1
            index[pos] += 1
            i_left_loop += left_incs[pos]
            i_right_loop += right_incs[pos]

    new_shape = tuple(v[0] for v in shape.values())
    return zrav.reshape(new_shape)
