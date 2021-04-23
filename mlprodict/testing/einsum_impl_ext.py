"""
@file
@brief Function to dig into Einsum computation.
"""
import numpy


def _numpy_extended_dot_equation(m1_dim, m2_dim, axes, left, right):
    """
    Returns the equation equivalent to an extended version
    of a matrix multiplication (see @see fn numpy_extended_dot).

    :param m1: number of dimensions of the first matrix
    :param m2: number of dimensions of the second matrix
    :param axes: summation axes
    :param axes: summation axes
    :param left: left axes
    :param right: right axes
    :return: equation
    """
    if m1_dim != m2_dim:
        raise RuntimeError(
            "Matrices m1 and m2 must have the same number of dimensions, "
            "m1=%r, m2=%r." % (m1_dim, m2_dim))
    total = set(axes) | set(left) | set(right)
    if len(total) > m1_dim:
        raise ValueError(
            "Whole set of involved axes should be inferior to the number "
            "of dimensions: %r = {%r} | {%r} | {%r} has more than %d elements"
            "." % (total, axes, left, right, m1_dim))

    def _check_(axs, n):
        for a in axs:
            if a < 0 or a >= n:
                raise ValueError(
                    "One axis %d (in %r) is negative or above the maximum "
                    "dimension %d." % (a, axs, n))
    _check_(axes, m1_dim)
    _check_(left, m1_dim)
    _check_(right, m1_dim)

    l1 = [chr(i + 97) for i in range(m1_dim)]
    l2 = [chr(i + 97) for i in range(m1_dim)]
    l3 = [chr(i + 97) for i in range(m1_dim)]
    for a in left:
        l1[a] = l1[a].upper()
        l3[a] = l3[a].upper()
    for a in right:
        l2[a] = l2[a].upper()
        l3[a] = l3[a].upper()
    for a in axes:
        l1[a] = l1[a].lower()
        l2[a] = l2[a].lower()
        if a not in right:
            l3[a] = None
        else:
            l3[a] = l3[a].lower()
    eq = "%s,%s->%s" % ("".join(l1), "".join(l2),
                        "".join(s for s in l3 if s))
    return eq


def numpy_extended_dot(m1, m2, axes, left, right, verbose=False):
    """
    Extended version of a matrix multiplication (:epkg:`numpy:dot`)
    with two matrices *m1*, *m2* of the same dimensions.
    Loops over *left* axes for *m1* and *right* axes for *m2*,
    summation is done over *axes*.
    Other axes must be empty.
    This multiplication combines matrix multiplication (dot)
    and broadcasted multiplication term by term.

    :param m1: first matrix
    :param m2: second matrix
    :param axes: summation axes
    :param left: left axes
    :param right: right axes
    :param verbose: display intermediate information
    :return: output

    The dot product is equivalent to:

    .. runpython::
        :showcode:

        import numpy
        from mlprodict.testing.einsum_impl_ext import numpy_extended_dot

        m1 = numpy.arange(4).reshape((2, 2))
        m2 = m1 + 10
        print("dot product")
        print(m1 @ m2)

        dm1 = m1.reshape((2, 2, 1))
        dm2 = m2.reshape((1, 2, 2))
        dot = numpy_extended_dot(dm1, dm2, axes=[1], left=[0], right=[2],
                                 verbose=True)
        print("extended dot product")
        print(dot)

    Empty axes should be squeezed to get identical results.
    Dot product when the second matrix is transposed.

    .. runpython::
        :showcode:

        import numpy
        from mlprodict.testing.einsum_impl_ext import numpy_extended_dot

        m1 = numpy.arange(4).reshape((2, 2))
        m2 = m1 + 10
        print("dot product")
        print(m1 @ m2.T)

        dm1 = m1.reshape((2, 1, 2))
        dm2 = m2.reshape((1, 2, 2))
        dot = numpy_extended_dot(dm1, dm2, axes=[2], left=[0], right=[1],
                                 verbose=True)
        print("extended dot product")
        print(dot)

    An example when right axes include the summation axis.

    .. runpython::
        :showcode:

        import numpy
        from mlprodict.testing.einsum_impl_ext import numpy_extended_dot

        m1 = numpy.arange(4).reshape((2, 2))
        m2 = m1 + 10
        dm1 = m1.reshape((2, 2, 1))
        dm2 = m2.reshape((1, 2, 2))
        dot = numpy_extended_dot(dm1, dm2, axes=[2], left=[0], right=[1, 2],
                                 verbose=True)
        print(dot)

    Example in higher dimension:

    .. runpython::
        :showcode:

        import numpy
        from mlprodict.testing.einsum_impl_ext import numpy_extended_dot

        m1 = numpy.arange(8).reshape((2, 2, 2))
        m2 = m1 + 10

        dot = numpy_extended_dot(m1, m2, [1], [0], [2], verbose=True))
        print(dot)

    The current implementation still uses :epkg:`numpy:einsum`
    but this should be replaced.
    """
    if m1.dtype != m2.dtype:
        raise TypeError(
            "Both matrices should share the same dtype %r != %r."
            "" % (m1.dtype, m2.dtype))
    eq = _numpy_extended_dot_equation(
        len(m1.shape), len(m2.shape), axes, left, right)
    if verbose:
        print("  [numpy_extended_dot] %s: %r @ %r" % (eq, m1.shape, m2.shape))
    output = numpy.einsum(eq, m1, m2)
    new_shape = list(output.shape)
    for a in axes:
        if a not in right:
            new_shape.insert(a, 1)
    if verbose:
        print("  [numpy_extended_dot] %r reshaped into %r " % (
            output.shape, new_shape))
    return output.reshape(tuple(new_shape))


def numpy_extended_dot_python(m1, m2, axes, left, right, verbose=False):
    """
    Implementation of @see fn numpy_extended_dot in pure python.
    This implementation is not efficient but shows how to
    implement this operation without :epkg:`numpy:einsum`.
    """
    if m1.dtype != m2.dtype:
        raise TypeError(
            "Both matrices should share the same dtype %r != %r."
            "" % (m1.dtype, m2.dtype))
    m1_dim = len(m1.shape)
    m2_dim = len(m2.shape)
    if m1_dim != m2_dim:
        raise RuntimeError(
            "Matrices m1 and m2 must have the same number of dimensions, "
            "m1=%r, m2=%r." % (m1_dim, m2_dim))
    total = set(axes) | set(left) | set(right)
    if len(total) > m1_dim:
        raise ValueError(
            "Whole set of involved axes should be inferior to the number "
            "of dimensions: %r = {%r} | {%r} | {%r} has more than %d elements"
            "." % (total, axes, left, right, m1_dim))

    new_shape = numpy.full(m1_dim, 1, dtype=numpy.int64)
    for i in left:
        new_shape[i] = m1.shape[i]
    for i in right:
        if i in left and m1.shape[i] != m2.shape[i]:
            raise RuntimeError(
                "Matrices should the same dimension for dimension %d, "
                "shapes=%r @ %r." % (i, m1.shape, m2.shape))
        new_shape[i] = m2.shape[i]

    t_left = 1
    d_left = []
    for n in left:
        t_left *= m1.shape[n]
        d_left.append(n)

    t_right = 1
    d_right = []
    d_common = []
    for n in right:
        if n not in left:
            t_right *= m2.shape[n]
            d_right.append(n)
        else:
            d_common.append(n)

    t_axes = 1
    d_axes = []
    d_common_axes_right = []
    for n in axes:
        if n not in left and n not in right:
            t_axes *= m2.shape[n]
            d_axes.append(n)
        elif n in right and n not in left:
            d_common_axes_right.append(n)
        else:
            raise NotImplementedError()

    if len(d_common_axes_right) == 0:
        res = numpy.full(tuple(new_shape), numpy.nan, dtype=m1.dtype)
    else:
        res = numpy.zeros(tuple(new_shape), dtype=m1.dtype)

    i_left = [0 for i in m1.shape]
    i_right = [0 for i in m1.shape]
    i_out = [0 for i in m1.shape]

    for i in range(t_left):

        for j in range(t_right):  # pylint: disable=W0612

            if len(d_common_axes_right) == 0:
                for d in d_common:
                    i_left[d] = i_right[d]
                add = 0
                for s in range(t_axes):  # pylint: disable=W0612

                    add += m1[tuple(i_left)] * m2[tuple(i_right)]

                    p = len(d_axes) - 1
                    i_left[d_axes[p]] += 1
                    i_right[d_axes[p]] += 1
                    while i_left[d_axes[p]] >= m1.shape[d_axes[p]]:
                        i_left[d_axes[p]] = 0
                        i_right[d_axes[p]] = 0
                        p -= 1
                        if p < 0:
                            break
                        i_left[d_axes[p]] += 1
                        i_right[d_axes[p]] += 1

                res[tuple(i_out)] = add
            elif len(d_axes) == 0:
                for s in range(t_axes):

                    for d in d_common_axes_right:
                        i_out[d] = i_right[d]

                    res[tuple(i_out)] += m1[tuple(i_left)] * m2[tuple(i_right)]

                    p = len(d_common_axes_right) - 1
                    i_right[d_common_axes_right[p]] += 1
                    while (i_left[d_common_axes_right[p]] >=
                            m1.shape[d_common_axes_right[p]]):
                        i_right[d_common_axes_right[p]] = 0
                        p -= 1
                        if p < 0:
                            break
                        i_right[d_common_axes_right[p]] += 1
                    for d in d_common_axes_right:
                        i_out[d] = i_right[d]
                    for d in d_common:
                        i_left[d] = i_right[d]
            else:
                raise NotImplementedError()

            p = len(d_right) - 1
            i_right[d_right[p]] += 1
            i_out[d_right[p]] += 1
            while i_right[d_right[p]] >= m2.shape[d_right[p]]:
                i_right[d_right[p]] = 0
                i_out[d_right[p]] = 0
                p -= 1
                if p < 0:
                    break
                i_right[d_right[p]] += 1
                i_out[d_right[p]] += 1

        p = len(d_left) - 1
        i_left[d_left[p]] += 1
        i_out[d_left[p]] += 1
        while i_left[left[p]] >= m1.shape[d_left[p]]:
            i_left[d_left[p]] = 0
            i_out[d_left[p]] = 0
            p -= 1
            if p < 0:
                break
            i_left[d_left[p]] += 1
            i_out[d_left[p]] += 1

    return res


def numpy_diagonal(m, axis, axes):
    """
    Extracts diagonal coefficients from an array.

    :param m: input array
    :param axis: kept axis among the diagonal ones
    :param axes: diagonal axes (axis must be one of them)
    :return: output

    .. runpython::
        :showcode:

        import numpy
        from mlprodict.testing.einsum_impl_ext import numpy_diagonal

        mat = numpy.arange(8).reshape((2, 2, 2))
        print(mat)
        diag = numpy_diagonal(mat, 1, [1, 2])
        print(diag)
    """
    if axis not in axes:
        raise RuntimeError(
            "axis %r must be in axes %r." % (axis, axes))
    shape = []
    new_shape = []
    for i, s in enumerate(m.shape):
        if i in axes:
            if i == axis:
                shape.append(s)
                new_shape.append(s)
            else:
                shape.append(1)
        else:
            shape.append(s)
            new_shape.append(s)

    # Extracts coefficients.
    output = numpy.empty(tuple(shape), dtype=m.dtype)
    index_in = [slice(s) for s in m.shape]
    index_out = [slice(s) for s in m.shape]
    for i in range(0, shape[axis]):
        for a in axes:
            index_in[a] = i
            index_out[a] = i if a == axis else 0
        output[tuple(index_out)] = m[tuple(index_in)]

    # Removes axis.
    return output.reshape(tuple(new_shape))
