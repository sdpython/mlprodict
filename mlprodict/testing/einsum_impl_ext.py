"""
@file
@brief Function to dig into Einsum computation.
"""
import numpy


def numpy_extended_dot(m1, m2, axes, left, right, verbose=False):
    """
    Extended version of a matrix multiplication (:epkg:`numpy:dot`)
    with two matrices *m1*, *m2* of the same dimensions.
    Loops over *left* axes for *m1* and *right* axes for *m2*,
    summation is done over *axes*.
    Other axes must be empty.

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
    if len(m1.shape) != len(m2.shape):
        raise RuntimeError(
            "Matrices m1 and m2 must have the same dimension, "
            "m1=%r, m2=%r." % (m1.shape, m2.shape))

    def _check_(axs, n):
        for a in axs:
            if a < 0 or a >= n:
                raise ValueError(
                    "One axis %d (in %r) is negative or above the maximum "
                    "dimension %d." % (a, axs, n))
    _check_(axes, len(m1.shape))
    _check_(left, len(m1.shape))
    _check_(right, len(m1.shape))

    # This implementation should not use einsum.
    # Temporary solution.
    l1 = [chr(i + 97) for i in range(len(m1.shape))]
    l2 = [chr(i + 97) for i in range(len(m2.shape))]
    l3 = [chr(i + 97) for i in range(len(m2.shape))]
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
