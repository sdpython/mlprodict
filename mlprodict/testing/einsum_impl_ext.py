"""
@file
@brief Function to dig into Einsum computation.
"""
import numpy


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

    for a in axes:
        if a in left and a in right:
            raise RuntimeError(
                "One axis belongs to every set (axes, left, right). "
                "axes=%r, left=%r, right=%r." % (axes, left, right))

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
    def dispb(c):
        return "".join("o" if b else "." for b in c)

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
        if (i in left and m1.shape[i] != m2.shape[i] and
                m1.shape[i] != 1 and m2.shape[i] != 1):
            raise RuntimeError(
                "Matrices should the same dimension for dimension %d, "
                "shapes=%r @ %r." % (i, m1.shape, m2.shape))
        new_shape[i] = m2.shape[i]

    # output shapes
    res = numpy.full(tuple(new_shape), 0, dtype=m1.dtype)

    # indices
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
            l3[a] = "-"
        else:
            l3[a] = l3[a].lower()

    def intermediate(l1, l2, l3):
        names = list(sorted(set(l1 + l2)))
        kind = numpy.zeros(len(names), dtype=numpy.int64)
        cols = {}

        for i, n in enumerate(names):
            if n in l1:
                kind[i] += 1
                cols[n] = l1.index(n)
            if n in l2:
                kind[i] += 2
                cols[n] = l2.index(n)
            if n in l3:
                kind[i] += 4

        pos = numpy.zeros(len(names), dtype=numpy.int64)
        for j in range(0, pos.shape[0]):
            pos[j] = cols[names[j]]
        common = [(kind[i] & 3) == 3 for i in range(len(kind))]
        broadcast = [common[i] and m1.shape[pos[i]] != m2.shape[pos[i]]
                     for i in range(len(common))]

        return names, kind, cols, common, broadcast, pos

    names, kind, cols, common, broadcast, pos = intermediate(l1, l2, l3)

    if any(broadcast):
        if verbose:
            print("GENERICDOT: before broadcast %s,%s->%s      or %s" % (
                "".join(l1), "".join(l2), "".join(l3),
                _numpy_extended_dot_equation(
                    len(m1.shape), len(m1.shape), axes, left, right)))
            print("GENERICDOT: names=%s kind=%r common=%s broadcast=%s" % (
                "".join(names), kind.tolist(),
                dispb(common), dispb(broadcast)))

        for i in range(len(broadcast)):  # pylint: disable=C0200
            if broadcast[i] and not (kind[i] & 3) == 3:
                raise RuntimeError(
                    "Broadcast should only happen on common axes, "
                    "axes=%r left=%r right=%r shape1=%r shape2=%r."
                    "" % (axes, left, right, m1.shape, m2.shape))
            if not broadcast[i]:
                continue
            # We split letters.
            p = cols[names[i]]
            dim = (m1.shape[p], m2.shape[p])
            let = [l1[p], l2[p], l3[p]]
            inp = 1 if dim[0] == 1 else 0
            if verbose:
                print("GENERICDOT: name=%s dim=%r let=%r inp=%r p=%r" % (
                    names[i], dim, let, inp, p))
                print("    B0 l1=%r, l2=%r l3=%r" % (l1, l2, l3))
            if (kind[i] & 4) > 0:
                # Summation axis is part of the output.
                if let[inp].lower() == let[inp]:
                    let[inp] = let[inp].upper()
                else:
                    let[inp] = let[inp].lower()
                l3[p] = let[inp]
                if inp == 1:
                    l2[p] = let[inp]
                else:
                    l1[p] = let[inp]
                if verbose:
                    print("    B1 l1=%r, l2=%r l3=%r" % (l1, l2, l3))
            else:
                # Summation axis is not part of the output.
                if let[inp].lower() == let[inp]:
                    let[inp] = let[inp].upper()
                else:
                    let[inp] = let[inp].lower()
                if inp == 1:
                    l2[p] = let[inp]
                else:
                    l1[p] = let[inp]
                if verbose:
                    print("    B2 l1=%r, l2=%r l3=%r" % (l1, l2, l3))

        names, kind, cols, common, broadcast, pos = intermediate(l1, l2, l3)

    indices = numpy.array([0 for n in names], dtype=numpy.int64)
    pl1 = numpy.array([names.index(c) for c in l1], dtype=numpy.int64)
    pl2 = numpy.array([names.index(c) for c in l2], dtype=numpy.int64)
    limits = numpy.array(
        [m1.shape[pos[n]] if (kind[n] & 1) == 1 else m2.shape[pos[n]]
         for n in range(len(names))], dtype=numpy.int64)
    plo = numpy.array(
        [-1 if c not in names else names.index(c) for c in l3], dtype=numpy.int64)

    if verbose:
        print("GENERICDOT: %s,%s->%s      or %s" % (
            "".join(l1), "".join(l2), "".join(l3),
            _numpy_extended_dot_equation(
                len(m1.shape), len(m1.shape), axes, left, right)))
        print("GENERICDOT: shape1=%r shape2=%r shape=%r" % (
            m1.shape, m2.shape, res.shape))
        print("GENERICDOT: axes=%r left=%r right=%r" % (axes, left, right))
        print("GENERICDOT: pl1=%r pl2=%r plo=%r" % (pl1, pl2, plo))
        print("GENERICDOT: names=%s kind=%r common=%s broadcast=%s" % (
            "".join(names), kind.tolist(),
            dispb(common), dispb(broadcast)))
        print("GENERICDOT: pos=%r" % pos.tolist())
        print("GENERICDOT: cols=%r" % cols)
        print("GENERICDOT: limits=%r" % limits)

    while indices[0] < limits[0]:

        # The function spends most of its time is these three lines.
        t1 = tuple(indices[n] for n in pl1)
        t2 = tuple(indices[n] for n in pl2)
        to = tuple(0 if n == -1 else indices[n] for n in plo)

        c = m1[t1] * m2[t2]

        if verbose:
            print(" %r x %r -> %r v=%r I=%r" % (t1, t2, to, c, indices))

        res[to] += c

        last = len(indices) - 1
        indices[last] += 1
        for i in range(last, 0, -1):
            if indices[i] < limits[i]:
                break
            indices[i] = 0
            if i > 0:
                indices[i - 1] += 1

    return res
