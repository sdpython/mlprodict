"""
@file
@brief Functions implemented einsum computation for two
matrices having the same dimensions.
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
        from mlprodict.testing.einsum import numpy_diagonal

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
    of an aligned matrix multiplication
    (see @see fn numpy_extended_dot).

    :param m1: number of dimensions of the first matrix
    :param m2: number of dimensions of the second matrix
    :param axes: summation axes
    :param axes: summation axes
    :param left: left axes
    :param right: right axes
    :return: equation

    .. runpython::
        :showcode:

        import numpy
        from mlprodict.testing.einsum.einsum_impl_ext import (
            numpy_extended_dot_python, _numpy_extended_dot_equation)

        a = numpy.arange(6).reshape((3, 2, 1))
        b = numpy.arange(12).reshape((3, 1, 4))

        print(numpy_extended_dot_python(
            a, b, axes=(0, ), left=(1,), right=(2,)))

        # Equivalent einsum equation
        print('equation', _numpy_extended_dot_equation(
            len(a.shape), len(a.shape), axes=(0, ), left=(1,), right=(2,)))

        # Same einsum computation written in a different way.
        print(numpy.einsum('kix,kxj->xij', a, b))
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


def _common_check_numpy_extended_dot(m1, m2, axes, left, right):
    """
    Common verifications for all implementations of
    @see fn numpy_extended_dot.
    """
    if m1.dtype != m2.dtype:
        raise TypeError(
            "Both matrices should share the same dtype %r != %r."
            "" % (m1.dtype, m2.dtype))
    m1_dim = len(m1.shape)
    m2_dim = len(m2.shape)
    if m1_dim != m2_dim:
        raise RuntimeError(  # pragma: no cover
            "Matrices m1 and m2 must have the same number of dimensions, "
            "m1=%r, m2=%r." % (m1_dim, m2_dim))
    total = set(axes) | set(left) | set(right)
    if len(total) > m1_dim:
        raise ValueError(
            "Whole set of involved axes should be inferior to the number "
            "of dimensions: %r = {%r} | {%r} | {%r} has more than %d elements"
            "." % (total, axes, left, right, m1_dim))


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
        from mlprodict.testing.einsum import numpy_extended_dot

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
        from mlprodict.testing.einsum import numpy_extended_dot

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
        from mlprodict.testing.einsum import numpy_extended_dot

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
        from mlprodict.testing.einsum import numpy_extended_dot

        m1 = numpy.arange(8).reshape((2, 2, 2))
        m2 = m1 + 10

        dot = numpy_extended_dot(m1, m2, [1], [0], [2], verbose=True)
        print(dot)

    The current implementation still uses :epkg:`numpy:einsum`
    but this should be replaced.
    """
    _common_check_numpy_extended_dot(m1, m2, axes, left, right)
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


def numpy_extended_dot_ouput_shape(m1, m2, axes, left, right):
    """
    Computes the output shape of results produced by function
    :func:`numpy_extended_dot
    <mlprodict.testing.einsum_impl_ext.numpy_extended_dot>` or
    :func:`numpy_extended_dot_python
    <mlprodict.testing.einsum_impl_ext.numpy_extended_dot_python>`.
    """
    _common_check_numpy_extended_dot(m1, m2, axes, left, right)
    m1_dim = len(m1.shape)

    new_shape = numpy.full(m1_dim, 1, dtype=numpy.int64)
    for i in left:
        new_shape[i] = m1.shape[i]
    for i in right:
        if (i in left and m1.shape[i] != m2.shape[i] and
                m1.shape[i] != 1 and m2.shape[i] != 1):
            raise RuntimeError(  # pragma: no cover
                "Matrices should have the same dimension for dimension %d, "
                "shapes=%r @ %r." % (i, m1.shape, m2.shape))
        new_shape[i] = m2.shape[i]
    return new_shape


def _numpy_extended_dot_python_l1l2l3(m1_dim, axes, left, right):
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
    return l1, l2, l3


def _numpy_extended_dot_python_intermediate(m1_shape, m2_shape, l1, l2, l3):
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
    broadcast = [common[i] and m1_shape[pos[i]] != m2_shape[pos[i]]
                 for i in range(len(common))]

    return names, kind, cols, common, broadcast, pos


def _numpy_extended_dot_python_update_broadcast(
        m1, m2, axes, left, right, l1, l2, l3, names, broadcast, cols,
        kind, common, verbose=False):

    def dispb(c):
        return "".join("o" if b else "." for b in c)

    if verbose:
        print("[GENERICDOT] before broadcast %s,%s->%s      or %s" % (
            "".join(l1), "".join(l2), "".join(l3),
            _numpy_extended_dot_equation(
                len(m1.shape), len(m1.shape), axes, left, right)))
        print("[GENERICDOT] names=%s kind=%r common=%s broadcast=%s" % (
            "".join(names), kind.tolist(),
            dispb(common), dispb(broadcast)))

    for i in range(len(broadcast)):  # pylint: disable=C0200
        if broadcast[i] and not (kind[i] & 3) == 3:
            raise RuntimeError(  # pragma: no cover
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
            print("[GENERICDOT] name=%s dim=%r let=%r inp=%r p=%r" % (
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

    return l1, l2, l3


def numpy_extended_dot_python(m1, m2, axes, left, right, verbose=False):
    """
    Implementation of @see fn numpy_extended_dot in pure python.
    This implementation is not efficient but shows how to
    implement this operation without :epkg:`numpy:einsum`.

    .. runpython::
        :showcode:

        import numpy
        from mlprodict.testing.einsum import numpy_extended_dot_python
        from mlprodict.testing.einsum.einsum_impl_ext import (
            _numpy_extended_dot_equation)

        a = numpy.arange(6).reshape((3, 2, 1))
        b = numpy.arange(12).reshape((3, 1, 4))

        print(numpy_extended_dot_python(
            a, b, axes=(0, ), left=(1,), right=(2,)))

        # Equivalent einsum equation
        print('equation', _numpy_extended_dot_equation(
            len(a.shape), len(a.shape), axes=(0, ), left=(1,), right=(2,)))

        # Same einsum computation written in a different way.
        print(numpy.einsum('kix,kxj->xij', a, b))
    """
    def dispb(c):
        return "".join("o" if b else "." for b in c)

    new_shape = numpy_extended_dot_ouput_shape(m1, m2, axes, left, right)
    m1_dim = len(m1.shape)

    # output result
    res = numpy.full(tuple(new_shape), 0, dtype=m1.dtype)

    # indices
    l1, l2, l3 = _numpy_extended_dot_python_l1l2l3(m1_dim, axes, left, right)
    names, kind, cols, common, broadcast, pos = (
        _numpy_extended_dot_python_intermediate(
            m1.shape, m2.shape, l1, l2, l3))

    if any(broadcast):
        l1, l2, l3 = _numpy_extended_dot_python_update_broadcast(
            m1, m2, axes, left, right, l1, l2, l3, names, broadcast, cols,
            kind, common, verbose=verbose)

        names, kind, cols, common, broadcast, pos = (
            _numpy_extended_dot_python_intermediate(
                m1.shape, m2.shape, l1, l2, l3))

    indices = numpy.array([0 for n in names], dtype=numpy.int64)
    pl1 = numpy.array([names.index(c) for c in l1], dtype=numpy.int64)
    pl2 = numpy.array([names.index(c) for c in l2], dtype=numpy.int64)
    limits = numpy.array(
        [m1.shape[pos[n]] if (kind[n] & 1) == 1 else m2.shape[pos[n]]
         for n in range(len(names))], dtype=numpy.int64)
    plo = numpy.array(
        [-1 if c not in names else names.index(c) for c in l3],
        dtype=numpy.int64)

    if verbose:
        print("[GENERICDOT] %s,%s->%s      or %s" % (
            "".join(l1), "".join(l2), "".join(l3),
            _numpy_extended_dot_equation(
                len(m1.shape), len(m1.shape), axes, left, right)))
        print("[GENERICDOT] shape1=%r shape2=%r shape=%r" % (
            m1.shape, m2.shape, res.shape))
        print("[GENERICDOT] axes=%r left=%r right=%r" % (axes, left, right))
        print("[GENERICDOT] pl1=%r pl2=%r plo=%r" % (pl1, pl2, plo))
        print("[GENERICDOT] names=%s kind=%r common=%s broadcast=%s" % (
            "".join(names), kind.tolist(),
            dispb(common), dispb(broadcast)))
        print("[GENERICDOT] pos=%r" % pos.tolist())
        print("[GENERICDOT] cols=%r" % cols)
        print("[GENERICDOT] limits=%r" % limits)

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


def numpy_extended_dot_matrix(m1, m2, axes, left, right, verbose=False):
    """
    Implementation of @see fn numpy_extended_dot using dot product,
    multiplication, transpose and reduction
    but not a custom python implementation like
    @see fn numpy_extended_dot_python.

    .. runpython::
        :showcode:

        import numpy
        from mlprodict.testing.einsum import numpy_extended_dot_matrix
        from mlprodict.testing.einsum.einsum_impl_ext import (
            _numpy_extended_dot_equation)

        a = numpy.arange(6).reshape((3, 2, 1))
        b = numpy.arange(12).reshape((3, 1, 4))

        print(numpy_extended_dot_matrix(
            a, b, axes=(0, ), left=(1,), right=(2,)))

        # Equivalent einsum equation
        print('equation', _numpy_extended_dot_equation(
            len(a.shape), len(a.shape), axes=(0, ), left=(1,), right=(2,)))

        # Same einsum computation written in a different way.
        print(numpy.einsum('kix,kxj->xij', a, b))
    """
    _common_check_numpy_extended_dot(m1, m2, axes, left, right)

    if verbose:
        print("[GENERICDOT] shape1=%r shape2=%r axes=%r "
              "left=%r right=%r -- %s" % (
                  m1.shape, m2.shape, axes, left, right,
                  _numpy_extended_dot_equation(
                      len(m1.shape), len(m1.shape), axes, left, right)))

    if len(axes) == 0 and len(set(left) & set(right)) == 0:
        # Simple multiplication
        res = m1 * m2
        if verbose:
            print("[GENERICDOT] Mul %r @ %r -> %r" % (
                m1.shape, m2.shape, res.shape))
        return res

    if (len(set(axes) & set(left)) == 0 and
            len(set(axes) & set(right)) == 0):

        # No intersection between axes and right: matrix multiplication
        # ReduceSum
        right_no_left = set(right) - (set(right) & (set(left) | set(axes)))
        if right_no_left:
            red1 = m1.sum(axis=tuple(sorted(right_no_left)), keepdims=True)
            if verbose:
                print("[GENERICDOT] reducesumL=%r, %r -> %r" % (
                    right_no_left, m1.shape, red1.shape))
        else:
            red1 = m1

        left_no_right = set(left) - (set(left) & (set(right) | set(axes)))
        if left_no_right:
            red2 = m2.sum(axis=tuple(sorted(left_no_right)), keepdims=True)
            if verbose:
                print("[GENERICDOT] reducesumR=%r, %r -> %r" % (
                    left_no_right, m2.shape, red2.shape))
        else:
            red2 = m2

        # Transpose
        common_axes = sorted(set(left) & set(right))
        i_axes = [(-1 if i in common_axes
                   else (1 if i in axes else 0), i)
                  for i in range(len(m1.shape))]
        i_axes.sort()
        perm = [_[1] for _ in i_axes]
        trm1 = numpy.transpose(red1, axes=perm)
        trm2 = numpy.transpose(red2, axes=perm)
        if verbose:
            print("[GENERICDOT] transposeL=%r, %r -> %r" % (
                perm, red1.shape, trm1.shape))
            print("[GENERICDOT] transposeR=%r, %r -> %r" % (
                perm, red2.shape, trm2.shape))
        final_shape = numpy_extended_dot_ouput_shape(
            m1, m2, axes, left, right)
        perm_left = [i for i in range(len(perm)) if perm[i] in left]
        perm_right = [i for i in range(len(perm)) if perm[i] in right]
        perm_common_axes = [i for i in range(len(perm))
                            if perm[i] in common_axes]

        if verbose:
            print("[GENERICDOT] MatMul %r @ %r -> %r  --  %s" % (
                m1.shape, m2.shape, final_shape,
                _numpy_extended_dot_equation(
                    len(m1.shape), len(m1.shape), axes, left, right)))
            print("[GENERICDOT] axes=%r left=%r right=%r" %
                  (axes, left, right))
            print("[GENERICDOT] perm=%r perm_left=%r "
                  "perm_right=%r perm_common_axes=%r" % (
                      perm, perm_left, perm_right, perm_common_axes))

        # Reshape
        dim0 = int(numpy.prod([trm1.shape[i] for i in perm_common_axes]))
        dim0b = int(numpy.prod([trm2.shape[i] for i in perm_common_axes]))
        if len(axes) > 0:
            all_axes = list(range(0, len(m1.shape)))
            new_axes = all_axes[-len(axes):]
        else:
            new_axes = []
        dim1 = int(numpy.prod([trm1.shape[i] for i in new_axes]))
        dim2 = int(numpy.prod([trm2.shape[i] for i in new_axes]))
        if dim1 != dim2:
            raise RuntimeError(  # pragma: no cover
                "Summation axis do not have the same length %d != %d, "
                "trshape1=%r trshape2=%r "
                "p_axes=%r p_left=%r p_right=%r p_common=%r"
                "." % (dim1, dim2, trm1.shape, trm2.shape,
                       new_axes, perm_left, perm_right, perm_common_axes))
        else:
            shm1 = trm1.reshape((dim0, -1, dim1))
            shm2 = trm2.reshape((dim0b, -1, dim2))

            if verbose:
                print("[GENERICDOT] Reshape %r @ %r -> %r @ %r" % (
                    (dim0, -1, dim1), (dim0, -1, dim2),
                    shm1.shape, shm2.shape))
                print("[GENERICDOT] matmul")

            # Multiplication (this should be done in a different way.
            res = shm1 @ numpy.transpose(shm2, axes=(0, 2, 1))

        if verbose:
            print("[GENERICDOT] Shape after multiplication %s" % (res.shape, ))

        # Transpose again
        not_in_both = []
        for i in range(0, len(m1.shape)):
            if i not in left and i not in right:
                not_in_both.append(i)
        ordered_axes = (common_axes +
                        list(i for i in left if i not in right) +
                        list(i for i in right if i not in left) +
                        not_in_both)

        perm_not_in_both = [i for i in range(len(perm))
                            if perm[i] in not_in_both]
        current_shape = ([max(trm1.shape[i], trm2.shape[i])
                          for i in sorted(perm_common_axes)] +
                         [trm1.shape[i] for i in sorted(perm_left)
                          if i not in perm_common_axes] +
                         [trm2.shape[i] for i in sorted(perm_right)
                          if i not in perm_common_axes] +
                         [1 for i in perm_not_in_both])

        if verbose:
            print("[GENERICDOT] current_shape=%r final_shape=%r "
                  "last_shape=%r" % (current_shape, final_shape, res.shape))

        if len(current_shape) != len(final_shape):
            raise RuntimeError(
                "Shapes mismatch %r > %r, "
                "shape1=%r shape2=%r axes=%r left=%r right=%r." % (
                    current_shape, final_shape,
                    m1.shape, m2.shape, axes, left, right))

        res = res.reshape(current_shape)

        perm = [(a, i) for i, a in enumerate(ordered_axes)]
        perm.sort()
        perm = [p[1] for p in perm]

        if verbose:
            print("[GENERICDOT] ordered_axes=%r perm=%r" % (
                ordered_axes, perm))

        return numpy.transpose(res, axes=perm)

    else:
        # Multiplication and Matrix multiplication at the same time.
        l_axes = set(left) & set(axes)
        r_axes = set(right) & set(axes)
        if r_axes and not l_axes:
            new_axes = list(a for a in axes if a not in right)
            new_left = list(sorted(set(left) | r_axes))
            if verbose:
                eq1 = _numpy_extended_dot_equation(
                    len(m1.shape), len(m1.shape), axes, left, right)
                eq2 = _numpy_extended_dot_equation(
                    len(m1.shape), len(m1.shape), new_axes, new_left, right)
                print("[GENERICDOT] replace left %r by %r axes %r by %r, "
                      "eq %r by %r" % (
                          left, new_left, axes, new_axes, eq1, eq2))
            return numpy_extended_dot_matrix(m1, m2, new_axes, new_left, right,
                                             verbose=verbose)
        raise RuntimeError(  # pragma: no cover
            "shape1=%r shape2=%r axes=%r left=%r right=%r eq=%s." % (
                m1.shape, m2.shape, axes, left, right,
                _numpy_extended_dot_equation(
                    len(m1.shape), len(m1.shape), axes, left, right)))
