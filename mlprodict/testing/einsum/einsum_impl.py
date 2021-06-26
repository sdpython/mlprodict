"""
@file
@brief Main functions decomposing einsum computation into
more simple functions.
"""
import numpy
from .einsum_impl_classes import EinsumSubOp, GraphEinsumSubOp


def analyse_einsum_equation(equation):
    """
    Analyses an einsum equation.

    :param equation: :epkg:`numpy:einsum` equation
    :return: three results, list of letters,
        a matrix (see below), lengths of each components,
        duplicates

    The returned a matrix is defined as follows:

    .. math::

        m_{ij}=\\left\\{\\begin{array}{ll}-1 &
        \\text{if letter j is involved in input i} \\\\
        p & \\text{p is position of letter j in equation i}
        \\end{array}\\right.
    """
    spl = equation.strip(' ,').split("->")
    if len(spl) != 2 or len(spl[1]) == 0 or len(spl[0]) == 0:
        raise NotImplementedError(
            "The function only implements the case when there are "
            "two sides in the equation: %r." % equation)
    inputs = list(map(lambda s: s.strip(), spl[0].split(',')))
    output = spl[1]
    all_letters = set(inputs[0])

    # Set of letters
    for inp in inputs[1:]:
        all_letters |= set(inp)
    letters = list(sorted(all_letters))
    for c in letters:
        if not(('a' <= c <= 'z') or ('A' <= c <= 'Z')):
            raise ValueError(
                "Equation %r must only contain lower or upper letters "
                "but %r is not." % (equation, c))

    rev = {c: i for i, c in enumerate(letters)}
    for c in output:
        if c not in letters:
            raise ValueError(
                "Output contains one unexpected letter %r in "
                "equation %r." % (c, equation))
    mat = numpy.full((len(inputs) + 1, len(letters)), -1, dtype=numpy.int8)
    for i, inp in enumerate(inputs):
        for k, c in enumerate(inp):
            mat[i, rev[c]] = k
    for k, c in enumerate(output):
        mat[len(inputs), rev[c]] = k
    lengths = [len(inp) for inp in inputs]
    lengths.append(len(output))

    # Look for duplicates
    duplicates = []
    for inp in inputs + [output]:
        if len(inp) == len(set(inp)):
            duplicates.append(None)
            continue
        # There is some duplicates.
        counts = {}
        for i, c in enumerate(inp):
            if c in counts:
                counts[c].append(i)
            else:
                counts[c] = [i]
        duplicates.append(counts)

    return "".join(letters), mat, lengths, duplicates


def decompose_einsum_equation(equation, *shapes, strategy="simple",
                              clean=False, verbose=False):
    """
    Decomposes an equation used in :epkg:`numpy:einsum` knowing
    the input shapes. It returns a sequence of operations
    to do to compute the results.

    :param equation: a string
    :param shapes: sequence of input shapes
    :param strategy: there are different way to decompose the equation,
        this parameters defines the way to do it (see below)
    :param clean: clean the unnecessary node in the graph
    :param verbose: verbosity
    :return: instance of @see cl GraphEinsumSubOp

    About *strategy*:
    * `'simple'`: align all dimensions in the alphabetical order,
      some generic matrix multiplication remains implemented with
      :epkg:`numpy:einsum` but only with two matrices aligned on
      the same dimension (see @see fn numpy_extended_dot)
    * `'numpy'`: same as `simple` but the decomposition does not use
      :epkg:`numpy:einsum` anymore but only multiplication or
      matrix multiplication merged into a single operator called
      *batch_dot* (see @see fn numpy_extended_dot_matrix)

    Available operations: *expand_dims*, *transpose*, *matmul*, *reduce_sum*,
    *id*, *squeeze*, *diagonal*. It analyses an equation and produces a graph
    where node are instance of class @see cl EinsumSubOp.

    .. runpython::
        :showcode:

        from mlprodict.testing.einsum import decompose_einsum_equation
        seq = decompose_einsum_equation("bac,cd,def->ebc")
        for op in seq:
            print(op)

    It can be better displayed as the following.

    .. gdot::
        :script: DOT-SECTION
        :process:

        from mlprodict.testing.einsum import decompose_einsum_equation
        seq = decompose_einsum_equation(
            "bac,cd,def->ebc", (2, 2, 2), (2, 2), (2, 2, 2))
        print("DOT-SECTION", seq.to_dot())

    See notebook :ref:`einsumdecompositionrst`.
    """
    if len(shapes) > 0:
        for sh in shapes:
            if not isinstance(sh, tuple):
                raise TypeError(
                    "All shapes must be tuples for %r is not." % sh)
    if strategy in ("simple", "numpy"):
        op_matmul = {'simple': 'matmul',
                     'numpy': 'batch_dot'}
        graph = _decompose_einsum_equation_simple(
            equation, *shapes, verbose=verbose, op_matmul=op_matmul[strategy])
    else:
        raise ValueError("Unknown strategy %r." % strategy)

    # Last step: clean unused nodes.
    if clean:
        last_node = graph.last_added_op
        graph.append(EinsumSubOp(last_node.full_dim, 'id', last_node))
        graph.mark_last_node()
        graph.simplify_mm_nodes(verbose=verbose)
        graph.remove_duplicate_transpose(verbose=verbose)
        graph.clean_unused_nodes(verbose=verbose)
    else:
        graph.mark_last_node()
    return graph


def apply_einsum_sequence(seq, *inputs, verbose=False, **kwargs):
    """
    Applies a sequence of operations on a list of inputs.
    The sequence of operations is produced by function
    @see fn decompose_einsum_equation.

    :param seq: sequence of operations
    :param inputs: inputs
    :param kwargs: additional parameters,
        see :meth:`apply_sequence
        <mlprodict.testing.einsum.einsum_impl_classes.
        GraphEinsumSubOp.apply_sequence>`.
    :return: output

    .. runpython::
        :showcode:

        import numpy
        from mlprodict.testing.einsum import (
            decompose_einsum_equation, apply_einsum_sequence)

        m1 = numpy.arange(2 * 2 * 2).reshape((2, 2, 2)) + 10
        m2 = numpy.arange(4).reshape((2, 2)) + 100
        m3 = numpy.arange(8).reshape((2, 2, 2)) + 1000

        seq = decompose_einsum_equation("bac,cd,def->ebc")
        res = apply_einsum_sequence(seq, m1, m2, m3)
        print(res)

    See notebook :ref:`einsumdecompositionrst`.
    """
    return seq.apply_sequence(*inputs, verbose=verbose, **kwargs)


def is_transpose_identity(perm):
    """
    Tells if the permutation *perm* does nothing (itentity).

    :param perm: permutation
    :return: boolean
    """
    return list(perm) == list(range(len(perm)))


def _basic_verification(lengths, shapes, equation):
    if len(lengths) - 1 != len(shapes):
        raise ValueError(
            "Equation %r has %d inputs but %d shapes are given."
            "" % (equation, len(lengths), len(shapes)))
    for i, (le, sh) in enumerate(zip(lengths, shapes)):
        if le != len(sh):
            raise ValueError(
                "Inputs %d has %d dimensions but shapes %r has %d "
                " in equation %r." % (i, le, sh, len(sh), equation))


def _apply_transpose_reshape(op, row):
    """
    Put all dimensions in the same order.

    :param op: integer (for one input) or an operator
    :param row: letter involved in this input (as a vector of binaries)
    :return: last created operator
    """
    axes = []
    p = 0
    perm = []
    for i, r in enumerate(row):
        if r == -1:
            axes.append((p, i))
        else:
            p += 1
            perm.append((r, i))
    op = EinsumSubOp(len(row), 'expand_dims', op, axes=tuple(axes))
    yield op
    perm.sort()
    p = 0
    new_perm = numpy.arange(len(row))
    for i, r in enumerate(row):
        if r == -1:
            continue
        new_perm[perm[p][1]] = i
        p += 1
    if not is_transpose_identity(new_perm):
        op = EinsumSubOp(len(row), 'transpose', op, perm=tuple(new_perm))
        yield op


def _apply_squeeze_transpose(op, row_last, row_output):
    """
    Puts output dimension in the expected order.
    """
    perm = []
    sq = []
    for i, d in enumerate(row_output):
        if d == -1:
            sq.append(i)
        else:
            perm.append((d, i))
    perm.sort()
    new_perm = numpy.arange(len(row_last))
    p = 0
    for i, d in enumerate(row_output):
        if d == -1:
            continue
        new_perm[i] = perm[p][1]
        p += 1
    perm = [p[1] for p in perm]
    if not is_transpose_identity(new_perm):
        op = EinsumSubOp(len(row_last), 'transpose', op,
                         perm=tuple(new_perm))
        yield op
    if len(sq) > 0:
        op = EinsumSubOp(len(row_last), 'squeeze', op, axes=tuple(sq))
        yield op


def _apply_einsum_matmul(fd, op1, op2, axes, left, right, ndim,
                         op_matmul, row1, row2, verbose=False):
    """
    Decomposes the generic matrix multiplication into numpy operations
    depending on the operator to use for matrix multiplication
    *op_matmul* (see @see fn decompose_einsum_equation).
    """
    allowed = {'matmul', 'batch_dot', 'dot'}
    if op_matmul not in allowed:
        raise ValueError(  # pragma: no cover
            "Unknown operator op_matmul=%r not in %r." % (op_matmul, allowed))
    if op_matmul == 'matmul':
        if verbose:  # pragma: no cover
            print("  -- MATMUL -> matmul axes=%r left=%r right=%r"
                  "" % (axes, left, right))
        yield EinsumSubOp(fd, 'matmul', op1, op2,
                          axes=axes, left=left, right=right, ndim=ndim)

    elif len(axes) == 0 and len(set(left) & set(right)) == 0:
        if verbose:  # pragma: no cover
            print("  -- MATMUL -> mul axes=%r left=%r right=%r"
                  "" % (axes, left, right))
        yield EinsumSubOp(fd, 'mul', op1, op2)

    elif (len(set(axes) & set(left)) == 0 and
            len(set(axes) & set(right)) == 0):

        # No intersection between axes and right: matrix multiplication
        if verbose:  # pragma: no cover
            print("  -- MATMUL -> batch_dot axes=%r left=%r right=%r"
                  "" % (axes, left, right))

        all_axes = set(left) | set(right) | set(axes)
        common_axes = list(set(left) & set(right))
        for i in range(ndim):
            if i not in all_axes:
                common_axes.append(i)
        common_axes.sort()

        # ReduceSum*
        has_dim = set(i for i in range(len(row1)) if row1[i] >= 0)
        right_no_left = (set(right) & has_dim) - \
            (set(right) & (set(left) | set(axes)))
        if right_no_left:
            if verbose:  # pragma: no cover
                print('  -- MATMUL reduce1 has_dim=%r axes=%r' %
                      (has_dim, right_no_left))
            op1 = EinsumSubOp(fd, 'reduce_sum_mm', op1, op2,
                              axes=tuple(sorted(right_no_left)))
            yield op1

        has_dim = set(i for i in range(len(row2)) if row2[i] >= 0)
        left_no_right = (set(left) & has_dim) - \
            (set(left) & (set(right) | set(axes)))
        if left_no_right:
            if verbose:  # pragma: no cover
                print('  -- MATMUL reduce2 has_dim=%r axes=%r' %
                      (has_dim, left_no_right))
            op2 = EinsumSubOp(fd, 'reduce_sum', op2,
                              axes=tuple(sorted(left_no_right)))
            yield op2

        # Transpose
        i_axes = [(-1 if i in common_axes
                   else (1 if i in axes else 0), i)
                  for i in range(ndim)]
        i_axes.sort()
        perm = [_[1] for _ in i_axes]
        perm_left = [i for i in range(len(perm)) if perm[i] in left]
        perm_right = [i for i in range(len(perm)) if perm[i] in right]
        if not is_transpose_identity(perm):
            op1 = EinsumSubOp(fd, 'transpose_mm', op1, op2, perm=tuple(perm))
            yield op1
            op2 = EinsumSubOp(fd, 'transpose', op2, perm=tuple(perm))
            yield op2

        # Reshape
        all_axes = list(range(0, ndim))
        new_axes = all_axes[-len(axes):] if len(axes) > 0 else []
        new_common_axes = all_axes[:len(common_axes)]
        not_in_both = []
        for i in range(0, ndim):
            if i not in left and i not in right and i not in common_axes:
                not_in_both.append(i)

        op = EinsumSubOp(fd, 'batch_dot', op1, op2,
                         batch_axes=tuple(new_common_axes),
                         keep_axes=None, sum_axes=tuple(new_axes),
                         left=tuple(perm_left), right=tuple(perm_right),
                         ndim=ndim)
        yield op

        # Transpose again
        ordered_axes = (common_axes +
                        list(i for i in left if i not in right) +
                        list(i for i in right if i not in left) +
                        not_in_both)
        rev_perm = [(a, i) for i, a in enumerate(ordered_axes)]
        rev_perm.sort()
        rev_perm = [p[1] for p in rev_perm]

        if not is_transpose_identity(rev_perm):
            op_unused = EinsumSubOp(fd, 'transpose_mm', op1,
                                    op, perm=tuple(rev_perm))
            yield op_unused
            op = EinsumSubOp(fd, 'transpose', op, perm=tuple(rev_perm))
            yield op
    else:
        raise NotImplementedError(  # pragma: no cover
            "axes and right or left have axes in common, "
            "axes=%r left=%r right=%r ndim=%r." % (
                axes, left, right, ndim))


def _decompose_einsum_equation_simple(equation, *shapes, verbose=False,
                                      op_matmul='matmul'):
    """
    Applies strategy `simple`, `numpy`
    defined in by function @see fn decompose_einsum_equation.

    :param op_matmul: which operator to use for matrix multiplication,
        a single operator *matmul*, or *batch_dot* with *transposes*,
        *reduce_sum*, or just *dot*
    """
    letters, mat, lengths, duplicates = analyse_einsum_equation(equation)
    if len(letters) != mat.shape[1]:
        raise RuntimeError(  # pragma: no cover
            "Unexpected number of letters %r, shape=%r." % (
                letters, mat.shape))
    if len(shapes) == 0:
        shapes = [(2, ) * le for le in lengths[:-1]]
    _basic_verification(lengths, shapes, equation)

    # last_row, current_row (row = shape)
    rows = numpy.full((2, mat.shape[1]), -1)
    graph = GraphEinsumSubOp(letters, mat, lengths, duplicates)
    fd = mat.shape[1]
    if verbose:
        print("EQUATION=%r" % equation)
        print("LETTERS=%r" % letters, "LENGTHS=%r" % lengths)
        print("DUPLICATES=%r" % duplicates)

    for i, sh in enumerate(shapes):
        if verbose:
            print()
            print("######### ROW %d shape=%r row=%r" % (i, sh, rows[1, :]))
        graph.append(i)

        # Input matrix aligned to the same dimensions.
        op = EinsumSubOp(fd, 'id', i)
        op.compute_output_row(rows[1, :], mat[i, :], verbose=verbose)
        marked = graph.append(op)

        duplicate = duplicates[i]
        if duplicate is not None:
            # Diagonal
            diag = []
            for _, v in duplicate.items():
                if len(v) == 1:
                    continue
                diag.append((v[0], tuple(v)))
            op = EinsumSubOp(fd, 'diagonal', op, diag=diag)
            op.compute_output_row(rows[1, :], mat[i, :], verbose=verbose)
            tr_row = rows[1, :]
            marked = graph.append(op)
        else:
            diag = None
            tr_row = mat[i]

        for op in _apply_transpose_reshape(op, tr_row):
            op.compute_output_row(rows[1, :], verbose=verbose)
            marked = graph.append(op)

        # Reduction? (a dimension not used later)
        red = []
        for d in range(0, mat.shape[1]):
            if (mat[i + 1:, d].max() == -1 and rows[1, d] != -1 and
                    rows[0, d] == -1):
                red.append(d)
        if len(red) > 0:
            if verbose:
                print("  -- REDUCE1 row=%d axes=%r" % (i, red))
                print(mat)
                print('  -')
                print(rows)
            op = EinsumSubOp(fd, 'reduce_sum',
                             graph.last_added_op, axes=tuple(red))
            op.compute_output_row(rows[1, :], verbose=verbose)
            marked = graph.append(op)

        if graph.last_op is not None:
            # Matrix multiplication?
            common_dims = []
            left = []
            right = []
            for d in range(0, mat.shape[1]):
                if rows[:, d].min() >= 0:
                    if mat[i + 1:, d].max() >= 0:
                        left.append(d)
                        right.append(d)
                    else:
                        common_dims.append(d)
                else:
                    if rows[0, d] >= 0:
                        left.append(d)
                    if rows[1, d] >= 0:
                        right.append(d)
            if verbose:
                print("  -- MATMUL common_dims=%r" % common_dims)
                print(rows)
            for iop in _apply_einsum_matmul(
                    fd, graph.last_op, op, axes=tuple(common_dims),
                    left=tuple(left), right=tuple(right),
                    ndim=rows.shape[1], op_matmul=op_matmul,
                    row1=rows[0, :], row2=rows[1, :], verbose=verbose):
                op = iop
                op.compute_output_row(rows[0, :], rows[1, :],
                                      ab=True, verbose=verbose)
                marked = graph.append(op)

        # End
        graph.mark(i, marked)
        rows[0, :] = rows[1, :]

    # Final output
    if verbose:
        print()
        print("######### FIN row=%r" % rows[1, :])

    if mat[len(shapes), :].max() >= 0:
        rows[1, :] = mat[len(shapes), :]
        red = []
        for d in range(0, mat.shape[1]):
            if rows[0, d] > 0 and rows[1, d] == -1:
                red.append(d)
            elif rows[0, d] == -1 and rows[1, d] >= 0:
                raise RuntimeError(  # pragma: no cover
                    "Issue in equation %r, variable %d, last_result is %r, "
                    "output is %r." % (equation, d, rows[0, :], rows[1, :]))
        if len(red) > 0:
            if verbose:  # pragma: no cover
                print("-- REDUCE2 axes=%r" % red)
                print(mat)
            op = EinsumSubOp(fd, 'reduce_sum', op, axes=tuple(red))
            graph.append(op)
            op.compute_output_row(rows[1, :], verbose=verbose)

        # Removes empty axes.
        for op in _apply_squeeze_transpose(op, rows[1, :], mat[len(shapes), :]):
            op.compute_output_row(rows[1, :], verbose=verbose)
            graph.append(op)
    return graph
