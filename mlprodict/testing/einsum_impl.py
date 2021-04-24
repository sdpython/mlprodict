"""
@file
@brief Function to dig into Einsum computation.
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


def decompose_einsum_equation(equation, *shapes, strategy="simple", verbose=False):
    """
    Decomposes an equation used in :epkg:`numpy:einsum` knowing
    the input shapes. It returns a sequence of operations
    to do to compute the results.

    :param equation: a string
    :param shapes: sequence of input shapes
    :param strategy: there are different way to decompose the equation,
        this parameters defines the way to do it (see below)
    :param verbose: verbosity
    :return: instance of @see cl GraphEinsumSubOp

    About *strategy*:
    * `'simple'`: align all dimensions in the alphabetical order

    Available operations: *expand_dims*, *transpose*, *matmul*, *reduce_sum*,
    *id*, *squeeze*, *diagonal*. It analyses an equation and produces a graph
    where node are instance of class @see cl EinsumSubOp.

    .. runpython::
        :showcode:

        from mlprodict.testing.einsum_impl import decompose_einsum_equation
        seq = decompose_einsum_equation(
            "bac,cd,def->ebc", (2, 2, 2), (2, 2), (2, 2, 2))
        for op in seq:
            print(op)

    It can be better displayed as the following.

    .. gdot::
        :script: DOT-SECTION
        :process:

        from mlprodict.testing.einsum_impl import decompose_einsum_equation
        seq = decompose_einsum_equation(
            "bac,cd,def->ebc", (2, 2, 2), (2, 2), (2, 2, 2))
        print("DOT-SECTION", seq.to_dot())

    See notebook :ref:`einsumdecompositionrst`.
    """
    if len(shapes) == 0:
        raise ValueError("No input shapes.")
    for sh in shapes:
        if not isinstance(sh, tuple):
            raise TypeError(
                "All shapes must be tuples for %r is not." % sh)
    if strategy == "simple":
        return _decompose_einsum_equation_simple(equation, *shapes, verbose=verbose)
    raise ValueError("Unknown strategy %r." % strategy)


def apply_einsum_sequence(seq, *inputs, verbose=False):
    """
    Applies a sequence of operations on a list of inputs.
    The sequence of operations is produced by function
    @see fn decompose_einsum_equation.

    :param seq: sequence of operations
    :param inputs: inputs:
    :return: output

    .. runpython::
        :showcode:

        from mlprodict.testing.einsum_impl import (
            decompose_einsum_equation, apply_einsum_sequence)

        m1 = numpy.arange(2 * 2 * 2).reshape((2, 2, 2)) + 10
        m2 = numpy.arange(4).reshape((2, 2)) + 100
        m3 = numpy.arange(2 * 2).reshape((2, 2)) + 1000

        seq = decompose_einsum_equation(
            "bac,cd,def->ebc", (2, 2, 2), (2, 2), (2, 2, 2))
        res = apply_einsum_sequence(seq, m1, m2, verbose=verbose)
        print(res)

    See notebook :ref:`einsumdecompositionrst`.
    """
    return seq.apply_sequence(*inputs, verbose=verbose)


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
    for a in reversed(axes):
        op = EinsumSubOp(len(row), 'expand_dims', op, axis=a)
        yield op
    perm.sort()
    p = 0
    new_perm = numpy.arange(len(row))
    for i, r in enumerate(row):
        if r == -1:
            continue
        new_perm[perm[p][1]] = i
        p += 1
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
    op = EinsumSubOp(len(row_last), 'transpose', op, perm=tuple(new_perm))
    yield op
    if len(sq) > 0:
        op = EinsumSubOp(len(row_last), 'squeeze', op, axes=tuple(sq))
        yield op


def _decompose_einsum_equation_simple(equation, *shapes, verbose=False):
    """
    Applies strategy simple of function @see fn decompose_einsum_equation.
    """
    letters, mat, lengths, duplicates = analyse_einsum_equation(equation)
    if len(letters) != mat.shape[1]:
        raise RuntimeError(  # pragma: no cover
            "Unexpected number of letters %r, shape=%r." % (
                letters, mat.shape))
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
            op = EinsumSubOp(fd, 'matmul', graph.last_op, op,
                             axes=tuple(common_dims),
                             left=tuple(left), right=tuple(right),
                             ndim=rows.shape[1])
            op.compute_output_row(rows[0, :], rows[1, :], verbose=verbose)
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
                raise RuntimeError(
                    "Issue in equation %r, variable %d, last_result is %r, "
                    "output is %r." % (equation, d, rows[0, :], rows[1, :]))
        if len(red) > 0:
            if verbose:
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
