"""
@file
@brief Function to dig into Einsum computation.
"""
import numpy


def analyse_einsum_equation(equation):
    """
    Analyses an einsum equation.

    :param equation: :epkg:`numpy:einsum` equation
    :return:
    """
    spl = equation.strip(' ,').split("->")
    if len(spl) != 2 or len(spl[1]) == 0 or len(spl[0]) == 0:
        raise NotImplementedError(
            "The function only implements the case when there are "
            "two sides in the equation: %r." % equation)
    inputs = list(map(lambda s: s.strip(), spl[0].split(',')))
    for inp in inputs:
        if len(inp) != len(set(inp)):
            raise NotImplementedError(
                "One input uses more than once the same indice %r in "
                "equation %r." % (inp, equation))
    output = spl[1]
    all_letters = set(inputs[0])
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
    return "".join(letters), mat, lengths


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
    :return: sequence of operations of typ @see cl EinsumSubOp

    About *strategy*:
    * `'simple'`: align all dimensions in the alphabetical order

    Available operations: *expand_dims*, *transpose*, *matmul*, *reduce_sum*,
    *id*, *squeeze*.
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


def apply_sequence(seq, *inputs, verbose=False):
    """
    Applies a sequence of operations on a list of inputs.

    :param seq: sequence of operations
    :param inputs: inputs:
    :return: output
    """
    data = {i: inp for i, inp in enumerate(inputs)}
    for op in seq:
        op.apply(data)
    return data[id(seq[-1])]


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


class EinsumSubOp:
    """
    Defines a sub operation used in Einsum decomposition.

    :param name: name (reshape, transpose, reduce_sum, matmul, id)
    :param inputs: inputs
    :param kwargs: arguments
    """
    _allowed = {'expand_dims', 'transpose', 'reduce_sum', 'matmul', 'id',
                'squeeze'}

    def __init__(self, name, *inputs, **kwargs):
        self.name = name
        self.inputs = inputs
        self.kwargs = kwargs
        if name not in EinsumSubOp._allowed:
            raise ValueError(
                "Unexpected name %r. It should be in %r."
                "" % (name, EinsumSubOp._allowed))
        if len(inputs) not in (1, 2):
            raise RuntimeError(
                "Inputs must contains 1 or 2 inputs not %d." % len(inputs))
        if name == 'matmul' and len(inputs) != 2:
            raise RuntimeError(
                "Inputs must contains 2 inputs not %d for operator 'matmul'."
                "" % len(inputs))
        for i, inp in enumerate(inputs):
            if not isinstance(inp, (int, EinsumSubOp)):
                raise TypeError(
                    "Input %d has type %r, int or EinsumSubOp is expected."
                    "" % (i, type(inp)))

    def __repr__(self):
        inps = ", ".join(map(str, self.inputs))
        kw = ", ".join("%s=%r" % (k, w) for k, w in self.kwargs.items())
        m = "%s(%r, %s, %s)" % (
            self.__class__.__name__, self.name, inps, kw)
        return m

    def _check_arg_(self, name, typ):
        if name not in self.kwargs:
            raise RuntimeError(
                "Parameter %r not found for operator %r." % (name, self.name))
        if not isinstance(self.kwargs[name], typ):
            raise TypeError(
                "Unexpected type %r for parameter %r and parameter %r."
                "" % (type(self.kwargs[name]), name, self.name))

    def _check_row_(self, row, inp=False, verbose=False):
        """
        Checks input or output is valid.
        """
        if verbose:
            if inp:
                print()
            print('<-' if inp else '->', self.name, row, self.kwargs)
        if not inp or self.name != 'id':
            if row.max() == -1:
                raise RuntimeError(  # pragma: no cover
                    "Shape is empty %r." % row)

    def compute_output_row(self, row, row2=None, verbose=False):
        """
        Updates *row* based on the operator.
        """
        self._check_row_(row, True, verbose=verbose)

        if self.name == "id":
            row[:] = row2[:]
            self._check_row_(row, verbose=verbose)
            return

        if self.name == "transpose":
            self._check_arg_('perm', tuple)
            if len(self.kwargs['perm']) != len(row):
                raise RuntimeError(
                    "Unexpected permutation %r (row=%r)."
                    "" % (self.kwargs['perm'], row))
            cpy = row.copy()
            for i, p in enumerate(self.kwargs['perm']):
                row[i] = cpy[p]
            self._check_row_(row, verbose=verbose)
            return

        if self.name == "expand_dims":
            self._check_arg_('axis', tuple)
            if row[self.kwargs['axis'][1]] != -1:
                raise RuntimeError(
                    "Dimension should be -1 in row %r axis=%r." % (
                        row, self.kwargs['axis']))
            self._check_row_(row, verbose=verbose)
            return

        if self.name == "reduce_sum":
            self._check_arg_('axes', tuple)
            for a in self.kwargs['axes']:
                row[a] = -1
            self._check_row_(row, verbose=verbose)
            return

        if self.name == "matmul":
            self._check_arg_('axes', tuple)
            if row2 is None:
                raise RuntimeError("matmul expects two inputs.")
            if verbose:
                print("    MATMUL %r @ %r" % (row, row2))
            row2[:] = numpy.maximum(row, row2)
            for a in self.kwargs['axes']:
                row2[a] = -1
            self._check_row_(row2, verbose=verbose)
            return

        if self.name == "squeeze":
            self._check_arg_('axes', tuple)
            for a in self.kwargs['axes']:
                row[a] = -1
            self._check_row_(row, verbose=verbose)
            return

        raise NotImplementedError(
            "compute_output_row not implemented for %r." % self.name)

    def _check_inputs_(self, n_expected):
        if len(self.inputs) != n_expected:
            raise RuntimeError(
                "Number of inputs must be %d not %d for operator %r."
                "" % (n_expected, len(self.inputs), self.name))

    def _get_data(self, data, key):
        if isinstance(key, int):
            return data[key]
        if isinstance(key, EinsumSubOp):
            return data[id(key)]
        raise TypeError(
            "Unexpected input type %r." % type(key))

    def apply(self, data):
        """
        Applies one operator on the data.

        :param data: dictionary storing the results
        """
        if self.name == 'id':
            self._check_inputs_(1)
            inp = self.inputs[0]
            return self._get_data(data, inp)

        if self.name == 'expand_dims':
            self._check_inputs_(1)

        raise NotImplementedError(
            "apply not implemented for %r." % self.name)


def _apply_transpose_reshape(op, row):
    """
    Put all dimensions in the same order.
    """
    axes = []
    p = 0
    perm = []
    for i, r in enumerate(row):
        if r == -1:
            axes.append((p, i))
            perm.append(-1)
        else:
            p += 1
            perm.append(r)
    for a in reversed(axes):
        op = EinsumSubOp('expand_dims', op, axis=a)
        yield op
    dec = [0]
    for i in range(1, len(perm)):
        if perm[i - 1] == -1:
            dec.append(dec[-1] + 1)
        else:
            dec.append(dec[-1])
    for i in range(0, len(perm)):  # pragma: disable=C0200
        if perm[i] == -1:
            perm[i] = i
        else:
            perm[i] = perm[i] + dec[i]
    op = EinsumSubOp('transpose', op, perm=tuple(perm))
    yield op


def _apply_squeeze_transpose(op, row_last, row_output):
    """
    Put output dimension in the expected order.
    """

    perm = []
    sq = []
    for i, d in enumerate(row_output):
        if d == -1:
            perm.append((i, i))
            sq.append(i)
        else:
            perm.append((d, i))
    perm = [p[1] for p in perm]
    op = EinsumSubOp('transpose', op, perm=tuple(perm))
    yield op
    if len(sq) > 0:
        op = EinsumSubOp('squeeze', op, axes=tuple(sq))
        yield op


def _decompose_einsum_equation_simple(equation, *shapes, verbose=False):
    """
    Applies strategy simple of function @see fct decompose_einsum_equation.
    """
    letters, mat, lengths = analyse_einsum_equation(equation)
    if len(letters) != mat.shape[1]:
        raise RuntimeError(  # pragma: no cover
            "Unexpected number of letters %r, shape=%r." % (
                letters, mat.shape))
    _basic_verification(lengths, shapes, equation)
    ops = []
    # last_row, current_row (row = shape)
    rows = numpy.full((2, mat.shape[1]), -1)
    last_op = None
    if verbose:
        print("EQUATION=%r" % equation)

    for i, sh in enumerate(shapes):
        if verbose:
            print()
            print("######### ROW %d shape=%r row=%r" % (i, sh, rows[1, :]))

        # Input matrix aligned to the same dimensions.
        op = EinsumSubOp('id', i)
        op.compute_output_row(rows[1, :], mat[i, :], verbose=verbose)
        ops.append(op)
        for op in _apply_transpose_reshape(op, mat[i]):
            op.compute_output_row(rows[1, :], verbose=verbose)
            ops.append(op)

        # Reduction? (a dimension not used later)
        red = []
        for d in range(0, mat.shape[1]):
            if (mat[i + 1:, d].max() == -1 and rows[1, d] != -1 and
                    rows[0, d] == -1):
                red.append(d)
        if len(red) > 0:
            if verbose:
                print("-- REDUCE1 row=%d axes=%r" % (i, red))
                print(mat)
                print('-')
                print(rows)
            op = EinsumSubOp('reduce_sum', ops[-1], axes=tuple(red))
            op.compute_output_row(rows[1, :], verbose=verbose)

        if last_op is not None:
            # Matrix multiplication?
            common_dims = []
            for d in range(0, mat.shape[1]):
                if rows[:, d].min() >= 0:
                    common_dims.append(d)
            if verbose:
                print("-- MATMUL common_dims=%r" % common_dims)
                print(rows)
            if len(common_dims) > 0:
                op = EinsumSubOp('matmul', last_op, op,
                                 axes=tuple(common_dims))
                ops.append(op)
                op.compute_output_row(rows[0, :], rows[1, :], verbose=verbose)
            else:
                raise NotImplementedError(
                    "Unable to interpret equation %r at position %i "
                    "(starting at 0) common_dims=%r rows-=%r rows+=%r."
                    "" % (equation, i, common_dims, rows[0, :], rows[1, :]))

        # End
        rows[0, :] = rows[1, :]
        last_op = op

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
                    "Issue in equation %r, last_result is %r, "
                    "output is %r." % (equation, rows[0, :], rows[1, :]))
        if len(red) > 0:
            if verbose:
                print("-- REDUCE2 axes=%r" % red)
                print(mat)
            op = EinsumSubOp('reduce_sum', op, axes=red)
            ops.append(op)
            op.compute_output_row(rows[1, :], verbose=verbose)

        # Final transpose and reshape if needed
        for op in _apply_squeeze_transpose(op, rows[1, :], mat[len(shapes), :]):
            op.compute_output_row(rows[1, :], verbose=verbose)
            ops.append(op)
    return ops
