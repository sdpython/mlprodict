"""
@file
@brief Function to dig into Einsum computation.
"""
import numpy


def numpy_extended_dot(m1, m2, axes, left, right, verbose=False):
    """
    Extended version of a matrix multiplication
    with two matrices *m1*, *m2* of the same dimension.
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

    The current implementation still uses :epkg:`numpy:einsum`
    but this should be replaced.
    """
    if len(m1.shape) != len(m2.shape):
        raise RuntimeError(
            "Matrices m1 and m2 must have the same dimension, "
            "m1=%r, m2=%r." % (m1.shape, m2.shape))
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


def analyse_einsum_equation(equation):
    """
    Analyses an einsum equation.

    :param equation: :epkg:`numpy:einsum` equation
    :return: three results, list of letters,
        a matrix (see below), lengths of each components

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
    :return: instance @see cl GraphEinsumSubOp

    About *strategy*:
    * `'simple'`: align all dimensions in the alphabetical order

    Available operations: *expand_dims*, *transpose*, *matmul*, *reduce_sum*,
    *id*, *squeeze*. It analyses an equation and produces a graph
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


class EinsumSubOp:
    """
    Defines a sub operation used in Einsum decomposition.

    :param name: name (reshape, transpose, reduce_sum, matmul, id)
    :param inputs: inputs
    :param kwargs: arguments
    """
    _allowed = {'expand_dims', 'transpose', 'reduce_sum', 'matmul', 'id',
                'squeeze'}

    def __init__(self, full_dim, name, *inputs, **kwargs):
        self.full_dim = full_dim
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
        self._check_()

    def _check_(self):
        if self.name == 'transpose':
            self._check_arg_('perm', tuple)
            perm = self.kwargs['perm']
            if len(perm) != len(set(perm)):
                raise RuntimeError(
                    "perm has duplicated values %r (name=%r)."
                    "" % (perm, self.name))

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
            self._check_arg_('left', tuple)
            self._check_arg_('right', tuple)
            if row2 is None:
                raise RuntimeError("matmul expects two inputs.")
            if verbose:
                axes = self.kwargs['axes']
                left = self.kwargs['left']
                right = self.kwargs['right']
                print("    MATMUL %r @ %r axes=%r left=%r right=%r" % (
                    row, row2, axes, left, right))
            row2[:] = numpy.maximum(row, row2)
            for a in self.kwargs['axes']:
                if a not in self.kwargs['right']:
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

    def _check_inputs_(self, n_expected, check_dim=False):
        if len(self.inputs) != n_expected:
            raise RuntimeError(
                "Number of inputs must be %d not %d for operator %r."
                "" % (n_expected, len(self.inputs), self.name))

    def _check_shape_(self, m):
        if len(m.shape) != self.full_dim:
            raise RuntimeError(
                "Number of dimensions %r is different from expected value "
                "%d." % (m.shape, self.full_dim))

    def _get_data(self, data, key):
        if isinstance(key, int):
            if key not in data:
                raise RuntimeError(
                    "Unable to find key %d in %r." % (
                        key, list(sorted(data))))
            return data[key]
        if isinstance(key, EinsumSubOp):
            if id(key) not in data:
                raise RuntimeError(
                    "Unable to find key %d in %r." % (
                        id(key), list(sorted(data))))
            return data[id(key)]
        raise TypeError(
            "Unexpected input type %r." % type(key))

    def apply(self, data, verbose=False):
        """
        Applies one operator on the data.

        :param data: dictionary storing the results
        """
        if verbose:
            print()
        if self.name == 'id':
            self._check_inputs_(1)
            inp = self.inputs[0]
            output = self._get_data(data, inp)

        elif self.name == 'expand_dims':
            self._check_inputs_(1)
            inp = self.inputs[0]
            m = self._get_data(data, inp)
            if verbose:
                print("- %s, shape=%r axis=%r" % (
                    self.name, m.shape, self.kwargs['axis']))
            output = numpy.expand_dims(m, self.kwargs['axis'][0])

        elif self.name == 'transpose':
            self._check_inputs_(1, True)
            inp = self.inputs[0]
            m = self._get_data(data, inp)
            self._check_shape_(m)
            if verbose:
                print("- %s, shape=%r perm=%r" % (
                    self.name, m.shape, self.kwargs['perm']))
            output = numpy.transpose(m, self.kwargs['perm'])
            self._check_shape_(output)

        elif self.name == 'matmul':
            self._check_inputs_(2)
            inp1 = self.inputs[0]
            inp2 = self.inputs[1]
            m1 = self._get_data(data, inp1)
            m2 = self._get_data(data, inp2)
            self._check_shape_(m1)
            self._check_shape_(m2)
            axes = self.kwargs['axes']
            left = self.kwargs['left']
            right = self.kwargs['right']

            if verbose:
                print("- %s, shapes=%r @ %r axes=%r left=%r right=%r" % (
                    self.name, m1.shape, m2.shape, axes, left, right))

            output = numpy_extended_dot(m1, m2, axes, left, right,
                                        verbose=verbose)
            self._check_shape_(output)

        elif self.name == 'reduce_sum':
            self._check_inputs_(1)
            inp = self.inputs[0]
            m = self._get_data(data, inp)
            self._check_shape_(m)
            axes = self.kwargs['axes']
            if verbose:
                print("- %s, shape=%r axes=%r" % (
                    self.name, m.shape, self.kwargs['axes']))
            output = numpy.sum(m, axis=axes, keepdims=True)
            self._check_shape_(output)

        elif self.name == 'squeeze':
            self._check_inputs_(1)
            inp = self.inputs[0]
            m = self._get_data(data, inp)
            axes = self.kwargs['axes']
            if verbose:
                print("- %s, shape=%r axes=%r" % (
                    self.name, m.shape, self.kwargs['axes']))
            output = m
            for a in axes[::-1]:
                output = numpy.squeeze(output, axis=a)
            return output

        else:
            raise NotImplementedError(
                "apply not implemented for %r." % self.name)

        data[id(self)] = output
        if verbose:
            print("+ %s, shape=%r -- %d" % (self.name, output.shape, id(self)))
        return output


class GraphEinsumSubOp:
    """
    Class gathering all nodes produced to explicit einsum
    operators.
    """

    def __init__(self, letters, mat, lengths):
        self._nodes = {}
        self._mark = {}
        self._ops = []
        self.last_op = None
        self.last_added_op = None
        self.metadata = dict(
            letters=letters, mat=mat, lengths=lengths,
            mat0=mat.copy())

    def append(self, op):
        """
        Adds one input or result.

        :param op: integer (an input) or an instance of @see cl EinsumSubOp.
        :return: op or None if op is an integer
        """
        if isinstance(op, int):
            if op in self._nodes:
                raise RuntimeError("Key %d already added." % op)
            self._nodes[op] = op
            self.last_added_op = op
            return None
        if isinstance(op, EinsumSubOp):
            if op in self._nodes:
                raise RuntimeError(
                    "Key %d already added, op=%r." % (id(op), op))
            self._nodes[id(op)] = op
            self._ops.append(op)
            self.last_added_op = op
            return op
        raise TypeError("Unexpected type %r." % type(op))

    def mark(self, i, op):
        """
        Marks one input or result as an intermediate result
        after a full einsum step.

        :param op: integer (an input) or an instance of @see cl EinsumSubOp.
        """
        if not isinstance(i, int):
            raise TypeError("i must an integer not %r." % type(i))
        if isinstance(op, EinsumSubOp):
            if id(op) not in self._nodes:
                raise RuntimeError(
                    "Key %d not found, op=%r." % (id(op), op))
            self._mark[i] = op
            self._mark[id(op)] = i
            self.last_op = op
        else:
            raise TypeError("Unexpected type %r." % type(i))

    def __iter__(self):
        "Iterates on nodes."
        for op in self._ops:
            yield op

    def to_dot(self, **kwargs):
        """
        Produces a graph in :epkg:`dot`.

        :param kwargs: additional graph option
        :return: string
        """
        options = {
            'orientation': 'portrait',
            'ranksep': '0.25',
            'nodesep': '0.05',
            'width': '0.5',
            'height': '0.1',
            'size': '5',
            'node': '[shape=record]',
        }
        options.update(kwargs)

        def d2s(d):
            it = []
            for k, v in sorted(d.items()):
                it.append("%s=%s" % (k, v))
            return " ".join(it)

        rows = ["digraph{"]
        for k, v in options.items():
            if isinstance(v, str) and "[" in v:
                rows.append("{} {};".format(k, v))
            else:
                rows.append("{}={};".format(k, v))
        for k, v in self._nodes.items():
            if isinstance(v, int):
                let = [(r, self.metadata['letters'][i])
                       for i, r in enumerate(self.metadata['mat0'][v])
                       if r != -1]
                let.sort()
                letters = "".join(_[1] for _ in let)
                lab = "input %d\\\\n%s\\\\n%s" % (
                    v, letters, str(self.metadata['mat0'][v]))
                sk = v
            else:
                lab = "%s\\\\n%s" % (v.name, d2s(v.kwargs))
                sk = id(v)
            if sk in self._mark and isinstance(self._mark[sk], int):
                la = self._mark[sk]
                lab = lab.replace("\\\\n", " - I%d\\\\n" % la)
                s = ('%d [label="%s" style=filled '
                     'fillcolor=red];' % (k, lab))
            else:
                s = '%d [label="%s"];' % (k, lab)
            rows.append(s)
            if not hasattr(v, 'inputs'):
                continue
            for i in v.inputs:
                vid = i if isinstance(i, int) else id(i)
                s = "%d -> %d;" % (vid, k)
                rows.append(s)
        rows.append("}")
        return "\n".join(rows)

    def apply_sequence(self, *inputs, verbose=False):
        """
        Applies a sequence of operations on a list of inputs.

        :param inputs: inputs:
        :return: output
        """
        if verbose:
            print('######### apply_sequence')
        data = {i: inp for i, inp in enumerate(inputs)}
        last = None
        for op in self:
            last = op.apply(data, verbose=verbose)
        if last is None:
            raise RuntimeError(
                "Sequence of operations is empty.")
        return last


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
    Applies strategy simple of function @see fct decompose_einsum_equation.
    """
    letters, mat, lengths = analyse_einsum_equation(equation)
    if len(letters) != mat.shape[1]:
        raise RuntimeError(  # pragma: no cover
            "Unexpected number of letters %r, shape=%r." % (
                letters, mat.shape))
    _basic_verification(lengths, shapes, equation)

    # last_row, current_row (row = shape)
    rows = numpy.full((2, mat.shape[1]), -1)
    graph = GraphEinsumSubOp(letters, mat, lengths)
    fd = mat.shape[1]
    if verbose:
        print("EQUATION=%r" % equation)
        print("LETTERS=%r" % letters, "LENGTHS=%r" % lengths)

    for i, sh in enumerate(shapes):
        if verbose:
            print()
            print("######### ROW %d shape=%r row=%r" % (i, sh, rows[1, :]))
        graph.append(i)

        # Input matrix aligned to the same dimensions.
        op = EinsumSubOp(fd, 'id', i)
        op.compute_output_row(rows[1, :], mat[i, :], verbose=verbose)
        marked = graph.append(op)

        for op in _apply_transpose_reshape(op, mat[i]):
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
                    common_dims.append(d)
                    if mat[i + 1:, d].max() >= 0:
                        left.append(d)
                        right.append(d)
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
                             left=tuple(left), right=tuple(right))
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
