"""
@file
@brief Function to dig into Einsum computation.
"""
import numpy
from .einsum_impl_ext import (
    numpy_extended_dot, numpy_diagonal,
    _numpy_extended_dot_equation,
    numpy_extended_dot_python,
    numpy_extended_dot_matrix)


class EinsumSubOp:
    """
    Defines a sub operation used in Einsum decomposition.

    :param name: name (reshape, transpose, reduce_sum, matmul, id,
        squeeze, diagonal, mul, batch_dot)
    :param inputs: inputs
    :param kwargs: arguments

    Operator suffixed by `_mm` (*transpose_mm*, *reduce_sum_mm*)
    are equivalent to the same operator without the suffix
    but takes two inputs and only changes the first one.
    """
    _allowed = {'expand_dims', 'transpose', 'reduce_sum', 'matmul', 'id',
                'squeeze', 'diagonal', 'mul', 'batch_dot',
                'transpose_mm', 'reduce_sum_mm'}

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
            if False and list(perm) == list(range(len(perm))):
                raise ValueError(
                    "Transpose = identity perm=%r. It must be removed."
                    "" % perm)
        elif self.name == 'matmul':
            self._check_arg_('axes', tuple)
            self._check_arg_('left', tuple)
            self._check_arg_('right', tuple)
            axes = self.kwargs['axes']
            left = self.kwargs['left']
            right = self.kwargs['right']
            for a in axes:
                if a in left and a in right:
                    raise RuntimeError(
                        "One axis belongs to every set (axes, left, right). "
                        "axes=%r, left=%r, right=%r." % (axes, left, right))

    def __repr__(self):
        inps = ", ".join(map(str, self.inputs))
        kw = ", ".join("%s=%r" % (k, w) for k, w in self.kwargs.items())
        m = "%s(%r, %s, %s)" % (
            self.__class__.__name__, self.name, inps, kw)
        return m

    def dot_label(self):
        """
        Displays some informations useful to understand the operator.
        """
        if self.name == "matmul":
            ndim = self.kwargs['ndim']
            axes = self.kwargs['axes']
            left = self.kwargs['left']
            right = self.kwargs['right']
            eq = _numpy_extended_dot_equation(ndim, ndim, axes, left, right)
            eq = eq.replace(">", "\\\\>")
            return "~" + eq
        return None

    def _check_arg_(self, name, typ, empty=False):
        if name not in self.kwargs:
            raise RuntimeError(
                "Parameter %r not found for operator %r." % (name, self.name))
        if empty and self.kwargs[name] is None:
            return
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
                print('<<' if inp else '>>', self.name, row, self.kwargs)
            else:
                print('<<' if inp else '>>', self.name, row)

    def _compute_output_row_id(self, row, row2=None, ab=False, verbose=False):
        if ab:
            raise RuntimeError("ab option not allowed.")
        self._check_row_(row, True, verbose=verbose)
        row[:] = row2[:]
        self._check_row_(row, verbose=verbose)

    def _compute_output_row_transpose(self, row, row2=None, ab=False, verbose=False):
        if ab:
            self._compute_output_row_transpose(row2, verbose=verbose)
            return
        self._check_row_(row, True, verbose=verbose)
        self._check_arg_('perm', tuple)
        if len(self.kwargs['perm']) != len(row):
            raise RuntimeError(
                "Unexpected permutation %r (row=%r)."
                "" % (self.kwargs['perm'], row))
        perm = self.kwargs['perm']
        cpy = row.copy()
        for i, p in enumerate(perm):
            row[i] = cpy[p]
        self._check_row_(row, verbose=verbose)

    def _compute_output_row_transpose_mm(self, row, row2=None, ab=False, verbose=False):
        if not ab:
            raise RuntimeError("ab must be True.")
        self._check_row_(row, True, verbose=verbose)
        if row2 is None:
            raise RuntimeError("transpose_mm expects a second input.")
        self._compute_output_row_transpose(row, row2=None, verbose=verbose)

    def _compute_output_row_expand_dims(self, row, row2=None, ab=False, verbose=False):
        if ab:
            raise RuntimeError("ab option not allowed.")
        self._check_row_(row, True, verbose=verbose)
        self._check_arg_('axis', tuple)
        if row[self.kwargs['axis'][1]] != -1:
            raise RuntimeError(
                "Dimension should be -1 in row %r axis=%r." % (
                    row, self.kwargs['axis']))
        self._check_row_(row, verbose=verbose)

    def _compute_output_row_reduce_sum(self, row, row2=None, ab=False, verbose=False):
        if ab:
            raise RuntimeError("ab option not allowed.")
        self._check_row_(row, True, verbose=verbose)
        self._check_arg_('axes', tuple)
        for a in self.kwargs['axes']:
            row[a] = -1
        self._check_row_(row, verbose=verbose)

    def _compute_output_row_reduce_sum_mm(self, row, row2=None, ab=False, verbose=False):
        if not ab:
            raise RuntimeError("ab must be true.")
        self._check_row_(row2, True, verbose=verbose)
        if row2 is None:
            raise RuntimeError("reduce_sum_mm expects a second input.")
        self._compute_output_row_reduce_sum(row2, row2=None, verbose=verbose)

    def _compute_output_row_squeeze(self, row, row2=None, ab=False, verbose=False):
        if ab:
            raise RuntimeError("ab option not allowed.")
        self._check_row_(row, True, verbose=verbose)
        self._check_arg_('axes', tuple)
        for a in self.kwargs['axes']:
            row[a] = -1
        self._check_row_(row, verbose=verbose)

    def _compute_output_row_diagonal(self, row, row2=None, ab=False, verbose=False):
        if ab:
            raise RuntimeError("ab option not allowed.")
        self._check_row_(row, True, verbose=verbose)
        self._check_arg_('diag', list)
        to_remove = []
        for choice, choices in self.kwargs['diag']:
            for ch in choices:
                if ch != choice:
                    to_remove.append(ch)
            for i in range(len(row)):  # pylint: disable=C0200
                if row[i] in choices:
                    if row[i] != choice:
                        row[i] = choice
        to_remove.sort()
        for r in to_remove:
            for i in range(len(row)):  # pylint: disable=C0200
                if row[i] == r:
                    raise RuntimeError(
                        "Unexpected result r=%r row=%r to_remove=%r "
                        "diag=%r." % (
                            r, row, to_remove, self.kwargs['diag']))
                if row[i] > r:
                    row[i] -= 1
        self._check_row_(row, verbose=verbose)

    def _compute_output_row_matmul(self, row, row2=None, ab=False, verbose=False):
        if not ab:
            raise RuntimeError("ab must be True.")
        self._check_row_(row, True, verbose=verbose)
        self._check_row_(row2, True, verbose=verbose)
        self._check_arg_('axes', tuple)
        self._check_arg_('left', tuple)
        self._check_arg_('right', tuple)
        self._check_arg_('ndim', int)
        if row2 is None:
            raise RuntimeError("matmul expects two inputs.")
        if verbose:
            ndim = self.kwargs['ndim']
            axes = self.kwargs['axes']
            left = self.kwargs['left']
            right = self.kwargs['right']
            print("    MATMUL %r @ %r axes=%r left=%r right=%r - eq=%s" % (
                row, row2, axes, left, right,
                _numpy_extended_dot_equation(ndim, ndim, axes, left, right)))
        row2[:] = numpy.maximum(row, row2)
        for a in self.kwargs['axes']:
            if a not in self.kwargs['right']:
                row2[a] = -1
        self._check_row_(row2, verbose=verbose)

    def _compute_output_row_batch_dot(self, row, row2=None, ab=False, verbose=False):
        if not ab:
            raise RuntimeError("ab must be True.")
        self._check_row_(row, True, verbose=verbose)
        self._check_row_(row2, True, verbose=verbose)
        self._check_arg_('batch_axes', tuple)
        self._check_arg_('keep_axes', tuple, empty=True)
        self._check_arg_('sum_axes', tuple)
        self._check_arg_('left', tuple)
        self._check_arg_('right', tuple)
        self._check_arg_('ndim', int)
        if row2 is None:
            raise RuntimeError("batch_dot expects two inputs.")
        if verbose:
            batch_axes = self.kwargs['batch_axes']
            keep_axes = self.kwargs['keep_axes']
            sum_axes = self.kwargs['sum_axes']
            left = self.kwargs['left']
            right = self.kwargs['right']
            ndim = self.kwargs['ndim']
            print("    BATCH_DOT batch_axes=%r keep_axes=%r sum_axes=%r "
                  "left=%r right=%r eq=%r" % (
                      batch_axes, keep_axes, sum_axes, left, right,
                      _numpy_extended_dot_equation(ndim, ndim, sum_axes, left, right)))
        row2[:] = numpy.maximum(row, row2)
        for a in self.kwargs['sum_axes']:
            if a not in self.kwargs['right']:
                row2[a] = -1
        self._check_row_(row2, verbose=verbose)

    def _compute_output_row_mul(self, row, row2=None, ab=False, verbose=False):
        if not ab:
            raise RuntimeError("ab must be True.")
        self._check_row_(row, True, verbose=verbose)
        self._check_row_(row2, True, verbose=verbose)
        if row2 is None:
            raise RuntimeError("mul expects two inputs.")
        if verbose:
            print("    MUL %r @ %r" % (row, row2))
        row2[:] = numpy.maximum(row, row2)
        self._check_row_(row2, verbose=verbose)

    def compute_output_row(self, row, row2=None, ab=False, verbose=False):
        """
        Updates *row* based on the operator.
        """
        method_name = "_compute_output_row_%s" % self.name
        meth = getattr(self, method_name, None)
        if meth is None:
            raise NotImplementedError(
                "compute_output_row not implemented for %r." % self.name)
        if verbose and ab:
            print("  -- called as a binary operator")
        meth(row, row2=row2, ab=ab, verbose=verbose)

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

    def _apply_id(self, data, verbose=False, **kwargs):
        self._check_inputs_(1)
        inp = self.inputs[0]
        output = self._get_data(data, inp)
        return output

    def _apply_diagonal(self, data, verbose=False, **kwargs):
        self._check_inputs_(1)
        inp = self.inputs[0]
        m = self._get_data(data, inp)
        if verbose:
            print("- %s, shape=%r diag=%r" % (
                self.name, m.shape, self.kwargs['diag']))
        diag = self.kwargs['diag']
        if len(diag) != 1:
            raise NotImplementedError(
                "Not implemented with more than one duplicated indice "
                "%r." % diag)
        diag0 = diag[0]
        output = numpy_diagonal(m, axis=diag0[0], axes=diag0[1])
        return output

    def _apply_expand_dims(self, data, verbose=False, **kwargs):
        self._check_inputs_(1)
        inp = self.inputs[0]
        m = self._get_data(data, inp)
        if verbose:
            print("- %s, shape=%r axis=%r" % (
                self.name, m.shape, self.kwargs['axis']))
        output = numpy.expand_dims(m, self.kwargs['axis'][0])
        return output

    def _apply_transpose(self, data, verbose=False, **kwargs):
        self._check_inputs_(1, True)
        inp = self.inputs[0]
        m = self._get_data(data, inp)
        self._check_shape_(m)
        if verbose:
            print("- %s, shape=%r perm=%r" % (
                self.name, m.shape, self.kwargs['perm']))
        output = numpy.transpose(m, self.kwargs['perm'])
        self._check_shape_(output)
        return output

    def _apply_transpose_mm(self, data, verbose=False, **kwargs):
        self._check_inputs_(2, True)
        inp = self.inputs[0]
        m = self._get_data(data, inp)
        self._check_shape_(m)
        if verbose:
            print("- %s, shape=%r perm=%r" % (
                self.name, m.shape, self.kwargs['perm']))
        output = numpy.transpose(m, self.kwargs['perm'])
        self._check_shape_(output)
        return output

    def _apply_matmul(self, data, verbose=False, **kwargs):
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

        impl = kwargs.get('matmul_impl', None)
        if impl == 'pyf':
            output = numpy_extended_dot_matrix(m1, m2, axes, left, right,
                                               verbose=verbose)
        elif impl == 'py':
            output = numpy_extended_dot_python(m1, m2, axes, left, right,
                                               verbose=verbose)
        elif impl is None:
            output = numpy_extended_dot(m1, m2, axes, left, right,
                                        verbose=verbose)
        else:
            raise ValueError(
                "Unknown implementation of numpy_extended_dot ({}).".format(impl))
        self._check_shape_(output)
        return output

    def _apply_mul(self, data, verbose=False, **kwargs):
        self._check_inputs_(2)
        inp1 = self.inputs[0]
        inp2 = self.inputs[1]
        m1 = self._get_data(data, inp1)
        m2 = self._get_data(data, inp2)
        self._check_shape_(m1)
        self._check_shape_(m2)

        if verbose:
            print("- %s, shapes=%r @ %r" % (self.name, m1.shape, m2.shape))

        output = m1 * m2
        self._check_shape_(output)
        return output

    def _apply_batch_dot(self, data, verbose=False, **kwargs):
        self._check_inputs_(2)
        inp1 = self.inputs[0]
        inp2 = self.inputs[1]
        m1 = self._get_data(data, inp1)
        m2 = self._get_data(data, inp2)
        self._check_shape_(m1)
        self._check_shape_(m2)
        batch_axes = self.kwargs['batch_axes']
        keep_axes = self.kwargs['keep_axes']
        sum_axes = self.kwargs['sum_axes']
        left = self.kwargs['left']
        right = self.kwargs['right']

        if verbose:
            print("- %s, shapes=%r @ %r batch_axes=%r keep_axes=%r "
                  "sum_axes=%r" % (
                      self.name, m1.shape, m2.shape, batch_axes, keep_axes, sum_axes))

        if len(m1.shape) != len(m2.shape):
            raise RuntimeError(
                "batch_dot only work with two tensors with the same number "
                "of dimensions not %r @ %r." % (m1.shape, m2.shape))

        dim0 = int(numpy.prod([m1.shape[i] for i in batch_axes]))
        dim0b = int(numpy.prod([m2.shape[i] for i in batch_axes]))
        dimb = int(-1 if keep_axes is None else numpy.prod(
            [m1.shape[i] for i in keep_axes]))
        dim1 = int(numpy.prod([m1.shape[i] for i in sum_axes]))
        dim2 = int(numpy.prod([m2.shape[i] for i in sum_axes]))

        m1sh = m1.reshape((dim0, dimb, dim1))
        m2sh = m2.reshape((dim0b, dimb, dim2))
        dot = m1sh @ numpy.transpose(m2sh, (0, 2, 1))

        # new shape
        taken = set(batch_axes) | set(sum_axes)
        ax = [i for i in range(len(m1.shape)) if i not in taken]
        new_shape = ([max(m1.shape[i], m2.shape[i]) for i in batch_axes] +
                     [m1.shape[i] for i in left if i not in batch_axes] +
                     [m2.shape[i] for i in right if i not in batch_axes])
        while len(new_shape) < len(m1.shape):
            new_shape.append(1)

        if verbose:
            print("- %s, shapes=%r @ %r -> %r" % (
                self.name, m1sh.shape, m2sh.shape, dot.shape))
            print("- %s, batch_axes=%r ax=%r new_shape=%r left=%r right=%r" % (
                self.name, batch_axes, ax, new_shape, left, right))

        output = dot.reshape(tuple(new_shape))
        self._check_shape_(output)
        return output

    def _apply_reduce_sum(self, data, verbose=False, **kwargs):
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
        return output

    def _apply_reduce_sum_mm(self, data, verbose=False, **kwargs):
        self._check_inputs_(2, True)
        inp = self.inputs[0]
        m = self._get_data(data, inp)
        self._check_shape_(m)
        if verbose:
            print("- %s, shape=%r axes=%r" % (
                self.name, m.shape, self.kwargs['axes']))
        output = numpy.sum(m, self.kwargs['axes'])
        self._check_shape_(output)
        return output

    def _apply_squeeze(self, data, verbose=False, **kwargs):
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

    def apply(self, data, verbose=False, **kwargs):
        """
        Applies one operator on the data.

        :param data: dictionary storing the results
        :param verbose: prints out intermediate results
        :param kwargs: additional parameters, see
            methods `_apply*`
        :return: output

        Known additional paramaters:
        * 'matmul_impl': if None calls :epkg:`numpy:einsum` through
          @see fn numpy_extended_dot (default) or 'py' to call
          @see fn numpy_extended_dot_python instead.
        """
        if verbose:
            print()
            print("apply %r  (%s)." % (
                self.name, ", ".join(map(lambda s: str(id(s)), self.inputs))))

        method_name = "_apply_%s" % self.name
        meth = getattr(self, method_name, None)
        if meth is None:
            raise NotImplementedError(
                "apply not implemented for %r." % self.name)
        output = meth(data, verbose, **kwargs)

        data[id(self)] = output
        if verbose:
            print("+ %s, shape=%r -- %d" % (self.name, output.shape, id(self)))
        return output


class GraphEinsumSubOp:
    """
    Class gathering all nodes produced to explicit einsum
    operators.
    """

    def __init__(self, letters, mat, lengths, duplicates):
        self._nodes = {}
        self._mark = {}
        self._ops = []
        self.last_op = None
        self.last_added_op = None
        self.metadata = dict(
            letters=letters, mat=mat, lengths=lengths,
            mat0=mat.copy(), duplicates=duplicates)

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

        def d2sd(d):
            it = []
            for k, v in sorted(d.items()):
                if len(v) > 1:
                    it.append("%s=%s" % (k, ",".join(map(str, v))))
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
                dup = self.metadata['duplicates'][v]
                if dup is None:
                    dup = ""
                else:
                    dup = " - %s" % d2sd(dup)
                let.sort()
                letters = "".join(_[1] for _ in let)
                lab = "input %d\\\\n%s\\\\n%s%s" % (
                    v, letters, str(self.metadata['mat0'][v]), dup)
                sk = v
                extended_lab = ""
            else:
                lab = "%s\\\\n%s" % (v.name, d2s(v.kwargs))
                sk = id(v)
                extended_lab = v.dot_label()
                if extended_lab:
                    extended_lab = "\\\\n" + extended_lab

            if sk in self._mark and isinstance(self._mark[sk], int):
                la = self._mark[sk]
                lab = lab.replace("\\\\n", " - I%d\\\\n" % la)
                s = ('%d [label="%s%s" style=filled '
                     'fillcolor=red];' % (k, lab, extended_lab))
            else:
                s = '%d [label="%s%s"];' % (k, lab, extended_lab)
            rows.append(s)
            if not hasattr(v, 'inputs'):
                continue
            for i in v.inputs:
                vid = i if isinstance(i, int) else id(i)
                s = "%d -> %d;" % (vid, k)
                rows.append(s)
        rows.append("}")
        return "\n".join(rows)

    def apply_sequence(self, *inputs, verbose=False, **kwargs):
        """
        Applies a sequence of operations on a list of inputs.

        :param inputs: inputs:
        :param verbose: prints out intermediate results
        :param kwargs: additional parameters,
            see :meth:`apply
            <mlprodict.testing.einsum_impl_classes.EinsumSubOp.apply>`.
        :return: output
        """
        if verbose:
            print('######### apply_sequence')
        data = {i: inp for i, inp in enumerate(inputs)}
        last = None
        for op in self:
            last = op.apply(data, verbose=verbose, **kwargs)
        if last is None:
            raise RuntimeError(
                "Sequence of operations is empty.")
        return last
