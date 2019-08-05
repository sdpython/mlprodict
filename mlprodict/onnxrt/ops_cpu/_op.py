# -*- encoding: utf-8 -*-
"""
@file
@brief Shortcut to *ops_cpu*.
"""
import numpy
import onnx.defs
from ..shape_object import ShapeObject

_schemas = {
    schema.name: schema for schema in onnx.defs.get_all_schemas_with_history()}


class RuntimeTypeError(RuntimeError):
    """
    Raised when a type of a variable is unexpected.
    """
    pass


class OpRun:
    """
    Ancestor to all operators in this subfolder.
    The runtime for every node can checked into
    `ONNX unit tests
    <https://github.com/onnx/onnx/tree/master/onnx/backend/test/case/node>`_.
    """

    def __init__(self, onnx_node, desc=None, expected_attributes=None,
                 **options):
        """
        @param      onnx_node               :epkg:`onnx` node
        @param      desc                    internal representation
        @param      expected_attributes     expected attributes for this node
        @param      options                 runtime options
        """
        self._provider = 'python'
        self.onnx_node = onnx_node
        self.desc = desc
        if onnx_node.op_type in _schemas:
            self._schema = _schemas[onnx_node.op_type]
        else:
            self._schema = self._find_custom_operator_schema(onnx_node.op_type)
        if desc is not None:
            if 'atts' in desc:
                for a, b in desc['atts'].items():
                    if not isinstance(b, dict) or 'value' not in b:
                        raise ValueError("Unexpected value {}.".format(b))
                    options[a] = b['value_rt'] if 'value_rt' in b else b['value']
        if expected_attributes is not None:
            for a, b in expected_attributes.items():
                if a not in options:
                    if b is None:
                        raise RuntimeError("Parameter '{}' is missing from operator '{}', given {}.".format(
                            a, onnx_node.op_type, list(sorted(options))))
                    else:
                        setattr(self, a, b)
        for k, v in options.items():
            setattr(self, k, v)

        for k, v in self._schema.attributes.items():
            if not hasattr(self, k):
                raise RuntimeError("Attribute '{}' is expected based on ONNX specifications '{}'.".format(
                    k, v))

    def _find_custom_operator_schema(self, op_name):
        raise NotImplementedError(
            "This method should be overwritten for operator '{}'.".format(op_name))

    def __str__(self):
        """
        usual
        """
        atts = [self.__class__.__name__ + '(',
                "    op_type={}".format(self.onnx_node.op_type)]
        for k, v in sorted(self.__dict__.items()):
            if k in {'desc', 'onnx_node'}:
                continue
            if 'a' <= k[0] <= 'z' and k[-1] != '_':
                atts.append('    {0}={1}'.format(k, v))
        atts.append(')')
        return "\n".join(atts)

    def _run(self, *args, **kwargs):
        """
        Should be overwritten.
        """
        raise NotImplementedError("This method should be overwritten.")

    def run(self, *args, **kwargs):  # pylint: disable=E0202
        """
        Calls method ``_run``.
        """
        try:
            return self._run(*args, **kwargs)
        except TypeError as e:
            raise TypeError("Issues with types {} (operator {}).".format(
                ", ".join(str(type(_)) for _ in args),
                self.__class__.__name__)) from e

    def switch_initializers_dtype(self, dtype_in=numpy.float32,
                                  dtype_out=numpy.float64):
        """
        Switches all initializers to ``numpy.float64``. If *model*
        is None, a simple cast is done.

        @param      dtype_in    previous type
        @param      dtype_out   next type
        @return                 done operations
        """
        done = []
        for k, v in sorted(self.__dict__.items()):
            if k in {'desc', 'onnx_node'}:
                continue
            if isinstance(v, numpy.ndarray):
                if v.dtype == dtype_in:
                    v = v.astype(dtype_out)
                    setattr(self, k, v)
                    done.append(("+", "att", k, getattr(self, k)))
                else:
                    done.append(("-", "att", k, getattr(self, k)))
        if hasattr(self, '_run_no_checks_') and hasattr(self, 'run'):
            self.run = self._run_no_checks_  # pylint: disable=E0202,E1101
        return done

    def infer_shapes(self, *args, **kwargs):
        """
        Infer shapes of the output given the shapes
        of the input. It works the same way as method
        *run*.
        """
        try:
            return self._infer_shapes(*args, **kwargs)
        except TypeError as e:
            raise TypeError("Issues with types {} (operator {}).".format(
                ", ".join(str(type(_)) for _ in args),
                self.__class__.__name__)) from e

    def _infer_shapes(self, *args, **kwargs):
        """
        Should be overwritten.
        """
        raise NotImplementedError(
            "This method should be overwritten for operator '{}'.".format(
                self.__class__.__name__))


class OpRunUnary(OpRun):
    """
    Ancestor to all unary operators in this subfolder.
    Checks that inputs type are the same.
    """

    def __init__(self, onnx_node, desc=None, expected_attributes=None,
                 **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=expected_attributes,
                       **options)

    def run(self, x):  # pylint: disable=E0202,W0221
        """
        Calls method ``_run``.
        """
        try:
            res = self._run(x)
        except TypeError as e:
            raise TypeError("Issues with types {} (binary operator {}).".format(
                ", ".join(str(type(_)) for _ in [x]),
                self.__class__.__name__)) from e
        return res

    def infer_shapes(self, x):  # pylint: disable=E0202,W0221
        try:
            return self._infer_shapes(x)
        except TypeError as e:
            raise TypeError("Issues with types {} (operator {}).".format(
                x.dtype, self.__class__.__name__)) from e

    def _infer_shapes(self, x):  # pylint: disable=E0202,W0221
        """
        Returns the same shape by default.
        """
        return (x, )


class OpRunArg(OpRunUnary):
    """
    Ancestor to all unary operators in this subfolder
    and which produces posution of extremas.
    Checks that inputs type are the same.
    The class must have attributes *axis*, *keepdim*.
    """

    def __init__(self, onnx_node, desc=None, expected_attributes=None,
                 **options):
        OpRunUnary.__init__(self, onnx_node, desc=desc,
                            expected_attributes=expected_attributes,
                            **options)
        if not hasattr(self, 'keepdims'):
            raise AttributeError("Attribute 'keepdims' is missing.")
        if not hasattr(self, 'axis'):
            raise AttributeError("Attribute 'axis' is missing.")

    def run(self, x):  # pylint: disable=E0202
        """
        Calls method ``_run``.
        """
        res = OpRunUnary.run(self, x)
        if res[0].dtype != numpy.int64:
            raise RuntimeTypeError(
                "Output type mismatch: should be '{}' != output '{}' (operator '{}')".format(
                    numpy.int64, res[0].dtype, self.__class__.__name__))
        return res

    def _infer_shapes(self, x):  # pylint: disable=W0221
        """
        Returns the same shape by default.
        """
        sh = x.reduce(self.axis, self.keepdims,  # pylint: disable=E1101
                      dtype=numpy.int64)  # pylint: disable=E1101
        return (sh, )

    def _run_no_checks_(self, x):  # pylint: disable=W0221
        return OpRunUnary.run(self, x)


class OpRunUnaryNum(OpRunUnary):
    """
    Ancestor to all unary and numerical operators
    in this subfolder. Checks that inputs type
    are the same.
    """

    def __init__(self, onnx_node, desc=None, expected_attributes=None,
                 **options):
        OpRunUnary.__init__(self, onnx_node, desc=desc,
                            expected_attributes=expected_attributes,
                            **options)

    def run(self, x):  # pylint: disable=E0202
        """
        Calls method ``_run``.
        """
        res = OpRunUnary.run(self, x)
        if res[0].dtype != x.dtype:
            raise RuntimeTypeError(
                "Output type mismatch: input '{}' != output '{}' (operator '{}')".format(
                    x.dtype, res[0].dtype, self.__class__.__name__))
        return res

    def _infer_shapes(self, x):  # pylint: disable=W0221
        """
        Returns the same shape by default.
        """
        return (x, )

    def _run_no_checks_(self, x):  # pylint: disable=W0221
        return OpRunUnary.run(self, x)


class OpRunClassifierProb(OpRunUnary):
    """
    Ancestor to all binary operators in this subfolder.
    Checks that inputs type are the same.
    """

    def __init__(self, onnx_node, desc=None, expected_attributes=None,
                 **options):
        OpRunUnary.__init__(self, onnx_node, desc=desc,
                            expected_attributes=expected_attributes,
                            **options)

    def run(self, x):  # pylint: disable=E0202
        """
        Calls method ``_run``.
        """
        res = OpRunUnary.run(self, x)
        if res[1].dtype != x.dtype:
            raise RuntimeTypeError(
                "Output type mismatch: {} != {} (operator '{}')".format(
                    x.dtype, res[1].dtype, self.__class__.__name__))
        return res

    @property
    def nb_classes(self):
        """
        Returns the number of expected classes.
        """
        return max(len(getattr(self, 'classlabels_ints', [])),
                   len(getattr(self, 'classlabels_int64s', [])),
                   len(self.classlabels_strings))  # pylint: disable=E1101

    def _run_no_checks_(self, x):  # pylint: disable=W0221
        return OpRunUnary.run(self, x)

    def _infer_shapes(self, x):  # pylint: disable=W0221
        """
        Returns the same for the labels and the probabilities.
        """
        return (ShapeObject((x[0], ), dtype=numpy.int64),
                ShapeObject((x[0], self.nb_classes), dtype=x.dtype))


class OpRunBinary(OpRun):
    """
    Ancestor to all binary operators in this subfolder.
    Checks that inputs type are the same.
    """

    def __init__(self, onnx_node, desc=None, expected_attributes=None,
                 **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=expected_attributes,
                       **options)

    def run(self, x, y):  # pylint: disable=E0202,W0221
        """
        Calls method ``_run``.
        """
        if x.dtype != y.dtype:
            raise RuntimeTypeError(
                "Input type mismatch: {} != {} (operator '{}', shapes {}, {})".format(
                    x.dtype, y.dtype, self.__class__.__name__,
                    x.shape, y.shape))
        try:
            res = self._run(x, y)
        except TypeError as e:
            raise TypeError("Issues with types {} (binary operator {}).".format(
                ", ".join(str(type(_)) for _ in [x, y]),
                self.__class__.__name__)) from e
        return res

    def _run_no_checks_(self, x, y):  # pylint: disable=W0221
        """
        Calls method ``_run``.
        """
        try:
            res = self._run(x, y)
        except TypeError as e:
            raise TypeError("Issues with types {} (binary operator {}).".format(
                ", ".join(str(type(_)) for _ in [x, y]),
                self.__class__.__name__)) from e
        return res

    def _infer_shapes(self, x, y):  # pylint: disable=W0221
        """
        Returns the same shape by default.
        We assume the operator returns the biggest
        shapes as the operator could be using broacasting.
        """
        try:
            return (max(x, y), )
        except RuntimeError:
            # We know x and y and the same number of dimensions.
            # We pick the first one even if it might be wrong.
            return (x, )


class OpRunBinaryNum(OpRunBinary):
    """
    Ancestor to all binary operators in this subfolder.
    Checks that inputs type are the same.
    """

    def __init__(self, onnx_node, desc=None, expected_attributes=None,
                 **options):
        OpRunBinary.__init__(self, onnx_node, desc=desc,
                             expected_attributes=expected_attributes,
                             **options)

    def run(self, x, y):  # pylint: disable=E0202
        """
        Calls method ``_run``.
        """
        res = OpRunBinary.run(self, x, y)
        if res[0].dtype != x.dtype:
            raise RuntimeTypeError(
                "Output type mismatch: {} != {} (operator '{}')".format(
                    x.dtype, res[0].dtype, self.__class__.__name__))
        return res

    def _run_no_checks_(self, x, y):  # pylint: disable=W0221
        """
        Calls method ``_run``.
        """
        return OpRunBinary._run_no_checks_(self, x, y)
