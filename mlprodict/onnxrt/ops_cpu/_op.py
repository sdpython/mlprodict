# -*- encoding: utf-8 -*-
"""
@file
@brief Shortcut to *ops_cpu*.
"""
import pprint
import numpy
import onnx
import onnx.defs
from onnx import GraphProto
from ._new_ops import OperatorSchema


def _build_schemas():
    res = {}
    for schema in onnx.defs.get_all_schemas_with_history():
        # Multiple version can coexist. The last one is kept.
        if schema.name in res:
            if schema.since_version > res[schema.name].since_version:
                # We keep the most recent one.
                res[schema.name] = schema
        else:
            res[schema.name] = schema
        res[schema.name + '_' + str(schema.since_version)] = schema
    return res


_schemas = _build_schemas()
_at_least_one = {'Constant'}


class RuntimeTypeError(RuntimeError):
    """
    Raised when a type of a variable is unexpected.
    """
    pass


class DefaultNone:
    """
    Default value for parameters when the parameter is not set
    but the operator has a default behaviour for it.
    """
    pass


class RefAttrName:
    """
    Implements a link between a parameter of a function
    and an attribute in node.

    :param name: name of the input
    """

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        "usual"
        return f"{self.__class__.__name__}({self.name!r})"


class OpRun:
    """
    Ancestor to all operators in this subfolder.
    The runtime for every node can checked into
    `ONNX unit tests
    <https://github.com/onnx/onnx/tree/master/onnx/backend/test/case/node>`_.

    :param onnx_node: :epkg:`onnx` node
    :param desc: internal representation
    :param expected_attributes: expected attributes for this node
    :param options: runtime options
    """

    def __init__(self, onnx_node, desc=None, expected_attributes=None,
                 **options):
        self._provider = 'python'
        self.onnx_node = onnx_node
        self.desc = desc
        self.inplaces = {}

        if onnx_node.op_type in _schemas:
            self._schema = _schemas[onnx_node.op_type]
        else:
            self._schema = self._find_custom_operator_schema(onnx_node.op_type)
        if self._schema is None:
            raise RuntimeError(  # pragma: no cover
                "Unable to find class name '{}' in available schemas:"
                "(onnx.__version__='{}')\n{}".format(
                    self.__class__.__name__,
                    onnx.__version__,
                    "\n".join(sorted(_schemas))))

        if desc is not None:
            if 'atts' in desc:
                for a, b in desc['atts'].items():
                    if not isinstance(b, dict) or (
                            'value' not in b and 'ref_attr_name' not in b):
                        raise ValueError(  # pragma: no cover
                            f"Unexpected value {b}.")
                    if 'ref_attr_name' in b:
                        options[a] = RefAttrName(b['ref_attr_name'])
                    else:
                        options[a] = (b['value_rt'] if 'value_rt' in b
                                      else b['value'])
        if expected_attributes is not None:
            if onnx_node.op_type in _at_least_one:
                done = 0
                for a, b in expected_attributes.items():
                    if a in options:
                        setattr(self, a, b)
                        done += 1
                if done == 0:
                    raise RuntimeError(  # pragma: no cover
                        "All parameters '{}' are missing from operator '{}', "
                        "given {}.".format(
                            a, onnx_node.op_type, list(sorted(options))))
            else:
                for a, b in expected_attributes.items():
                    if a not in options:
                        if b is DefaultNone:
                            setattr(self, a, None)
                        elif b is None:
                            raise RuntimeError(  # pragma: no cover
                                "Parameter '{}' is missing from operator '{}' "
                                "(class='{}'), given {}.".format(
                                    a, onnx_node.op_type,
                                    self.__class__.__name__,
                                    list(sorted(options))))
                        else:
                            setattr(self, a, b)
        for k, v in options.items():
            setattr(self, k, v)

        if onnx_node.op_type not in _at_least_one:
            for k, v in self._schema.attributes.items():
                if not hasattr(self, k) and getattr(v, 'required', True):
                    raise RuntimeError(  # pragma: no cover
                        "Attribute '{}' is expected based on ONNX specifications "
                        "for node '{}' and options {}.".format(
                            k, onnx_node.op_type, pprint.pformat(options)))

    @staticmethod
    def local_inputs(graph):
        """
        Returns all varibles not registered as inputs and not produced by
        an node inside the graph. This inputs are part of the context
        existing in the graph calling this one.
        """
        if not isinstance(graph, GraphProto):
            raise TypeError(
                f"Unexpected type {type(graph)!r}.")
        local = set()
        known = set()
        for init in graph.initializer:
            known.add(init.name)
        for init in graph.input:
            known.add(init.name)
        for node in graph.node:
            for o in node.output:
                known.add(o)
            for i in node.input:
                if i not in known:
                    local.add(i)
        return list(local)

    def need_context(self):
        """
        Tells the runtime if this node needs the context
        (all the results produced so far) as it may silently access
        one of them (operator Loop).
        The default answer is `False`.
        """
        return False

    def _find_custom_operator_schema(self, op_name):
        raise NotImplementedError(  # pragma: no cover
            f"This method should be overwritten for operator '{op_name}'.")

    def __str__(self):
        """
        usual
        """
        atts = [self.__class__.__name__ + '(',
                f"    op_type={self.onnx_node.op_type}"]
        for k, v in sorted(self.__dict__.items()):
            if k in {'desc', 'onnx_node'}:
                continue
            if 'a' <= k[0] <= 'z' and k[-1] != '_':
                atts.append(f'    {k}={v},')
        atts.append(')')
        return "\n".join(atts)

    def _run(self, *args, **kwargs):
        """
        Should be overwritten.
        """
        raise NotImplementedError(  # pragma: no cover
            "Method '_run' or 'to_python' should be overwritten for operator %s."
            "" % self.__class__.__name__)

    def run(self, *args, **kwargs):  # pylint: disable=E0202
        """
        Calls method ``_run``.
        """
        try:
            res = self._run(*args, **kwargs)
        except TypeError as e:
            raise TypeError(  # pragma: no cover
                "Issues with types {} (operator {}).".format(
                    ", ".join(str(type(_)) for _ in args),
                    self.__class__.__name__)) from e
        except AttributeError as e:
            raise AttributeError(  # pragma: no cover
                "Issues with types {} (operator {}).".format(
                    ", ".join(str(type(_)) for _ in args),
                    self.__class__.__name__)) from e
        return res

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

    def enable_inplace_compute(self, index):
        """
        Tells the node that one input can be overwritten.

        @param      index       input index
        """
        self.inplaces[index] = True

    @property
    def args_default(self):
        """
        Returns the list of arguments as well as
        the list of parameters with the default values
        (close to the signature).
        """
        inps = []
        if hasattr(self, 'atts'):
            for k, v in self.atts.items():  # pylint: disable=E1101
                if isinstance(v, (list, tuple, dict)) and len(v) == 0:
                    v = None
                inps.append(f'{k}={v!r}')
        return inps

    @property
    def args_default_modified(self):
        """
        Returns the list of modified parameters.
        """
        if not hasattr(self, 'atts'):
            return None

        inps = []
        for k, v in self.atts.items():  # pylint: disable=E1101
            val = getattr(self, k, None)
            if isinstance(val, numpy.ndarray) and isinstance(v, list):
                val = list(val)
            try:
                if val != v:
                    inps.append(f'{k}={val!r}')
            except ValueError as e:  # pragma: no cover
                raise ValueError(
                    f"Unexpected value for v={v!r} and val={val!r}.") from e
        return inps

    @property
    def args_optional(self):
        """
        Returns the list of optional arguments.
        """
        inps = []
        if hasattr(self, 'optional_inputs'):
            for k, v in self.optional_inputs.items():  # pylint: disable=E1101
                inps.append(f'{k}={v!r}')
        return inps

    @property
    def args_mandatory(self):
        """
        Returns the list of optional arguments.
        """
        if hasattr(self, 'mandatory_inputs'):
            return self.mandatory_inputs  # pylint:  disable=E1101
        return None

    def to_python(self, inputs):
        """
        Returns a python code equivalent to this operator.

        @param      inputs      inputs name
        @return                 imports, python code, both as strings
        """
        raise NotImplementedError(
            f"Operator '{self.__class__.__name__}' has no equivalent python code.")  # pragma: no cover

    def _to_python_numpy(self, inputs, numpy_name):
        return ("import numpy",
                f"return numpy.{numpy_name}({', '.join(inputs)})")

    @property
    def atts_value(self):
        "Returns all parameters in a dictionary."
        if hasattr(self, 'atts'):
            return {k: getattr(self, k)
                    for k in self.atts}  # pylint: disable=E1101
        return None


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

    def run(self, x, attributes=None, verbose=0, fLOG=None):  # pylint: disable=E0202,W0221
        """
        Calls method ``_run``.
        """
        try:
            res = self._run(x, attributes=attributes,
                            verbose=verbose, fLOG=fLOG)
        except TypeError as e:
            raise TypeError(  # pragma: no cover
                "Issues with types {} (binary operator {}).".format(
                    ", ".join(str(type(_)) for _ in [x]),
                    self.__class__.__name__)) from e
        return res


class OpRunArg(OpRunUnary):
    """
    Ancestor to all unary operators in this subfolder
    and which produces position of extremas (ArgMax, ...).
    Checks that inputs type are the same.
    The class must have attributes *axis*, *keepdim*.
    """

    def __init__(self, onnx_node, desc=None, expected_attributes=None,
                 **options):
        OpRunUnary.__init__(self, onnx_node, desc=desc,
                            expected_attributes=expected_attributes,
                            **options)
        if not hasattr(self, 'keepdims'):
            raise AttributeError(  # pragma: no cover
                "Attribute 'keepdims' is missing.")
        if not hasattr(self, 'axis'):
            raise AttributeError(  # pragma: no cover
                "Attribute 'axis' is missing.")

    def run(self, x, attributes=None, verbose=0, fLOG=None):  # pylint: disable=E0202
        """
        Calls method ``_run``.
        """
        res = OpRunUnary.run(self, x, attributes=attributes,
                             verbose=verbose, fLOG=fLOG)
        if res[0].dtype != numpy.int64:
            raise RuntimeTypeError(  # pragma: no cover
                "Output type mismatch: should be '{}' != output '{}' "
                "(operator '{}')".format(
                    numpy.int64, res[0].dtype, self.__class__.__name__))
        return res

    def _run_no_checks_(self, x, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        return OpRunUnary.run(self, x, attributes=attributes, verbose=verbose, fLOG=fLOG)


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

    def run(self, x, attributes=None, verbose=0, fLOG=None):  # pylint: disable=E0202
        """
        Calls method ``_run``.
        """
        res = OpRunUnary.run(self, x, attributes=attributes,
                             verbose=verbose, fLOG=fLOG)
        if len(res) == 0 or res[0] is None:
            return res
        if not isinstance(res[0], list) and res[0].dtype != x.dtype:
            raise RuntimeTypeError(  # pragma: no cover
                "Output type mismatch: input '{}' != output '{}' "
                "(operator '{}')".format(
                    x.dtype, res[0].dtype, self.__class__.__name__))
        return res

    def _run_no_checks_(self, x, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        return OpRunUnary.run(self, x, attributes=attributes, verbose=verbose, fLOG=fLOG)


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

    def run(self, x, attributes=None, verbose=0, fLOG=None):  # pylint: disable=E0202
        """
        Calls method ``_run``.
        """
        res = OpRunUnary.run(self, x, attributes=attributes,
                             verbose=verbose, fLOG=fLOG)
        if x.dtype in (numpy.float32, numpy.float64) and res[1].dtype != x.dtype:
            raise RuntimeTypeError(  # pragma: no cover
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

    def _run_no_checks_(self, x, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        return OpRunUnary.run(self, x, attributes=attributes, verbose=verbose, fLOG=fLOG)


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

    def run(self, x, y, attributes=None, verbose=0, fLOG=None):  # pylint: disable=E0202,W0221
        """
        Calls method ``_run``.
        """
        if x is None or y is None:
            raise RuntimeError(  # pragma: no cover
                f"x and y have different dtype: {type(x)} != {type(y)} ({type(self)})")
        if x.dtype != y.dtype:
            raise RuntimeTypeError(
                "Input type mismatch: {} != {} (operator '{}', shapes {}, {})".format(
                    x.dtype, y.dtype, self.__class__.__name__,
                    x.shape, y.shape))
        try:
            res = self._run(x, y, attributes=attributes,
                            verbose=verbose, fLOG=fLOG)
        except (TypeError, ValueError) as e:  # pragma: no cover
            raise TypeError(
                "Issues with types {} (binary operator {}).".format(
                    ", ".join(str(type(_)) for _ in [x, y]),
                    self.__class__.__name__)) from e
        return res

    def _run_no_checks_(self, x, y, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        """
        Calls method ``_run``.
        """
        try:
            res = self._run(x, y, attributes=attributes,
                            verbose=verbose, fLOG=fLOG)
        except TypeError as e:  # pragma: no cover
            raise TypeError(
                "Issues with types {} (binary operator {}).".format(
                    ", ".join(str(type(_)) for _ in [x, y]),
                    self.__class__.__name__)) from e
        return res


class OpRunBinaryComparison(OpRunBinary):
    """
    Ancestor to all binary operators in this subfolder
    comparing tensors.
    """

    def __init__(self, onnx_node, desc=None, expected_attributes=None,
                 **options):
        OpRunBinary.__init__(self, onnx_node, desc=desc,
                             expected_attributes=expected_attributes,
                             **options)


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

    def run(self, x, y, attributes=None, verbose=0, fLOG=None):  # pylint: disable=E0202
        """
        Calls method ``_run``.
        """
        res = OpRunBinary.run(
            self, x, y, attributes=attributes, verbose=verbose, fLOG=fLOG)
        if res[0].dtype != x.dtype:
            raise RuntimeTypeError(
                "Output type mismatch: {} != {} or {} (operator '{}')"
                " type(x)={} type(y)={}".format(
                    x.dtype, res[0].dtype, y.dtype,
                    self.__class__.__name__, type(x), type(y)))
        return res

    def _run_no_checks_(self, x, y, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        """
        Calls method ``_run``.
        """
        return OpRunBinary._run_no_checks_(
            self, x, y, attributes=attributes, verbose=verbose, fLOG=fLOG)


class OpRunBinaryNumpy(OpRunBinaryNum):
    """
    Implements the inplaces logic.
    *numpy_fct* is a binary numpy function which
    takes two matrices and has a argument *out*
    for inplace operations.
    """

    def __init__(self, numpy_fct, onnx_node, desc=None,
                 expected_attributes=None, **options):
        OpRunBinaryNum.__init__(self, onnx_node, desc=desc,
                                expected_attributes=expected_attributes,
                                **options)
        self.numpy_fct = numpy_fct
        self._cannot_inplace_int = self.numpy_fct in (
            numpy.divide, numpy.true_divide)

    def _run(self, a, b, attributes=None, verbose=0, fLOG=None):  # pylint: disable=W0221
        if (self._cannot_inplace_int and
                numpy.issubdtype(a.dtype, numpy.integer)):
            return (self.numpy_fct(a, b), )
        if self.inplaces.get(0, False) and a.flags['WRITEABLE'] and a.size >= b.size:
            if len(a.shape) == 1 and b.shape == (1, 1):
                a = a.reshape(1, a.shape[0])
            try:
                self.numpy_fct(a, b, out=a)
                return (a, )
            except (ValueError, TypeError):
                return (self.numpy_fct(a, b), )
        if self.inplaces.get(1, False) and b.flags['WRITEABLE'] and a.size <= b.size:
            if len(b.shape) == 1 and a.shape == (1, 1):
                b = b.reshape(b.shape[0], 1)
            try:
                self.numpy_fct(a, b, out=b)
                return (b, )
            except (ValueError, TypeError):
                return (self.numpy_fct(a, b), )
        return (self.numpy_fct(a, b), )

    def to_python(self, inputs):
        """
        Returns a python code equivalent to this operator.

        @param      inputs      inputs name
        @return                 imports, python code, both as strings
        """
        lines = [
            "# inplaces not take into account {}-{}".format(
                self.inplaces.get(0, False), self.inplaces.get(1, False)),
            f"return numpy.{self.numpy_fct.__name__}({', '.join(inputs)})"
        ]
        return "import numpy", "\n".join(lines)


class OpRunReduceNumpy(OpRunUnaryNum):
    """
    Implements the reduce logic.
    It must have a parameter *axes*.
    """

    def __init__(self, onnx_node, desc=None,
                 expected_attributes=None, **options):
        if ('noop_with_empty_axes' not in expected_attributes and
                'axes' not in expected_attributes):
            raise RuntimeError(  # pragma: no cover
                "Parameter 'axes' is expected but not found in {} "
                "from class {}".format(expected_attributes, type(self)))
        if (expected_attributes.get('noop_with_empty_axes', 0) and
                (expected_attributes['axes'] is None or
                    len(expected_attributes['axes']) == 0)):
            raise RuntimeError(  # pragma: no cover
                "Parameter 'axes' cannot be empty as {} (noop_with_empty_axes=1) "
                "from class {}".format(expected_attributes, type(self)))
        OpRunUnaryNum.__init__(self, onnx_node, desc=desc,
                               expected_attributes=expected_attributes,
                               **options)
        if isinstance(self.axes, numpy.ndarray):  # pylint: disable=E0203
            if (len(self.axes.shape) == 0 or  # pylint: disable=E0203,E1101
                    self.axes.shape[0] == 0):  # pylint: disable=E0203,E1101
                self.axes = None
            else:
                self.axes = tuple(self.axes)
        elif self.axes in [[], tuple()]:  # pylint: disable=E0203
            self.axes = None
        elif isinstance(self.axes, list):  # pylint: disable=E0203
            self.axes = tuple(self.axes)


class OpRunCustom(OpRun):
    """
    Automates some methods for custom operators defined
    outside *mlprodict*.
    """

    class OpRunCustomSchema(OperatorSchema):
        """
        Custom schema.
        """

        def __init__(self, cls):
            OperatorSchema.__init__(self, cls.__name__)
            self.attributes = cls.atts

    def __init__(self, onnx_node, desc=None,
                 expected_attributes=None, **options):
        OpRun.__init__(self, onnx_node, desc=desc,
                       expected_attributes=expected_attributes,
                       **options)

    def _find_custom_operator_schema(self, op_name):
        """
        Finds a custom operator defined by this runtime.
        """
        if (op_name == self.__class__.__name__ or
                (hasattr(self.__class__, 'op_name') and
                    self.__class__.op_name == op_name)):  # pylint: disable=E1101
            return OpRunCustom.OpRunCustomSchema(self.__class__)
        raise RuntimeError(  # pragma: no cover
            f"Unable to find a schema for operator '{op_name}'.")
